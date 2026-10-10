#!/usr/bin/env python3
"""
Nemotron-Mini-4B Trading Fine-Tune — 4-GPU Balanced device_map (v3)
2026-07-15

4× RX 6600 XT (8GB each) = 32GB total VRAM.
Model fp16 = ~8.4GB → split across 4 GPUs with MANUAL device_map for balanced utilization.

FIX: device_map='auto' caused GPU0 at 38-56% while GPU1-3 at 99%.
Manual split: 32 decoder layers → 8 per GPU (layers 0-7 → GPU0, 8-15 → GPU1, 16-23 → GPU2, 24-31 → GPU3).

CRITICAL: Do NOT set PYTORCH_HIP_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF —
they cause hipErrorIllegalAddress on ROCm 7.2 + PyTorch 2.12!

Uses PEFT LoRA (works on multi-GPU when alloc conf is NOT set).
Uses HF SFTTrainer for standard training loop.
"""

import os
import sys
import json
import time
import signal
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# ── ROCm environment (set BEFORE importing torch) ──
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_ENABLE_SDMA'] = '0'
# NOTE: PYTORCH_HIP_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF CRASHES on ROCm 7.2!
# Do NOT set expandable_segments:True or max_split_size_mb — causes hipErrorIllegalAddress
os.environ['HIP_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # ALL 4 GPUs

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# ── Config ──
MODEL_NAME = 'nvidia/Nemotron-Mini-4B-Instruct'
OUTPUT_DIR = '/home/nodeadmin/trading-llm/output/nemotron-trader-v3'
TRAIN_FILE = '/home/nodeadmin/trading-llm/training-v11/train_balanced.jsonl'
VAL_FILE = '/home/nodeadmin/trading-llm/training-v11/val_balanced.jsonl'

CHUNK_STEPS = int(os.environ.get('CHUNK_STEPS', '100'))
MAX_SEQ = 384
BATCH = 1
GRAD_ACCUM = 8
LR = 2e-4
LORA_R = 16  # Reference only — actual adapter uses r=8
LORA_ALPHA = 32  # Reference only — actual adapter uses alpha=16
SAVE_STEPS = 25
EVAL_STEPS = 9999

NUM_LAYERS = 32  # Nemotron-Mini-4B has 32 decoder layers
LAYERS_PER_GPU = NUM_LAYERS // 4  # 8 layers per GPU

_training_state = {'trainer': None}

def signal_handler(signum, frame):
    print(f'\n[SIGNAL] Received signal {signum}, saving emergency checkpoint...', flush=True)
    if _training_state['trainer']:
        _training_state['trainer'].save_model(f'{OUTPUT_DIR}/emergency-checkpoint')
    start_ollama()
    sys.exit(130)

def stop_ollama():
    print('[PREP] Stopping Ollama to free GPU VRAM...', flush=True)
    subprocess.run(['sudo', 'systemctl', 'stop', 'ollama'], capture_output=True)
    print('[PREP] Waiting 10s for GPU cleanup...', flush=True)
    time.sleep(10)
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if 'GPU[0]' in line and 'Used' in line:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    vram_gb = int(p) / 1e9
                    if vram_gb > 2.0:
                        print(f'[PREP] ❌ ABORT: GPU 0 VRAM {vram_gb:.2f}GB still used', flush=True)
                        start_ollama()
                        sys.exit(1)
                    print(f'[PREP] ✅ GPU 0 VRAM free: {vram_gb:.2f}GB used', flush=True)
                    return

def start_ollama():
    print('[POSTP] Starting Ollama...', flush=True)
    subprocess.run(['sudo', 'systemctl', 'start', 'ollama'], capture_output=True)
    print('[POSTP] Ollama restarted.', flush=True)

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f'[DATA] Loaded {len(data)} samples from {filepath}', flush=True)
    return data

def format_dataset(data, tokenizer, max_seq):
    formatted = []
    for item in data:
        if 'messages' in item:
            text = tokenizer.apply_chat_template(item['messages'], tokenize=False)
        elif 'text' in item:
            text = item['text']
        elif 'prompt' in item and 'response' in item:
            text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['response']}"
        else:
            continue
        formatted.append({"text": text})
    return Dataset.from_list(formatted)

def build_balanced_device_map():
    """
    Build a manual device_map for Nemotron-Mini-4B-Instruct (32 decoder layers).
    Split evenly across 4 GPUs: 8 layers each.

    NemotronForCausalLM structure:
      self.model (NemotronModel) contains: embed_tokens, rotary_emb, layers, norm
      self.lm_head

    Layer assignments:
      GPU 0: model.embed_tokens, model.model.rotary_emb, model.layers.0-7
      GPU 1: model.layers.8-15
      GPU 2: model.layers.16-23
      GPU 3: model.layers.24-31, model.norm, lm_head
    """
    device_map = {}

    # Embeddings and rotary → GPU 0 (note: model.model.* for NemotronModel submodules)
    device_map['model.embed_tokens'] = 0

    # 32 layers → 8 per GPU
    for i in range(NUM_LAYERS):
        gpu_id = i // LAYERS_PER_GPU  # 0-7→0, 8-15→1, 16-23→2, 24-31→3
        device_map[f'model.layers.{i}'] = gpu_id

    # Final norm + LM head → GPU 3
    device_map['model.norm'] = 3
    device_map['lm_head'] = 3

    return device_map

def inspect_model_layers():
    """Inspect model architecture BEFORE loading weights to verify layer names and count."""
    global NUM_LAYERS, LAYERS_PER_GPU
    print('\n' + '=' * 70, flush=True)
    print('[INSPECT] Loading model config to inspect architecture...', flush=True)
    print('=' * 70, flush=True)

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f'[INSPECT] Model type: {config.model_type}', flush=True)
    print(f'[INSPECT] Num hidden layers: {config.num_hidden_layers}', flush=True)
    print(f'[INSPECT] Hidden size: {config.hidden_size}', flush=True)
    print(f'[INSPECT] Vocab size: {config.vocab_size}', flush=True)

    expected_layers = NUM_LAYERS  # 32 for Nemotron-Mini-4B
    if config.num_hidden_layers != expected_layers:
        print(f'[INSPECT] ⚠️  WARNING: Expected {expected_layers} layers but config shows {config.num_hidden_layers}!', flush=True)
        print(f'[INSPECT] Adjusting layer split to {config.num_hidden_layers} layers / 4 GPUs', flush=True)
        NUM_LAYERS = config.num_hidden_layers
        LAYERS_PER_GPU = max(1, NUM_LAYERS // 4)

    # Build and print the device_map
    device_map = build_balanced_device_map()
    print(f'\n[INSPECT] Balanced device_map ({NUM_LAYERS} layers, {LAYERS_PER_GPU} per GPU):', flush=True)
    print('-' * 50, flush=True)

    # Group by GPU for nice display
    gpu_groups = {0: [], 1: [], 2: [], 3: []}
    for module, gpu in sorted(device_map.items()):
        gpu_groups[gpu].append(module)

    for gpu in range(4):
        modules = gpu_groups[gpu]
        layers = [m for m in modules if 'layers.' in m]
        others = [m for m in modules if 'layers.' not in m]
        layer_range = f'layers {min(int(l.split(".")[-1]) for l in layers)}-{max(int(l.split(".")[-1]) for l in layers)}' if layers else 'none'
        print(f'  GPU {gpu}: {len(modules)} modules ({layer_range}, +{others})', flush=True)

    print('-' * 50, flush=True)
    print(f'[INSPECT] Total mapped modules: {len(device_map)}', flush=True)

    return device_map

def load_model_with_device_map(device_map):
    """Load model with manual device_map using from_pretrained directly."""
    print('\n[STEP 1] Loading model with manual balanced device_map...', flush=True)

    # Load model directly with device_map — from_pretrained + low_cpu_mem_usage handles efficient loading
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    print(f'[INFO] Model loaded! {model.num_parameters()/1e6:.1f}M params', flush=True)
    print(f'[INFO] Device map: {model.hf_device_map}', flush=True)

    # Verify shards are on the right GPUs
    print(f'\n[VERIFY] Checking model shard placement...', flush=True)
    gpu_params = {0: 0, 1: 0, 2: 0, 3: 0}
    for name, param in model.named_parameters():
        if hasattr(param, 'device') and param.device.type == 'cuda':
            gpu_id = param.device.index
            gpu_params[gpu_id] += param.numel()
    for gpu_id in range(4):
        pct = 100 * gpu_params[gpu_id] / max(1, sum(gpu_params.values()))
        print(f'  GPU {gpu_id}: {gpu_params[gpu_id]/1e6:.1f}M params ({pct:.1f}%)', flush=True)

    return model

def verify_gpu_shards():
    """Run rocm-smi to verify all 4 GPUs have model shards loaded."""
    print('\n[GPU-MON] rocm-smi output after model loading:', flush=True)
    print('=' * 70, flush=True)
    result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    print('=' * 70, flush=True)

def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print('=' * 70, flush=True)
    print('Nemotron-Mini-4B Trading Fine-Tune — 4-GPU Balanced device_map (ROCm 7.2)', flush=True)
    print('v3: Manual layer split (8 layers/GPU) for balanced utilization', flush=True)
    print('=' * 70, flush=True)

    stop_ollama()

    # ── Inspect model architecture BEFORE loading ──
    device_map = inspect_model_layers()

    print(f'\n[DATA] Loading tokenizer: {MODEL_NAME}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'[DATA] Loading datasets...', flush=True)
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)
    train_formatted = format_dataset(train_data, tokenizer, MAX_SEQ)
    val_formatted = format_dataset(val_data, tokenizer, MAX_SEQ)
    print(f'[DATA] Train: {len(train_formatted)} | Val: {len(val_formatted)}', flush=True)

    # ── Load model with balanced device_map ──
    model = load_model_with_device_map(device_map)

    # ── Verify GPU shards with rocm-smi ──
    verify_gpu_shards()

    print(f'\n[STEP 2] Loading existing LoRA adapter from AcerNitro training...', flush=True)
    ADAPTER_PATH = '/home/nodeadmin/trading-llm/adapters/nemotron-mini-trader/final-adapter'
    if not Path(ADAPTER_PATH).exists():
        print(f'[ERROR] Adapter path does not exist: {ADAPTER_PATH}', flush=True)
        start_ollama()
        sys.exit(1)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'[INFO] Existing LoRA loaded! Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)', flush=True)
    print(f'[INFO] Adapter: r=8, alpha=16, target=q_proj+v_proj (from AcerNitro training)', flush=True)
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    # Resume from checkpoint
    resume_from = None
    checkpoint_dirs = sorted(Path(OUTPUT_DIR).glob('checkpoint-*'))
    if checkpoint_dirs:
        resume_from = str(checkpoint_dirs[-1])
        print(f'[INFO] Resuming from {resume_from}', flush=True)

    print(f'\n[STEP 3] Starting training (chunks of {CHUNK_STEPS} steps)...', flush=True)
    print(f'[INFO] BATCH={BATCH}, GRAD_ACCUM={GRAD_ACCUM}, MAX_SEQ={MAX_SEQ}, LR={LR}', flush=True)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_length=MAX_SEQ,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy='no',
        logging_steps=5,
        warmup_steps=10,
        lr_scheduler_type='cosine',
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0,
        report_to='none',
        save_total_limit=3,
        max_steps=CHUNK_STEPS,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=val_formatted,
        processing_class=tokenizer,
    )

    _training_state['trainer'] = trainer

    try:
        trainer.train(resume_from_checkpoint=resume_from)
        print(f'\n[STEP 4] Saving model to {OUTPUT_DIR}/final', flush=True)
        trainer.save_model(f'{OUTPUT_DIR}/final')
        tokenizer.save_pretrained(f'{OUTPUT_DIR}/final')

        metrics = {
            'model': MODEL_NAME,
            'params': f'{model.num_parameters()/1e6:.1f}M',
            'trainable_params': f'{trainable/1e6:.1f}M',
            'max_seq': MAX_SEQ,
            'batch': BATCH,
            'grad_accum': GRAD_ACCUM,
            'lora_r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'chunk_steps': CHUNK_STEPS,
            'note': 'Continued training from AcerNitro nemotron-mini-trader-v2 adapter (r=8, alpha=16, q_proj+v_proj)',
            'quantization': 'fp16-4gpu-manual-devicemap',
            'gpu': '4× RX 6600 XT 8GB (manual device_map, 8 layers/GPU)',
            'device_map_strategy': 'balanced: 32 layers / 4 GPUs = 8 per GPU',
            'timestamp': datetime.now().isoformat(),
        }
        with open(f'{OUTPUT_DIR}/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'[DONE] Training complete! Metrics saved.', flush=True)

    except Exception as e:
        print(f'\n[ERROR] Training failed: {e}', flush=True)
        traceback.print_exc()
        trainer.save_model(f'{OUTPUT_DIR}/emergency-checkpoint')
        raise
    finally:
        start_ollama()

if __name__ == '__main__':
    main()

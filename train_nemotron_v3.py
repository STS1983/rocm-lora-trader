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

Requirements:
  - ROCm 7.2, PyTorch 2.12+rocm7.2
  - transformers >= 4.57 (uses `dtype=` not deprecated `torch_dtype=`)
  - peft, trl, accelerate, datasets
  - 4× AMD RX 6600 XT (gfx1032) with HSA_OVERRIDE_GFX_VERSION=10.3.0
  - GRUB kernel params: amdgpu.runpm=0 amdgpu.gpu_recovery=1 pcie_aspm=off
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
os.environ['HIP_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # ALL 4 GPUs

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# ── Config ──
MODEL_NAME = 'nvidia/Nemotron-Mini-4B-Instruct'
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output/nemotron-trader-v3')
TRAIN_FILE = os.environ.get('TRAIN_FILE', './data/train_balanced.jsonl')
VAL_FILE = os.environ.get('VAL_FILE', './data/val_balanced.jsonl')

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
    sys.exit(130)

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
      GPU 0: model.embed_tokens, model.layers.0-7
      GPU 1: model.layers.8-15
      GPU 2: model.layers.16-23
      GPU 3: model.layers.24-31, model.norm, lm_head

    Note: rotary_emb is auto-placed by Accelerate (do NOT add explicitly).
    """
    device_map = {}
    device_map['model.embed_tokens'] = 0
    for i in range(NUM_LAYERS):
        gpu_id = i // LAYERS_PER_GPU
        device_map[f'model.layers.{i}'] = gpu_id
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

    if config.num_hidden_layers != NUM_LAYERS:
        print(f'[INSPECT] ⚠️  WARNING: Expected {NUM_LAYERS} layers but config shows {config.num_hidden_layers}!', flush=True)
        NUM_LAYERS = config.num_hidden_layers
        LAYERS_PER_GPU = max(1, NUM_LAYERS // 4)

    device_map = build_balanced_device_map()
    print(f'\n[INSPECT] Balanced device_map ({NUM_LAYERS} layers, {LAYERS_PER_GPU} per GPU):', flush=True)
    print('-' * 50, flush=True)

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

def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print('=' * 70, flush=True)
    print('Nemotron-Mini-4B Trading Fine-Tune — 4-GPU Balanced device_map (ROCm 7.2)', flush=True)
    print('v3: Manual layer split (8 layers/GPU) for balanced utilization', flush=True)
    print('=' * 70, flush=True)

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

    # ── Load LoRA adapter if specified ──
    ADAPTER_PATH = os.environ.get('ADAPTER_PATH', '')
    if ADAPTER_PATH and Path(ADAPTER_PATH).exists():
        print(f'\n[STEP 2] Loading existing LoRA adapter: {ADAPTER_PATH}', flush=True)
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'[INFO] LoRA loaded! Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)', flush=True)
    else:
        print(f'\n[STEP 2] No adapter path specified, creating new LoRA config', flush=True)
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
            'trainable_params': f'{trainable/1e6:.1f}M' if ADAPTER_PATH else 'new',
            'max_seq': MAX_SEQ, 'batch': BATCH, 'grad_accum': GRAD_ACCUM,
            'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA, 'chunk_steps': CHUNK_STEPS,
            'quantization': 'fp16-4gpu-manual-devicemap',
            'gpu': '4× RX 6600 XT 8GB (manual device_map, 8 layers/GPU)',
            'device_map_strategy': 'balanced: 32 layers / 4 GPUs = 8 per GPU',
            'timestamp': datetime.now().isoformat(),
        }
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(f'{OUTPUT_DIR}/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'[DONE] Training complete! Metrics saved.', flush=True)

    except Exception as e:
        print(f'\n[ERROR] Training failed: {e}', flush=True)
        traceback.print_exc()
        trainer.save_model(f'{OUTPUT_DIR}/emergency-checkpoint')
        raise

if __name__ == '__main__':
    main()

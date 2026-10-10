#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for qwen2.5:3b Trading Model — v5 (Resilient Multi-GPU)
Runs on MortySmith with 4× RX 6600 XT (ROCm 7.0)

v5: Based on v3 (which worked until step 70) with critical fixes:
1. Saves checkpoints every 25 steps (preserves progress on crash)
2. HIP environment variables for ROCm stability
3. Crash recovery: auto-resumes from latest checkpoint
4. Reduced gradient accumulation (12 instead of 8) for less memory pressure
"""

import os
import sys
import json
import time
import gc
import torch
from pathlib import Path
from datetime import datetime

# ============ ROCm/HIP Stability Environment ============
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HSA_ENABLE_SDMA"] = "0"            # Disable SDMA (stability fix for cross-GPU)
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["NCCL_P2P_LEVEL"] = "SYS"           # Enable P2P across GPUs
os.environ["NCCL_DEBUG"] = "WARN"               # Reduce NCCL spam
# Don't restrict GPUs — let device_map="auto" use all 4

def main():
    print("=" * 60)
    print("LoRA Fine-Tuning: qwen2.5:3b → trader-v2 (v5 Resilient Multi-GPU)")
    print("=" * 60)
    
    # Check GPU availability
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA/ROCm available: {torch.cuda.is_available()}")
    print(f"[INFO] GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = (props.total_memory - torch.cuda.memory_reserved(i)) / 1e9
        print(f"[INFO] GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB total, {free_mem:.1f} GB free")
    
    if not torch.cuda.is_available():
        print("[ERROR] No GPUs available! Exiting.")
        sys.exit(1)
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset
    
    # ============ Configuration ============
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    OUTPUT_DIR = "/home/nodeadmin/trading-llm/output/trader-v2"
    TRAINING_DIR = "/home/nodeadmin/trading-llm/training"
    
    # LoRA parameters
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training parameters
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    BATCH_SIZE = 1                  # per GPU
    GRADIENT_ACCUMULATION_STEPS = 8 # effective batch = 1 * 8 = 8
    MAX_SEQ_LENGTH = 512
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    SAVE_STEPS = 25                 # Save every 25 steps (was "epoch" in v3 — now we don't lose progress!)
    EVAL_STEPS = 25                 # Eval every 25 steps
    
    print(f"\n[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] Strategy: Multi-GPU (device_map=auto) with HIP stability fixes")
    print(f"[CONFIG] LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"[CONFIG] Target modules: {TARGET_MODULES}")
    print(f"[CONFIG] Learning rate: {LEARNING_RATE}")
    print(f"[CONFIG] Epochs: {NUM_EPOCHS}")
    print(f"[CONFIG] Batch size per GPU: {BATCH_SIZE}")
    print(f"[CONFIG] Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"[CONFIG] Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"[CONFIG] Save every {SAVE_STEPS} steps (CRASH-RESILIENT)")
    print(f"[CONFIG] Output: {OUTPUT_DIR}")
    
    # ============ Load Data ============
    print("\n[STEP 1] Loading training data...")
    
    train_samples = []
    with open(f"{TRAINING_DIR}/train.jsonl") as f:
        for line in f:
            train_samples.append(json.loads(line.strip()))
    
    val_samples = []
    with open(f"{TRAINING_DIR}/val.jsonl") as f:
        for line in f:
            val_samples.append(json.loads(line.strip()))
    
    print(f"[DATA] Train: {len(train_samples)} samples")
    print(f"[DATA] Validation: {len(val_samples)} samples")
    
    def messages_to_text(example):
        text = ""
        for msg in example["messages"]:
            if msg["role"] == "system":
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        if len(text) > MAX_SEQ_LENGTH:
            text = text[:MAX_SEQ_LENGTH]
        return {"text": text}
    
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    train_dataset = train_dataset.map(messages_to_text)
    val_dataset = val_dataset.map(messages_to_text)
    print(f"[DATA] Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # ============ Load Model ============
    print("\n[STEP 2] Loading base model (fp16, multi-GPU)...")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use device_map="auto" like v3 (which worked until step 70)
    # Model shards across 4 GPUs, LoRA on top
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"[INFO] Model loaded!")
    if hasattr(model, 'hf_device_map'):
        print(f"[INFO] Device map: {model.hf_device_map}")
    print(f"[INFO] Model device: {model.device}")
    
    # ============ LoRA Config ============
    print("\n[STEP 3] Configuring LoRA adapter...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # ============ Check for existing checkpoint to resume ============
    checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
    resume_from = None
    if checkpoints:
        latest_checkpoint = str(checkpoints[-1])
        print(f"\n[RESUME] Found checkpoint: {latest_checkpoint}")
        print(f"[RESUME] Resuming training from this checkpoint...")
        resume_from = latest_checkpoint
    
    # ============ Training Arguments ============
    print("\n[STEP 4] Setting up training arguments...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_dir=f"{OUTPUT_DIR}/logs",
        # KEY FIX v5: Save checkpoints every 25 steps
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    
    # ============ Train ============
    print("\n[STEP 5] Starting training (with checkpoint recovery)...")
    print(f"[INFO] Checkpoints saved every {SAVE_STEPS} steps — progress preserved on crash!")
    start_time = time.time()
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # Train with checkpoint resume support
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from)
        
        training_time = time.time() - start_time
        print(f"\n[TRAINING] Completed in {training_time/60:.1f} minutes")
        print(f"[TRAINING] Train loss: {train_result.training_loss:.4f}")
        
        # ============ Save Final Adapter ============
        print("\n[STEP 6] Saving LoRA adapter...")
        
        adapter_path = f"{OUTPUT_DIR}/final-adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"[SAVE] LoRA adapter saved to {adapter_path}")
        
        # Save training metrics
        metrics = {
            "training_loss": train_result.training_loss,
            "training_time_minutes": training_time / 60,
            "num_epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "quantization": "fp16",
            "multi_gpu": True,
            "strategy": "multi-GPU-v5-resilient",
            "save_steps": SAVE_STEPS,
            "timestamp": datetime.now().isoformat(),
            "status": "SUCCESS",
        }
        
        with open(f"{OUTPUT_DIR}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"LoRA adapter: {adapter_path}")
        print(f"Metrics: {OUTPUT_DIR}/training_metrics.json")
        print(f"\nNext: Run merge_and_convert.py to create Ollama model")
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"\n[ERROR] Training crashed after {training_time/60:.1f} minutes!")
        print(f"[ERROR] {type(e).__name__}: {e}")
        
        # Check for saved checkpoints
        checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"\n[RECOVERY] Found {len(checkpoints)} checkpoint(s)!")
            print(f"[RECOVERY] Latest checkpoint: {latest_checkpoint}")
            print(f"\nTo resume from this checkpoint, just re-run this script!")
            print(f"  python3 train_qlora_v5.py")
            print(f"It will auto-detect and resume from the checkpoint.")
        else:
            print(f"\n[RECOVERY] No checkpoints found. Consider reducing parameters.")
        
        # Save crash info
        crash_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "training_time_minutes": training_time / 60,
            "checkpoints_found": [str(c) for c in sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))],
            "timestamp": datetime.now().isoformat(),
            "status": "CRASHED",
        }
        with open(f"{OUTPUT_DIR}/crash_info.json", "w") as f:
            json.dump(crash_info, f, indent=2)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
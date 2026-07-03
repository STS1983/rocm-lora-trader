#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for qwen2.5:3b Trading Model
Runs on MortySmith with 4× RX 6600 XT (ROCm 7.0)

Fixed version: Load model on CPU first, then apply LoRA and move to GPU.
Avoids segfault from device_map="auto" with bitsandbytes on ROCm.
"""

import os
import sys
import json
import time
import gc
import torch
from pathlib import Path
from datetime import datetime

# ROCm/HIP environment setup - MUST be before torch imports
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"  # For RX 6600 XT (gfx1032)

def main():
    print("=" * 60)
    print("QLoRA Fine-Tuning: qwen2.5:3b → trader-v2")
    print("ROCm-compatible version (CPU load + manual GPU placement)")
    print("=" * 60)
    
    # Check GPU availability
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA/ROCm available: {torch.cuda.is_available()}")
    print(f"[INFO] GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[INFO] GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    if not torch.cuda.is_available():
        print("[ERROR] No GPUs available! Exiting.")
        sys.exit(1)
    
    # Import training libraries
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    
    # ============ Configuration ============
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    OUTPUT_DIR = "/home/nodeadmin/trading-llm/output/trader-v2"
    TRAINING_DIR = "/home/nodeadmin/trading-llm/training"
    
    # QLoRA parameters
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training parameters
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 4
    BATCH_SIZE = 1  # per GPU (conservative for 8GB VRAM)
    GRADIENT_ACCUMULATION_STEPS = 16  # effective batch = 1 * 4 GPUs * 16 = 64
    MAX_SEQ_LENGTH = 512
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    print(f"\n[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"[CONFIG] Target modules: {TARGET_MODULES}")
    print(f"[CONFIG] Learning rate: {LEARNING_RATE}")
    print(f"[CONFIG] Epochs: {NUM_EPOCHS}")
    print(f"[CONFIG] Batch size per GPU: {BATCH_SIZE}")
    print(f"[CONFIG] Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"[CONFIG] Effective batch size: {BATCH_SIZE * max(1, torch.cuda.device_count()) * GRADIENT_ACCUMULATION_STEPS}")
    print(f"[CONFIG] Max sequence length: {MAX_SEQ_LENGTH}")
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
        """Convert messages format to text for SFT."""
        text = ""
        for msg in example["messages"]:
            if msg["role"] == "system":
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        return {"text": text}
    
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    train_dataset = train_dataset.map(messages_to_text)
    val_dataset = val_dataset.map(messages_to_text)
    
    print(f"[DATA] Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # ============ Load Model (ROCm-safe approach) ============
    print("\n[STEP 2] Loading base model (ROCm-compatible method)...")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Strategy: Load model on CPU with 4-bit quantization, then manually place on GPU
    # This avoids the segfault from device_map="auto" with bitsandbytes on ROCm
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,  # Disable double quant for ROCm compatibility
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device_map pointing to GPU 0
    # This is more reliable on ROCm than "auto"
    print("[INFO] Loading model on GPU 0 with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},  # Place everything on GPU 0
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("[INFO] Model loaded on GPU 0!")
    except Exception as e:
        print(f"[WARN] GPU 0 loading failed: {e}")
        print("[INFO] Trying CPU load + manual GPU placement...")
        # Fallback: load on CPU, quantize manually
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to("cuda:0")
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
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
        eval_strategy="epoch",
        save_strategy="epoch",
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
        # Don't use DataParallel - keep on single GPU for stability
        ddp_find_unused_parameters=False,
    )
    
    # ============ Train ============
    print("\n[STEP 5] Starting training...")
    start_time = time.time()
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )
    
    # Train!
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    print(f"\n[TRAINING] Completed in {training_time/60:.1f} minutes")
    print(f"[TRAINING] Train loss: {train_result.training_loss:.4f}")
    
    # ============ Save ============
    print("\n[STEP 6] Saving LoRA adapter...")
    
    # Save LoRA adapter
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
        "effective_batch_size": BATCH_SIZE * max(1, torch.cuda.device_count()) * GRADIENT_ACCUMULATION_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{OUTPUT_DIR}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"LoRA adapter: {adapter_path}")
    print(f"Metrics: {OUTPUT_DIR}/training_metrics.json")
    print(f"\nNext: Run merge_and_convert.py to create Ollama model")


if __name__ == "__main__":
    main()
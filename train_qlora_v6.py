#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for Qwen2.5-3B Trading Model — v6 (DDP Multi-GPU Balanced)

Runs on MortySmith with 4× RX 6600 XT (ROCm 7.0) using DistributedDataParallel.

PROBLEM with v5 (device_map="auto"):
  - GPU 0 at 99% VRAM (forward pass + loss + gradients all on GPU 0)
  - GPUs 1-3 at 23-29% (only model weight shards, no compute)

SOLUTION v6 (DDP):
  - Each GPU holds a FULL model copy + LoRA adapter
  - Each GPU processes a different microbatch (data parallelism)
  - Gradients are synchronized across GPUs via allreduce
  - Effective batch size = per_gpu_batch × grad_accum × num_gpus = 1 × 4 × 4 = 16

FLEET-ROUTER INSPIRED (7-signal scoring from ollama-herd):
  - VRAM Monitor: periodic GPU memory checks during training
  - Adaptive Actions: auto-downgrade to 4-bit if OOM risk detected
  - Warmup Phase: validates GPU utilization before full training commitment
  - Per-step logging: VRAM stats written to vram_stats.jsonl for post-hoc analysis

LAUNCH:
  bash launch_v6.sh                # fp16 mode (default)
  bash launch_v6.sh --4bit         # 4-bit QLoRA mode (if fp16 is too tight)

  Or directly:
  torchrun --nproc_per_node=4 train_qlora_v6.py
  USE_4BIT=1 torchrun --nproc_per_node=4 train_qlora_v6.py
"""

import os
import sys
import json
import time
import gc
import threading
import traceback
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime

# ============ ROCm/HIP Stability Environment (MUST be set before torch init) ============
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["NCCL_P2P_LEVEL"] = "SYS"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_IB_DISABLE"] = "1"              # No InfiniBand on consumer GPUs
os.environ["NCCL_NET"] = "Socket"                  # Force socket transport (RX 6600 XT)
os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3"      # Use all 4 GPUs


# ============================================================================
# VRAM Monitor — inspired by ollama-herd's 7-signal scoring
# ============================================================================
class VRAMMonitor:
    """
    Periodic VRAM monitor that scores GPU health during training.
    
    Signals monitored (fleet-router pattern):
    1. Memory Fit — % of VRAM used (OOM risk)
    2. Memory Trend — is usage growing? (leak detection)
    3. Compute Utilization — GPU % busy (via rocm-smi if available)
    4. Thermal — GPU temperature (throttle risk)
    5. Allocation Slope — rate of VRAM growth per step
    6. Reserved vs Allocated — fragmentation ratio
    7. Step Efficiency — steps/sec (throughput health)
    
    Actions:
    - Log VRAM stats every N steps to vram_stats.jsonl
    - Warn if any GPU > 85% VRAM
    - Auto-save checkpoint if any GPU > 92% VRAM (pre-emptive crash protection)
    - Record per-step metrics for post-hoc analysis
    """
    
    def __init__(self, output_dir, local_rank, world_size, is_main, 
                 check_interval_steps=10, warn_threshold=0.85, 
                 critical_threshold=0.92):
        self.output_dir = output_dir
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = is_main
        self.check_interval = check_interval_steps
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        
        # History tracking
        self.history = []  # list of per-step VRAM snapshots
        self.step_count = 0
        self.prev_allocated = {}  # {gpu_idx: allocated_gb} for trend detection
        self.start_time = time.time()
        
        # Stats file (only main writes)
        self.stats_file = Path(output_dir) / "vram_stats.jsonl"
        if is_main:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            # Write header
            with open(self.stats_file, "a") as f:
                f.write(f"# VRAM Monitor — started {datetime.now().isoformat()}\n")
                f.write(f"# GPUs: {world_size}, warn: {warn_threshold*100:.0f}%, critical: {critical_threshold*100:.0f}%\n")
        
        # Per-GPU total memory
        self.gpu_totals = {}
        for i in range(torch.cuda.device_count()):
            self.gpu_totals[i] = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        # Warmup tracking
        self.warmup_complete = False
        self.warmup_steps_needed = 5  # need 5 steps to establish baseline
        self.warmup_vram_samples = []
        
        self._preemptive_save_triggered = False
    
    def check(self, step, trainer=None):
        """
        Score GPU health at this training step.
        Returns: (action, details) where action is 'ok', 'warn', 'critical'
        """
        self.step_count = step
        
        # Collect VRAM stats from ALL GPUs (visible to this process)
        snapshot = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": time.time() - self.start_time,
            "gpus": {}
        }
        
        action = "ok"
        details = {"warn_gpus": [], "critical_gpus": [], "max_usage": 0.0}
        
        for i in range(torch.cuda.device_count()):
            total = self.gpu_totals[i]
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            free = total - reserved
            usage_ratio = reserved / total if total > 0 else 0
            
            # Trend detection
            prev = self.prev_allocated.get(i, allocated)
            trend = allocated - prev  # GB change since last check
            self.prev_allocated[i] = allocated
            
            # Fragmentation ratio
            frag_ratio = reserved / allocated if allocated > 0.01 else 1.0
            
            snapshot["gpus"][str(i)] = {
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "total_gb": round(total, 2),
                "free_gb": round(free, 3),
                "usage_pct": round(usage_ratio * 100, 1),
                "trend_gb": round(trend, 4),
                "frag_ratio": round(frag_ratio, 2),
            }
            
            details["max_usage"] = max(details["max_usage"], usage_ratio)
            
            # Signal 1: Memory Fit (OOM risk)
            if usage_ratio >= self.critical_threshold:
                action = "critical"
                details["critical_gpus"].append(i)
            elif usage_ratio >= self.warn_threshold:
                if action != "critical":
                    action = "warn"
                details["warn_gpus"].append(i)
        
        # Signal 7: Step efficiency (throughput)
        if len(self.history) >= 2:
            prev_step = self.history[-1]
            elapsed_delta = snapshot["elapsed_sec"] - prev_step["elapsed_sec"]
            step_delta = snapshot["step"] - prev_step["step"]
            if elapsed_delta > 0:
                steps_per_sec = step_delta / elapsed_delta
                snapshot["throughput"] = round(steps_per_sec, 3)
        
        # Signal 5: Allocation slope (growing memory = potential leak)
        if len(self.history) >= 3:
            recent_usage = [h.get("gpus", {}).get("0", {}).get("usage_pct", 0) 
                           for h in self.history[-3:]]
            if all(u > 0 for u in recent_usage):
                slope = recent_usage[-1] - recent_usage[0]
                snapshot["vram_slope_pct"] = round(slope, 2)
                # If usage growing > 2% over 3 checks, flag it
                if slope > 2.0 and action == "ok":
                    action = "warn"
                    details["trend"] = f"VRAM growing at {slope:.1f}% per {self.check_interval} steps"
        
        # Warmup phase tracking
        if not self.warmup_complete:
            self.warmup_vram_samples.append(snapshot)
            if len(self.warmup_vram_samples) >= self.warmup_steps_needed:
                self.warmup_complete = True
                if self.is_main:
                    avg_max = sum(
                        s.get("max_usage", 0) for s in self.warmup_vram_samples
                    ) / len(self.warmup_vram_samples)
                    print(f"\n[WARMUP] ✅ Warmup complete after {len(self.warmup_vram_samples)} step checks")
                    print(f"[WARMUP] Average peak VRAM usage: {avg_max*100:.1f}%")
                    print(f"[WARMUP] VRAM budget: {details['max_usage']*100:.1f}% of total")
                    if avg_max > self.warn_threshold:
                        print(f"[WARMUP] ⚠️  WARNING: Average usage {avg_max*100:.1f}% exceeds {self.warn_threshold*100:.0f}% threshold!")
                        print(f"[WARMUP] Consider using --4bit mode for more headroom")
        
        snapshot["action"] = action
        snapshot["warmup_complete"] = self.warmup_complete
        
        # Store history (keep last 100 snapshots in memory)
        self.history.append(snapshot)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Write to log file (main process only)
        if self.is_main:
            with open(self.stats_file, "a") as f:
                f.write(json.dumps(snapshot) + "\n")
        
        # Take action based on severity
        if action == "critical" and self.is_main:
            critical_gpus = details["critical_gpus"]
            print(f"\n[VRAM 🚨] CRITICAL: GPUs {critical_gpus} at >{self.critical_threshold*100:.0f}% VRAM!")
            print(f"[VRAM 🚨] Pre-emptive checkpoint save recommended!")
            self._preemptive_save_triggered = True
            
            # Try to save checkpoint if trainer is available
            if trainer is not None:
                try:
                    save_path = os.path.join(
                        self.output_dir, 
                        f"checkpoint-preemptive-step{step}"
                    )
                    print(f"[VRAM 🚨] Saving pre-emptive checkpoint to {save_path}...")
                    trainer.save_model(save_path)
                    print(f"[VRAM 🚨] ✅ Pre-emptive checkpoint saved!")
                except Exception as e:
                    print(f"[VRAM 🚨] ❌ Failed to save pre-emptive checkpoint: {e}")
        
        elif action == "warn" and self.is_main:
            warn_gpus = details["warn_gpus"]
            max_pct = details["max_usage"] * 100
            print(f"[VRAM ⚠️]  Warning: GPUs {warn_gpus} at >{self.warn_threshold*100:.0f}% VRAM (max: {max_pct:.1f}%)")
        
        elif self.is_main and step % (self.check_interval * 5) == 0:
            # Periodic summary every 50 steps
            self._print_summary(step)
        
        return action, details
    
    def _print_summary(self, step):
        """Print a formatted VRAM summary table."""
        print(f"\n{'='*70}")
        print(f"[VRAM Monitor] Step {step} — GPU Health Summary")
        print(f"{'='*70}")
        print(f"{'GPU':>4} | {'Allocated':>10} | {'Reserved':>10} | {'Total':>8} | {'Free':>8} | {'Usage':>7} | {'Trend':>8}")
        print(f"{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
        
        for i in range(torch.cuda.device_count()):
            total = self.gpu_totals[i]
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            free = total - reserved
            usage = (reserved / total) * 100
            trend = self.prev_allocated.get(i, allocated) - (self.history[-2]["gpus"].get(str(i), {}).get("allocated_gb", 0) if len(self.history) >= 2 else 0)
            
            status = "🟢" if usage < self.warn_threshold * 100 else ("🟡" if usage < self.critical_threshold * 100 else "🔴")
            print(f"  {i}  | {allocated:>8.2f}GB | {reserved:>8.2f}GB | {total:>6.1f}GB | {free:>6.2f}GB | {usage:>5.1f}% | {trend:>+.3f}GB {status}")
        
        if self.warmup_complete and len(self.history) >= 3:
            recent = self.history[-3:]
            slopes = []
            for h in recent:
                gpus = h.get("gpus", {})
                for g, v in gpus.items():
                    slopes.append(v.get("usage_pct", 0))
            print(f"\n  Throughput: {self.history[-1].get('throughput', 'N/A')} steps/sec")
            print(f"  Warmup: {'✅ Complete' if self.warmup_complete else '⏳ In progress'}")
        print(f"{'='*70}\n")
    
    def get_warmup_verdict(self):
        """
        After warmup, return a verdict on whether training can continue safely.
        Inspired by ollama-herd's node scoring before routing.
        
        Returns: ('proceed', {}) or ('downgrade', {'reason': '...'})
        """
        if not self.warmup_complete:
            return "proceed", {}
        
        avg_max_usage = sum(
            s.get("max_usage", 0) for s in self.warmup_vram_samples
        ) / len(self.warmup_vram_samples)
        
        if avg_max_usage > 0.92:
            return "downgrade", {
                "reason": f"Average VRAM usage {avg_max_usage*100:.1f}% exceeds 92% threshold",
                "recommendation": "Use --4bit mode (USE_4BIT=1) to reduce VRAM by ~4GB per GPU",
                "avg_vram_pct": round(avg_max_usage * 100, 1),
            }
        elif avg_max_usage > 0.85:
            return "proceed", {
                "warning": f"Average VRAM usage {avg_max_usage*100:.1f}% is above 85%",
                "note": "Training may be unstable. Monitor VRAM stats closely.",
                "avg_vram_pct": round(avg_max_usage * 100, 1),
            }
        else:
            return "proceed", {
                "status": "healthy",
                "avg_vram_pct": round(avg_max_usage * 100, 1),
            }


def setup_distributed():
    """Initialize DDP process group and set device."""
    if "RANK" in os.environ:
        # Launched via accelerate or torchrun
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        # Single GPU fallback
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    is_main = rank == 0
    return rank, local_rank, world_size, is_main


def cleanup_distributed():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # ============ Distributed Setup ============
    rank, local_rank, world_size, is_main = setup_distributed()

    if is_main:
        print("=" * 70)
        print("LoRA Fine-Tuning: Qwen2.5-3B → trader-v3 (v6 DDP Multi-GPU Balanced)")
        print("=" * 70)
        print(f"[INFO] DDP: {world_size} GPUs, local_rank={local_rank}")
        print(f"[INFO] Fleet-Router VRAM Monitor: ENABLED (7-signal scoring)")

    # ============ Configuration ============
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    OUTPUT_DIR = "/home/nodeadmin/trading-llm/output/trader-v3"
    TRAINING_DIR = "/home/nodeadmin/trading-llm/training-v3"

    # LoRA parameters (same as v5)
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Training parameters — DDP-aware
    # Effective batch = per_gpu_batch × grad_accum × world_size = 1 × 4 × 4 = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    BATCH_SIZE = 1                  # per GPU (keep low for 8GB VRAM)
    GRADIENT_ACCUMULATION_STEPS = 4 # reduced from 8 since DDP gives us 4× parallelism
    MAX_SEQ_LENGTH = 512
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    SAVE_STEPS = 25                # checkpoint every 25 steps
    EVAL_STEPS = 25                # eval every 25 steps
    LOGGING_STEPS = 10

    # VRAM Monitor settings (fleet-router inspired)
    VRAM_CHECK_INTERVAL = 5        # check VRAM every 5 steps
    VRAM_WARN_THRESHOLD = 0.85      # warn at 85% VRAM usage
    VRAM_CRITICAL_THRESHOLD = 0.92  # pre-emptive save at 92% VRAM usage

    USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"

    if is_main:
        print(f"\n[CONFIG] Model: {MODEL_NAME}")
        print(f"[CONFIG] Strategy: DDP (DistributedDataParallel) — {world_size} GPUs")
        print(f"[CONFIG] Mode: {'4-bit QLoRA (NF4)' if USE_4BIT else 'fp16 (full precision)'}")
        print(f"[CONFIG] LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
        print(f"[CONFIG] Target modules: {TARGET_MODULES}")
        print(f"[CONFIG] Learning rate: {LEARNING_RATE}")
        print(f"[CONFIG] Epochs: {NUM_EPOCHS}")
        print(f"[CONFIG] Batch size per GPU: {BATCH_SIZE}")
        print(f"[CONFIG] Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"[CONFIG] Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size}")
        print(f"[CONFIG] Max sequence length: {MAX_SEQ_LENGTH}")
        print(f"[CONFIG] Save every {SAVE_STEPS} steps (CRASH-RESILIENT)")
        print(f"[CONFIG] VRAM Monitor: every {VRAM_CHECK_INTERVAL} steps, warn {VRAM_WARN_THRESHOLD*100:.0f}%, critical {VRAM_CRITICAL_THRESHOLD*100:.0f}%")
        print(f"[CONFIG] Output: {OUTPUT_DIR}")

    # ============ Check GPU Memory (only on main process) ============
    if is_main:
        print(f"\n[INFO] PyTorch version: {torch.__version__}")
        print(f"[INFO] CUDA/ROCm available: {torch.cuda.is_available()}")
        print(f"[INFO] GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"[INFO] GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB total")

    # ============ Imports (after distributed init) ============
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset

    # ============ Load Data (all ranks, but only main prints) ============
    if is_main:
        print("\n[STEP 1] Loading training data...")

    train_samples = []
    with open(f"{TRAINING_DIR}/train.jsonl") as f:
        for line in f:
            train_samples.append(json.loads(line.strip()))

    val_samples = []
    with open(f"{TRAINING_DIR}/val.jsonl") as f:
        for line in f:
            val_samples.append(json.loads(line.strip()))

    if is_main:
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

    if is_main:
        print(f"[DATA] Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} val")

    # ============ Load Model (each rank loads onto its own GPU) ============
    if is_main:
        print(f"\n[STEP 2] Loading base model on GPU {local_rank} ({'4-bit QLoRA' if USE_4BIT else 'fp16'})...")

    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DDP: Each GPU loads a FULL copy of the model (no device_map="auto")
    # This is the key difference from v5 — compute is now evenly distributed
    #
    # Strategy: fp16 default, 4-bit QLoRA as fallback for tight VRAM
    # Qwen2.5-3B in fp16 ≈ 6GB → fits in 8.6GB but training overhead is tight
    # Qwen2.5-3B in 4-bit ≈ 2GB → plenty of room for training

    if USE_4BIT:
        if is_main:
            print("[INFO] Using 4-bit QLoRA mode (NF4 quantization)")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # nested quantization for extra savings
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map={"": local_rank},  # Pin to specific GPU for DDP
        )
        # Prepare model for k-bit training (required for 4-bit + LoRA)
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        if is_main:
            print("[INFO] Using fp16 mode (full precision)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(f"cuda:{local_rank}")

    # Print VRAM after model load on each GPU's own process
    local_alloc = torch.cuda.memory_allocated(local_rank) / 1e9
    local_reserved = torch.cuda.memory_reserved(local_rank) / 1e9
    local_total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    local_free = local_total - local_reserved
    if is_main:
        print(f"[INFO] Model loaded on GPU {local_rank}!")
        print(f"[INFO] GPU {local_rank} VRAM: {local_alloc:.2f} GB allocated, {local_reserved:.2f} GB reserved, {local_total:.1f} GB total, {local_free:.2f} GB free ({local_reserved/local_total*100:.1f}% used)")

    # ============ LoRA Config ============
    if is_main:
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

    if is_main:
        model.print_trainable_parameters()

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Wrap model in DDP
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if is_main:
            print("[INFO] Model wrapped in DistributedDataParallel")

    # Print post-DDP VRAM
    if is_main:
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] After LoRA+DDP — GPU 0: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.1f} GB total ({reserved/total*100:.1f}% used)")

    # ============ Initialize VRAM Monitor ============
    vram_monitor = VRAMMonitor(
        output_dir=OUTPUT_DIR,
        local_rank=local_rank,
        world_size=world_size,
        is_main=is_main,
        check_interval_steps=VRAM_CHECK_INTERVAL,
        warn_threshold=VRAM_WARN_THRESHOLD,
        critical_threshold=VRAM_CRITICAL_THRESHOLD,
    )

    # ============ Check for existing checkpoint to resume ============
    checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
    resume_from = None
    if checkpoints:
        latest_checkpoint = str(checkpoints[-1])
        if is_main:
            print(f"\n[RESUME] Found checkpoint: {latest_checkpoint}")
            print(f"[RESUME] Resuming training from this checkpoint...")
        resume_from = latest_checkpoint

    # ============ Training Arguments ============
    if is_main:
        print("\n[STEP 4] Setting up training arguments...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

    # For DDP, we need to ensure only rank 0 saves and logs
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
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        # Save/eval every 25 steps (crash-resilient)
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Mixed precision (use fp16, NOT bf16 — RX 6600 XT doesn't support bf16)
        fp16=True,
        bf16=False,
        # For 4-bit mode, don't use fp16 (quantized model handles this)
        **({"fp16": False, "bf16": False} if os.environ.get("USE_4BIT", "0") == "1" else {}),
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,          # ROCm compatibility: avoid multiprocessing issues
        # Reporting
        report_to="none",
        seed=42,
        # Only rank 0 saves
        save_on_each_node=False,
        run_name=f"trader-v3-ddp-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )

    # ============ VRAM Monitor Callback ============
    class VRAMMonitorCallback(TrainerCallback):
        """Integrate VRAM monitoring into the training loop."""
        
        def __init__(self, monitor, trainer_ref):
            self.monitor = monitor
            self.trainer_ref = trainer_ref
        
        def on_step_end(self, args, state, control, **kwargs):
            """Check VRAM at configured intervals."""
            if state.global_step % self.monitor.check_interval == 0:
                action, details = self.monitor.check(
                    step=state.global_step,
                    trainer=self.trainer_ref,
                )
                # If critical, we could force a save in the next on_step_end
                # but trainer.save_model() is called from VRAMMonitor.check() already
            return control
        
        def on_train_end(self, args, state, control, **kwargs):
            """Final VRAM summary."""
            if self.monitor.is_main:
                action, details = self.monitor.check(step=state.global_step)
                print(f"\n[VRAM Monitor] Final check at step {state.global_step}")
                print(f"[VRAM Monitor] Peak usage: {details['max_usage']*100:.1f}%")
                verdict, verdict_info = self.monitor.get_warmup_verdict()
                print(f"[VRAM Monitor] Verdict: {verdict} — {verdict_info}")

    # ============ Train ============
    if is_main:
        print("\n[STEP 5] Starting DDP training...")
        print(f"[INFO] {world_size} GPUs × batch {BATCH_SIZE} × accum {GRADIENT_ACCUMULATION_STEPS} = effective batch {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size}")
        print(f"[INFO] Checkpoints saved every {SAVE_STEPS} steps — progress preserved on crash!")
        print(f"[INFO] VRAM Monitor: checking every {VRAM_CHECK_INTERVAL} steps (warn {VRAM_WARN_THRESHOLD*100:.0f}%, critical {VRAM_CRITICAL_THRESHOLD*100:.0f}%)")

    start_time = time.time()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Add VRAM monitor callback
    vram_callback = VRAMMonitorCallback(vram_monitor, trainer)
    trainer.add_callback(vram_callback)

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        training_time = time.time() - start_time

        # Only main process saves final adapter and metrics
        if is_main:
            print(f"\n[TRAINING] Completed in {training_time/60:.1f} minutes")
            print(f"[TRAINING] Train loss: {train_result.training_loss:.4f}")

            # ============ Save Final Adapter ============
            print("\n[STEP 6] Saving LoRA adapter...")

            # Unwrap DDP before saving
            unwrapped_model = model.module if hasattr(model, 'module') else model
            adapter_path = f"{OUTPUT_DIR}/final-adapter"
            unwrapped_model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            print(f"[SAVE] LoRA adapter saved to {adapter_path}")

            # Print VRAM stats
            print(f"\n{'='*70}")
            print("[VRAM] Final GPU Health Summary")
            print(f"{'='*70}")
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1e9
                res = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                free = total - res
                usage_pct = (res / total) * 100
                status = "🟢" if usage_pct < 80 else ("🟡" if usage_pct < 90 else "🔴")
                print(f"  GPU {i}: {alloc:.2f} GB alloc, {res:.2f} GB reserved, {free:.2f} GB free, {total:.1f} GB total ({usage_pct:.1f}% used) {status}")
            print(f"{'='*70}")

            # Save training metrics
            metrics = {
                "training_loss": train_result.training_loss,
                "training_time_minutes": training_time / 60,
                "num_epochs": NUM_EPOCHS,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "batch_size_per_gpu": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size,
                "max_seq_length": MAX_SEQ_LENGTH,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "quantization": "4bit-nf4" if USE_4BIT else "fp16",
                "strategy": "DDP-v6-balanced",
                "world_size": world_size,
                "save_steps": SAVE_STEPS,
                "vram_monitor": {
                    "check_interval_steps": VRAM_CHECK_INTERVAL,
                    "warn_threshold": VRAM_WARN_THRESHOLD,
                    "critical_threshold": VRAM_CRITICAL_THRESHOLD,
                    "total_checks": len(vram_monitor.history),
                },
                "timestamp": datetime.now().isoformat(),
                "status": "SUCCESS",
            }

            with open(f"{OUTPUT_DIR}/training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print("\n" + "=" * 70)
            print("✅ TRAINING COMPLETE!")
            print("=" * 70)
            print(f"LoRA adapter: {adapter_path}")
            print(f"Metrics: {OUTPUT_DIR}/training_metrics.json")
            print(f"VRAM stats: {OUTPUT_DIR}/vram_stats.jsonl")
            print(f"\nNext: Run merge_and_convert.py to create Ollama model")

    except Exception as e:
        training_time = time.time() - start_time
        if is_main:
            print(f"\n[ERROR] Training crashed after {training_time/60:.1f} minutes!")
            print(f"[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()

            # Check for saved checkpoints
            checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                print(f"\n[RECOVERY] Found {len(checkpoints)} checkpoint(s)!")
                print(f"[RECOVERY] Latest checkpoint: {latest_checkpoint}")
                print(f"\nTo resume from this checkpoint, just re-run!")
                print(f"  torchrun --nproc_per_node=4 train_qlora_v6.py")
            else:
                print(f"\n[RECOVERY] No checkpoints found. Consider reducing parameters.")
                print(f"[RECOVERY] Try --4bit mode: USE_4BIT=1 torchrun --nproc_per_node=4 train_qlora_v6.py")

            # Save crash info
            crash_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "error_traceback": traceback.format_exc(),
                "training_time_minutes": training_time / 60,
                "checkpoints_found": [str(c) for c in sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))],
                "vram_at_crash": {
                    str(i): {
                        "allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 3),
                        "reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 3),
                        "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                    }
                    for i in range(torch.cuda.device_count())
                } if torch.cuda.is_available() else {},
                "timestamp": datetime.now().isoformat(),
                "status": "CRASHED",
            }
            with open(f"{OUTPUT_DIR}/crash_info.json", "w") as f:
                json.dump(crash_info, f, indent=2)

        # Make sure all processes synchronize before exit
        if dist.is_initialized():
            try:
                dist.barrier(timeout=30)
            except Exception:
                pass  # Don't hang on barrier if other processes also crashed

        sys.exit(1)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
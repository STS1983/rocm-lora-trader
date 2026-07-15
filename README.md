# ROCm LoRA Trader — GPU Balanced Training

Fine-tune LLMs for trading on AMD ROCm with multi-GPU load balancing.

## Scripts

### train_nemotron_v3.py — Balanced device_map (Latest)
- **Model:** Nemotron-Mini-4B-Instruct (4B params, 32 decoder layers)
- **GPU:** 4× AMD RX 6600 XT (8GB each, gfx1032)
- **Strategy:** Manual `device_map` with even layer splitting (8 layers/GPU)
- **Fix:** Replaces `device_map='auto'` which caused GPU0 at 38-56% while GPU1-3 at 99%

**Usage:**
```bash
# Set environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training (100 steps per chunk)
python3 train_nemotron_v3.py

# Or with custom paths
TRAIN_FILE=./data/train.jsonl VAL_FILE=./data/val.jsonl \
ADAPTER_PATH=./adapters/my-adapter \
OUTPUT_DIR=./output/my-model \
CHUNK_STEPS=200 python3 train_nemotron_v3.py
```

### Older Versions
- `train_qlora_v3.py` — QLoRA v3 (single GPU, Qwen2.5-3B)
- `train_qlora_v5.py` — QLoRA v5 (improved data processing)
- `train_qlora_v6.py` — QLoRA v6 (multi-GPU DDP attempt, broken on ROCm)

## Requirements

- ROCm 7.2+, PyTorch 2.12+rocm7.2
- transformers >= 4.57 (uses `dtype=`, not deprecated `torch_dtype=`)
- peft, trl, accelerate, datasets
- AMD GPU with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (for gfx1032 / RX 6600 XT)

## GRUB Kernel Parameters (Critical for Multi-GPU Riser Rigs)

Add to `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.runpm=0 amdgpu.gpu_recovery=1 pcie_aspm=off"
```

- `amdgpu.runpm=0` — Disables runtime power management (prevents PSP failures)
- `amdgpu.gpu_recovery=1` — Enables automatic GPU reset on failure
- `pcie_aspm=off` — Disables PCIe Active State Power Management (riser stability)

See [docs/PSP-FAILURE-FIX.md](docs/PSP-FAILURE-FIX.md) for full PSP failure analysis.

## Known Issues

1. **DDP/FSDP doesn't work** on RX 6600 XT with PCIe x1 risers (P2P issue)
2. **`PYTORCH_HIP_ALLOC_CONF` crashes** on ROCm 7.2 + PyTorch 2.12 (hipErrorIllegalAddress)
3. **`device_map='auto'`** distributes layers sequentially causing GPU imbalance
4. **PSP failures** on multi-GPU rigs with risers — fixed by `amdgpu.runpm=0`

## Code Review Process

All training scripts go through multi-LLM code review:
1. **glm-5.2:cloud** — Primary coder
2. **kimi-k2.6:cloud** — Review #1 (code quality, bug detection)
3. **qwen3.5:397b-cloud** — Review #2 (QA, edge cases)

## Results (v3 Test-Run)

| Metric | Value |
|--------|-------|
| Steps | 10/10 ✅ |
| Loss | 3.45 → 2.51 (descending) |
| Speed | ~22.5s/step |
| GPU 0 params | 34.4% (layers 0-7 + embeddings) |
| GPU 1 params | 15.6% (layers 8-15) |
| GPU 2 params | 15.6% (layers 16-23) |
| GPU 3 params | 34.4% (layers 24-31 + norm + lm_head) |
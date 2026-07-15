# Nemotron v3 LoRA Merge & Deploy Pipeline

**Created:** 2026-07-16
**Goal:** Train Nemotron-Mini-4B LoRA on multi-GPU ROCm, merge to GGUF, deploy to Ollama

## Problem

Ollama 0.32 dropped support for `NemotronForCausalLM` architecture. New models
cannot be created from HuggingFace format. However, existing v2 GGUFs (created
with older Ollama) still work.

## Solution

Use the existing v2 Q4_K_M GGUF as a base, convert the v3 LoRA adapter to GGUF,
merge them with `llama-export-lora`, quantize, and deploy.

## Pipeline Steps

### Step 1: Train LoRA Adapter (MortySmith, 4× RX 6600 XT)

```bash
# Balanced device_map training (8 layers per GPU)
# See: tools/train_nemotron_v3.py
ssh mortysmith "cd ~/trading-llm && python3 train_nemotron_v3.py"
```

**Requirements:**
- ROCm 7.2, PyTorch 2.12+rocm7.2
- GRUB: `amdgpu.runpm=0 amdgpu.gpu_recovery=1 pcie_aspm=off`
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` (for gfx1032 / RX 6600 XT)
- transformers >= 4.57 (uses `dtype=`, not deprecated `torch_dtype=`)
- Do NOT set `PYTORCH_HIP_ALLOC_CONF` (crashes on ROCm 7.2)

**Output:** `output/nemotron-trader-v3/final/adapter_model.safetensors` (10.5 MB)

### Step 2: Convert LoRA Adapter to GGUF

```bash
# Uses llama.cpp's convert_lora_to_gguf.py
# See: tools/lora-merge-pipeline/convert_lora_to_gguf.py
CUDA_VISIBLE_DEVICES='' python3 ~/llama.cpp/convert_lora_to_gguf.py \
    ~/trading-llm/output/nemotron-trader-v3/final \
    --outfile ~/trading-llm/output/nemotron-trader-v3/nemotron-v3-lora.gguf \
    --outtype f16 \
    --trust-remote-code
```

**Output:** `nemotron-v3-lora.gguf` (5.2 MB)

### Step 3: Merge LoRA onto v2 Base GGUF

```bash
# Requires llama.cpp built (CPU-only is sufficient)
# Build: cmake -B build-cpu -DGGML_HIP=OFF && cmake --build build-cpu -j$(nproc) -- llama-export-lora llama-quantize

# Get v2 base GGUF from AcerNitro (or backup)
# Then merge:
~/llama.cpp/build-cpu/bin/llama-export-lora \
    -m base-v2.gguf \
    --lora nemotron-v3-lora.gguf \
    -o nemotron-trader-v3-merged.gguf
```

**What happens:**
- Dequantizes base Q4_K_M tensors to F32
- Applies LoRA: `W_new = W_base + scale * B @ A` (scale = alpha/r = 16/8 = 2.0)
- Only `q_proj` and `v_proj` tensors are modified (64 tensors)
- All other tensors are copied as-is
- Output is F16

**Output:** `nemotron-trader-v3-merged.gguf` (3.3 GB, F16)

### Step 4: Quantize to Q4_K_M

```bash
~/llama.cpp/build-cpu/bin/llama-quantize \
    nemotron-trader-v3-merged.gguf \
    nemotron-trader-v3-q4_k_m.gguf \
    Q4_K_M
```

**Output:** `nemotron-trader-v3-q4_k_m.gguf` (2.6 GB)

### Step 5: Deploy to Ollama

```bash
echo 'FROM ./nemotron-trader-v3-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER top_k 40
PARAMETER num_predict 128' > Modelfile

ollama create nemotron-trader-v3 -f Modelfile
```

**Note:** This works because the GGUF retains the `nemotron` architecture tag
from the v2 base. Ollama doesn't need to create a new model from HF format —
it just imports the GGUF directly.

### Step 6: Deploy to AcerNitro

```bash
# Copy GGUF to AcerNitro
scp -P 2200 nemotron-trader-v3-q4_k_m.gguf pouwfrontend@192.168.0.115:~/

# Create Modelfile and import
ssh -p 2200 pouwfrontend@192.168.0.115 "cd ~ && ollama create nemotron-trader-v3 -f Modelfile"
```

### Step 7: Backtest

```bash
# See: tools/backtest/backtest_nemotron_v3.py
python3 backtest_nemotron_v3.py 250  # Test 250 samples
```

## File Structure

```
rocm-lora-trader/
├── tools/
│   ├── train_nemotron_v3.py              # Step 1: Training script
│   ├── lora-merge-pipeline/
│   │   ├── convert_lora_to_gguf.py       # Step 2: LoRA → GGUF conversion
│   │   └── merge_nemotron_v3_gpu.py      # Alternative: HF merge (not used)
│   ├── inference/
│   │   └── serve_nemotron_v3.py          # HF Transformers API server (fallback)
│   └── backtest/
│       └── backtest_nemotron_v3.py       # Step 7: Backtest against val data
├── adapters/
│   └── nemotron-v3/
│       ├── adapter_config.json           # LoRA config (r=8, alpha=16)
│       └── nemotron-v3-lora.gguf         # Step 2 output (5.2 MB)
├── docs/
│   ├── PSP-FAILURE-FIX.md                # GPU PSP failure root cause
│   └── LORA-MERGE-PIPELINE.md            # This document
├── train_nemotron_v3.py                  # Main training script (copy)
├── serve_nemotron_v3.py                  # Inference server (copy)
└── README.md
```

## Build llama.cpp (CPU-only, for merge/quantize)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build-cpu -DGGML_HIP=OFF -DGGML_CUDA=OFF
cmake --build build-cpu --config Release -j$(nproc) -- llama-export-lora llama-quantize
# Binaries in: build-cpu/bin/
```

## Key Discoveries

1. **Ollama 0.32 dropped NemotronForCausalLM** — can't create new models from HF format
2. **Existing v2 GGUFs still work** — architecture tag is preserved in the file
3. **`llama-export-lora` can merge onto ANY GGUF** — dequantizes, applies LoRA, re-quantizes
4. **`convert_lora_to_gguf.py` works with Nemotron** — even though Ollama can't create the model
5. **CPU-only llama.cpp build is sufficient** for merge + quantize (no GPU needed)
6. **LoRA scale = alpha/r** = 16/8 = 2.0 (automatically calculated by export-lora)

## Lessons Learned

- **Always backup working GGUFs** — they may not be recreatable after Ollama updates
- **LoRA adapters are tiny** (5-10 MB) — cheap to backup, expensive to retrain
- **The merge pipeline bypasses Ollama's architecture restrictions** — as long as a base GGUF exists
- **GRUB `amdgpu.runpm=0` is critical** for multi-GPU rigs with PCIe risers (PSP failures)
- **`device_map='auto'` causes GPU imbalance** — manual layer splitting is better for multi-GPU
- **`torch_dtype` is deprecated** in transformers 4.57+ — use `dtype=` instead
- **`init_empty_weights()` is redundant** with `from_pretrained(low_cpu_mem_usage=True)`
- **kimi-k2.6 can be wrong** about model internals — always verify with runtime tests
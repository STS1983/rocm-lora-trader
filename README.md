# ROCm LoRA Trader

Fine-tune QLoRA/LoRA trading models on AMD GPUs (ROCm 7.0).

Tested on 4× AMD Radeon RX 6600 XT with PyTorch 2.11 + ROCm 7.0.

## Why This Exists

Standard QLoRA tutorials assume NVIDIA CUDA. On AMD GPUs with ROCm, `bitsandbytes` 4-bit quantization segfaults. This project provides a **working solution** using fp16 LoRA with multi-GPU model parallelism instead.

## Features

- ✅ **Works on AMD/ROCm** — No segfaults, no bitsandbytes needed
- ✅ **Multi-GPU Training** — Distributes model across 2-4 GPUs automatically
- ✅ **LoRA Fine-Tuning** — Only 0.96% trainable params (29.9M of 3.1B)
- ✅ **Auto Data Conversion** — Converts JSONL training data to TRL format
- ✅ **Merge & Convert** — Merges LoRA weights → GGUF → Ollama model
- ✅ **Auto Retrain** — Weekly retraining pipeline with validation
- ✅ **Trading-Focused** — System prompt for crypto trading signal recognition

## Quick Start

### Prerequisites

```bash
# On your training machine (ROCm required)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
pip install transformers peft trl datasets accelerate
```

### 1. Prepare Training Data

```bash
python convert_data.py \
  --input training-data.jsonl \
  --system-prompt system_prompt.txt \
  --output-dir training/ \
  --split 0.8
```

Training data format (JSONL):
```json
{"prompt": "BTC RSI 45, EMA8 61000 above EMA21 60500, vol_24h 25000000000, uptrend, 24h_change +2.1%, Signal?", "completion": "BUY | Confidence: 75% | Pattern: UPTREND_BUY | Reasoning: RSI neutral, EMA8 above EMA21 confirms uptrend"}
```

### 2. Train

```bash
python train_qlora_v3.py
```

Config options in the script:
- `LoRA_RANK`: 16 (default)
- `LoRA_ALPHA`: 32 (default)
- `LEARNING_RATE`: 2e-4 (default)
- `EPOCHS`: 3 (default)
- `BATCH_SIZE_PER_GPU`: 1 (default, increase if VRAM allows)
- `MAX_SEQUENCE_LENGTH`: 512 (default)

### 3. Merge & Convert to Ollama

```bash
python merge_and_convert.py \
  --adapter-path output/trader-v2/checkpoint-best \
  --model-name trader-v2 \
  --quantize Q4_K_M
```

### 4. Deploy to Ollama

```bash
# Copy GGUF to your Ollama server
ollama create trader-v2 -f Modelfile
ollama run trader-v2
```

## Architecture

```
Base Model: Qwen/Qwen2.5-3B-Instruct (or any compatible model)
           ↓
    LoRA Adapter (rank 16, alpha 32)
    Target Modules: q_proj, k_proj, v_proj, o_proj, 
                    gate_proj, up_proj, down_proj
           ↓
    fp16 Multi-GPU Training (2-4 GPUs)
           ↓
    Merge LoRA → Full Model → GGUF (Q4_K_M)
           ↓
    Ollama Deployment
```

## Multi-GPU Distribution

The model is automatically distributed across available GPUs:
- **2 GPUs**: Layers 0-17 on GPU 0, Layers 18-35 on GPU 1
- **4 GPUs**: Layers distributed evenly across all 4 GPUs

Each GPU gets ~2GB of model weights (for Qwen2.5-3B), leaving ~6GB VRAM for activations and gradients.

## Why Not 4-bit QLoRA?

On AMD GPUs with ROCm, `bitsandbytes` 4-bit quantization causes segfaults due to incompatible CUDA kernels. Our fp16 LoRA approach:

| Approach | VRAM/GPU | Speed | Status |
|----------|----------|-------|--------|
| 4-bit QLoRA | ~4GB | 40 tok/s | ❌ Segfault on ROCm |
| fp16 LoRA (this) | ~6GB | 58s/step | ✅ Works |

## Training Data Format

### Input (JSONL)
```json
{"prompt": "<market context>", "completion": "SIGNAL | Confidence: X% | Pattern: NAME | Reasoning: ..."}
```

### Output (TRL Chat Format)
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Auto-Retrain Pipeline

Set up weekly retraining with the auto-retrain script:

```bash
# Edit config section in auto-retrain.sh
# Then add to crontab:
0 3 * * 0 /path/to/auto-retrain.sh
```

The script:
1. Pulls latest trade data
2. Converts to training format
3. Runs LoRA fine-tuning
4. Validates against backtest data
5. Deploys if validation improves, rolls back if not

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× 8GB (RX 6600 XT) | 4× 8GB (RX 6600 XT) |
| RAM | 16GB | 32GB |
| Storage | 10GB free | 50GB free |
| ROCm | 6.0+ | 7.0+ |

## ROCm Environment

The training script sets:
```python
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"  # For RX 6600 XT
```

Adjust this for your GPU architecture:
- RX 6600 XT: `10.3.0`
- RX 6700 XT: `10.3.0`
- RX 7600: `11.0.0`
- RX 7900 XTX: `11.0.0`

## Use Cases Beyond Trading

The pipeline is completely generic — swap the system prompt and training data and you can fine-tune for any domain:

| Domain | System Prompt | Training Data |
|--------|---------------|---------------|
| **Trading** | Crypto signal recognition | Market indicators + signals |
| **Customer Support** | Help desk assistant | Support tickets + responses |
| **Code Review** | Code analysis assistant | Code snippets + review comments |
| **Medical** | Diagnostic assistant | Symptoms + diagnoses |
| **Legal** | Contract review assistant | Contracts + analysis |
| **Finance** | Risk assessment assistant | Financial data + risk ratings |
| **Education** | Tutor assistant | Questions + explanations |

The pipeline is universal:

```
Your Data (JSONL) → convert_data.py → TRL Format → train_qlora_v3.py → LoRA Adapter → merge_and_convert.py → Ollama Model
```

Every step is swappable. Just provide different `.jsonl` + different `system_prompt.txt` → different model.

### Quick Domain Switch Example

```bash
# Switch from trading to customer support:
# 1. Create your training data
{"prompt": "Customer asks about refund policy", "completion": "Our refund policy allows..."}

# 2. Write your system prompt
echo "You are a helpful customer support assistant." > my_system_prompt.txt

# 3. Convert and train
python convert_data.py --input my-data.jsonl --system-prompt my_system_prompt.txt --output-dir training/
python train_qlora_v3.py  # Uses whatever data is in training/
```

Same hardware, same ROCm workaround, different domain. That's it.

## License

MIT

## Acknowledgments

- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — Base model
- [PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [TRL](https://github.com/huggingface/trl) — Training framework
- [Ollama](https://ollama.ai) — Model deployment
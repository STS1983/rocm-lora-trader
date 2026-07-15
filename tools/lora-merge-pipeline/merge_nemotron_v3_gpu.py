#!/usr/bin/env python3
"""Merge Nemotron LoRA on single GPU, save, convert to GGUF."""
import os, sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # GPU3 (ASUS, direct x16, most stable)

MODEL_NAME = 'nvidia/Nemotron-Mini-4B-Instruct'
ADAPTER_PATH = '/home/nodeadmin/trading-llm/output/nemotron-trader-v3/final'
MERGED_PATH = '/home/nodeadmin/trading-llm/output/nemotron-trader-v3/merged'
GGUF_PATH = '/home/nodeadmin/trading-llm/output/nemotron-trader-v3/gguf'

def main():
    print('=' * 60, flush=True)
    print('Merge Nemotron LoRA → GGUF (Single GPU mode)', flush=True)
    print('=' * 60, flush=True)

    import torch
    print(f'[INFO] CUDA available: {torch.cuda.is_available()}', flush=True)
    print(f'[INFO] Device count: {torch.cuda.device_count()}', flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # ── Step 1: Load base model on GPU ──
    print('\n[STEP 1] Loading base model on GPU...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    print(f'[INFO] Base model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params', flush=True)
    print(f'[INFO] Device map: {model.hf_device_map}', flush=True)

    # ── Step 2: Load + merge LoRA ──
    print('\n[STEP 2] Loading LoRA adapter...', flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print('[INFO] Merging LoRA weights...', flush=True)
    model = model.merge_and_unload()
    print('[INFO] LoRA merged!', flush=True)

    # ── Step 3: Save merged model ──
    print(f'\n[STEP 3] Saving merged model to {MERGED_PATH}...', flush=True)
    if Path(MERGED_PATH).exists():
        shutil.rmtree(MERGED_PATH)
    os.makedirs(MERGED_PATH, exist_ok=True)
    model = model.cpu()  # Move to CPU before saving
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    for f in Path(MERGED_PATH).glob('*'):
        print(f'  {f.name}: {f.stat().st_size/1e6:.1f} MB', flush=True)

    # ── Step 4: Convert to GGUF ──
    print('\n[STEP 4] Converting to GGUF (Q4_K_M)...', flush=True)
    os.makedirs(GGUF_PATH, exist_ok=True)
    gguf_f16 = f'{GGUF_PATH}/nemotron-trader-v3-f16.gguf'
    gguf_q4 = f'{GGUF_PATH}/nemotron-trader-v3-q4_k_m.gguf'

    llama_cpp = '/home/nodeadmin/llama.cpp'
    convert_script = f'{llama_cpp}/convert_hf_to_gguf.py'

    if not Path(convert_script).exists():
        print(f'[ERROR] {convert_script} not found!', flush=True)
        print('[INFO] Skipping GGUF, using HF format for Ollama', flush=True)
        final_gguf = None
    else:
        print('[INFO] Converting to F16 GGUF...', flush=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU for conversion
        r = subprocess.run(['python3', convert_script, MERGED_PATH, '--outfile', gguf_f16, '--outtype', 'f16'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[ERROR] F16 conversion failed: {r.stderr[-500:]}', flush=True)
            sys.exit(1)
        print(f'[INFO] F16 GGUF: {Path(gguf_f16).stat().st_size/1e9:.2f} GB', flush=True)

        # Quantize to Q4_K_M
        quantize_bin = f'{llama_cpp}/build/bin/llama-quantize'
        if not Path(quantize_bin).exists():
            quantize_bin = f'{llama_cpp}/llama-quantize'
        if Path(quantize_bin).exists():
            print('[INFO] Quantizing to Q4_K_M...', flush=True)
            r = subprocess.run([quantize_bin, gguf_f16, gguf_q4, 'Q4_K_M'],
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(f'[INFO] Q4_K_M: {Path(gguf_q4).stat().st_size/1e9:.2f} GB', flush=True)
                final_gguf = gguf_q4
            else:
                print(f'[WARN] Q4_K_M failed, using F16', flush=True)
                final_gguf = gguf_f16
        else:
            print('[WARN] llama-quantize not found, using F16', flush=True)
            final_gguf = gguf_f16

    # ── Step 5: Create Ollama model ──
    print('\n[STEP 5] Creating Ollama model...', flush=True)
    if final_gguf and Path(final_gguf).exists():
        gguf_name = Path(final_gguf).name
        modelfile = f'FROM ./{gguf_name}\n\nPARAMETER temperature 0.3\nPARAMETER top_p 0.85\nPARAMETER top_k 40\nPARAMETER num_predict 128\n'
        modelfile_path = f'{GGUF_PATH}/Modelfile'
        with open(modelfile_path, 'w') as f:
            f.write(modelfile)
        os.chdir(GGUF_PATH)
        r = subprocess.run(['ollama', 'create', 'nemotron-trader-v3', '-f', 'Modelfile'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print('[INFO] Ollama model created: nemotron-trader-v3', flush=True)
        else:
            print(f'[ERROR] Ollama create failed: {r.stderr}', flush=True)
    else:
        print('[INFO] Using HF format for Ollama...', flush=True)
        modelfile = f'FROM {MERGED_PATH}\n\nPARAMETER temperature 0.3\nPARAMETER top_p 0.85\nPARAMETER top_k 40\nPARAMETER num_predict 128\n'
        modelfile_path = f'{MERGED_PATH}/Modelfile'
        with open(modelfile_path, 'w') as f:
            f.write(modelfile)
        os.chdir(MERGED_PATH)
        r = subprocess.run(['ollama', 'create', 'nemotron-trader-v3', '-f', 'Modelfile'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print('[INFO] Ollama model created: nemotron-trader-v3', flush=True)
        else:
            print(f'[ERROR] Ollama create failed: {r.stderr}', flush=True)

    info = {
        'base_model': MODEL_NAME,
        'adapter': ADAPTER_PATH,
        'merged': MERGED_PATH,
        'gguf': final_gguf,
        'quantization': 'Q4_K_M' if final_gguf and 'q4' in final_gguf else 'F16',
        'ollama_model': 'nemotron-trader-v3',
        'timestamp': datetime.now().isoformat(),
        'status': 'SUCCESS',
    }
    with open(f'{GGUF_PATH}/conversion_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print('\n' + '=' * 60, flush=True)
    print('✅ DONE! nemotron-trader-v3 in Ollama', flush=True)
    print('=' * 60, flush=True)

if __name__ == '__main__':
    main()

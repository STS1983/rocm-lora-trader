#!/usr/bin/env python3
"""
Merge LoRA adapter with base model — CPU-only to avoid ROCm multi-GPU crashes.
Then convert to GGUF for Ollama deployment.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only

def main():
    print("=" * 60)
    print("Merge LoRA + Convert to GGUF (CPU-only mode)")
    print("=" * 60)
    
    import torch
    print(f"[INFO] Using device: CPU (CUDA disabled)")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Paths
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_PATH = "/home/nodeadmin/trading-llm/output/trader-v2/final-adapter"
    MERGED_PATH = "/home/nodeadmin/trading-llm/output/trader-v2/merged"
    GGUF_PATH = "/home/nodeadmin/trading-llm/output/trader-v2/gguf"
    TRAINING_DIR = "/home/nodeadmin/trading-llm/training"
    
    # Load system prompt
    with open(f"{TRAINING_DIR}/system_prompt.txt") as f:
        system_prompt = f.read().strip()
    
    # ============ Step 1: Load base model on CPU ============
    print("\n[STEP 1] Loading base model on CPU...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",  # Force CPU
        low_cpu_mem_usage=True,
    )
    
    print(f"[INFO] Base model loaded on CPU")
    print(f"[INFO] Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # ============ Step 2: Load and merge LoRA ============
    print("\n[STEP 2] Loading LoRA adapter...")
    
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    print("[INFO] Merging LoRA weights...")
    model = model.merge_and_unload()
    
    print("[INFO] LoRA merged successfully!")
    
    # ============ Step 3: Save merged model ============
    print(f"\n[STEP 3] Saving merged model to {MERGED_PATH}...")
    
    # Clean up any partial saves from previous crash
    import shutil
    if Path(MERGED_PATH).exists():
        shutil.rmtree(MERGED_PATH)
    
    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    
    print(f"[INFO] Merged model saved to {MERGED_PATH}")
    
    # Verify files
    merged_files = list(Path(MERGED_PATH).glob("*"))
    print(f"[INFO] Saved {len(merged_files)} files")
    for f in merged_files:
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    
    # ============ Step 4: Convert to GGUF ============
    print("\n[STEP 4] Converting to GGUF format...")
    
    os.makedirs(GGUF_PATH, exist_ok=True)
    
    gguf_f16 = f"{GGUF_PATH}/trader-v2-f16.gguf"
    
    # Try llama.cpp conversion
    llama_cpp_path = "/home/nodeadmin/trading-llm/llama.cpp"
    convert_script = f"{llama_cpp_path}/convert_hf_to_gguf.py"
    
    if not Path(convert_script).exists():
        print("[INFO] Installing llama.cpp for GGUF conversion...")
        os.system(f"cd /home/nodeadmin/trading-llm && git clone https://github.com/ggerganov/llama.cpp.git 2>/dev/null || true")
    
    if Path(convert_script).exists():
        print(f"[INFO] Converting using llama.cpp script...")
        result = os.system(f"python3 {convert_script} {MERGED_PATH} --outfile {gguf_f16} --outtype f16")
        if result == 0:
            print(f"[INFO] F16 GGUF saved to {gguf_f16}")
        else:
            print(f"[WARN] llama.cpp conversion failed (exit {result}), trying alternative...")
    
    # Check if GGUF was created
    if not Path(gguf_f16).exists():
        # Try gguf Python package
        print("[INFO] Trying gguf Python package...")
        os.system("pip install gguf 2>/dev/null")
        result = os.system(f"python3 -m gguf.convert {MERGED_PATH} --outfile {gguf_f16} --outtype f16")
        if result == 0:
            print(f"[INFO] F16 GGUF saved to {gguf_f16}")
        else:
            print("[WARN] GGUF conversion failed. Will use HuggingFace format for Ollama.")
    
    # ============ Step 5: Create Ollama Modelfile ============
    print("\n[STEP 5] Creating Ollama Modelfile...")
    
    if Path(gguf_f16).exists():
        gguf_file = "trader-v2-f16.gguf"
        print(f"[INFO] Using GGUF: {gguf_file}")
    else:
        gguf_file = None
        print("[INFO] No GGUF file — will use HuggingFace format for Ollama")
    
    if gguf_file:
        modelfile_content = f"""FROM ./{gguf_file}

SYSTEM \"\"\"{system_prompt}\"\"\"

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
{{{{- else }}}}
<|im_start|>assistant
{{{{- end }}}}\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER top_k 40
PARAMETER num_predict 128
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
    else:
        modelfile_content = f"""FROM {MERGED_PATH}

SYSTEM \"\"\"{system_prompt}\"\"\"

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
{{{{- else }}}}
<|im_start|>assistant
{{{{- end }}}}\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER top_k 40
PARAMETER num_predict 128
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
    
    modelfile_path = f"{GGUF_PATH}/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"[INFO] Modelfile saved to {modelfile_path}")
    
    # Save conversion info
    conversion_info = {
        "base_model": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "merged_path": MERGED_PATH,
        "gguf_created": Path(gguf_f16).exists() if Path(gguf_f16).exists() else False,
        "modelfile": modelfile_path,
        "timestamp": datetime.now().isoformat(),
        "method": "cpu-only-merge",
        "status": "SUCCESS" if Path(MERGED_PATH).exists() else "PARTIAL",
    }
    
    with open(f"{GGUF_PATH}/conversion_info.json", "w") as f:
        json.dump(conversion_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 60)
    
    if Path(gguf_f16).exists():
        print(f"\nTo create Ollama model:")
        print(f"  cd {GGUF_PATH}")
        print(f"  ollama create trader-v2 -f Modelfile")
    else:
        print(f"\nGGUF conversion not available. Use HuggingFace format:")
        print(f"  cd {GGUF_PATH}")
        print(f"  ollama create trader-v2 -f Modelfile")


if __name__ == "__main__":
    main()
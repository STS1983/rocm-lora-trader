#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and convert to GGUF for Ollama.
v2: Works with fp16 LoRA (not QLoRA) since ROCm doesn't support 4-bit quantization.

Steps:
1. Load base model (full fp16)
2. Load and merge LoRA adapter
3. Save merged model
4. Convert to GGUF format (quantize to Q4_K_M for Ollama)
5. Create Ollama Modelfile with trading system prompt
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 60)
    print("Merge LoRA + Convert to GGUF for Ollama")
    print("=" * 60)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    # Paths
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_PATH = "/home/nodeadmin/trading-llm/output/trader-v2/final-adapter"
    MERGED_PATH = "/home/nodeadmin/trading-llm/output/trader-v2/merged"
    GGUF_PATH = "/home/nodeadmin/trading-llm/output/trader-v2/gguf"
    TRAINING_DIR = "/home/nodeadmin/trading-llm/training"
    
    # Load system prompt
    with open(f"{TRAINING_DIR}/system_prompt.txt") as f:
        system_prompt = f.read().strip()
    
    # ============ Step 1: Load base model ============
    print("\n[STEP 1] Loading base model (fp16)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"[INFO] Base model loaded")
    
    # ============ Step 2: Load and merge LoRA ============
    print("\n[STEP 2] Loading LoRA adapter...")
    
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    print("[INFO] Merging LoRA weights...")
    model = model.merge_and_unload()
    
    print("[INFO] LoRA merged successfully!")
    
    # ============ Step 3: Save merged model ============
    print(f"\n[STEP 3] Saving merged model to {MERGED_PATH}...")
    
    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    
    print(f"[INFO] Merged model saved to {MERGED_PATH}")
    
    # Free GPU memory for conversion
    del model
    torch.cuda.empty_cache()
    
    # ============ Step 4: Convert to GGUF ============
    print("\n[STEP 4] Converting to GGUF format...")
    
    os.makedirs(GGUF_PATH, exist_ok=True)
    
    gguf_f16 = f"{GGUF_PATH}/trader-v2-f16.gguf"
    gguf_q4 = f"{GGUF_PATH}/trader-v2-q4_k_m.gguf"
    
    # Method 1: Use llama.cpp convert script
    llama_cpp_path = "/home/nodeadmin/trading-llm/llama.cpp"
    
    # Install llama.cpp if not present
    if not Path(f"{llama_cpp_path}/convert_hf_to_gguf.py").exists():
        print("[INFO] Installing llama.cpp for GGUF conversion...")
        os.system(f"cd /home/nodeadmin/trading-llm && git clone https://github.com/ggerganov/llama.cpp.git 2>/dev/null || true")
    
    # Try Python conversion
    convert_script = f"{llama_cpp_path}/convert_hf_to_gguf.py"
    if Path(convert_script).exists():
        print(f"[INFO] Converting using llama.cpp script...")
        result = os.system(f"python3 {convert_script} {MERGED_PATH} --outfile {gguf_f16} --outtype f16")
        if result == 0:
            print(f"[INFO] F16 GGUF saved to {gguf_f16}")
        else:
            print(f"[WARN] Conversion with llama.cpp failed (exit {result}), trying alternative...")
    
    # Method 2: Use gguf Python package
    if not Path(gguf_f16).exists():
        print("[INFO] Trying gguf Python package...")
        os.system("pip install gguf 2>/dev/null")
        result = os.system(f"python3 -m gguf.convert {MERGED_PATH} --outfile {gguf_f16} --outtype f16")
        if result == 0:
            print(f"[INFO] F16 GGUF saved to {gguf_f16}")
        else:
            # Method 3: Use llama-cpp-python convert
            print("[INFO] Trying llama-cpp-python conversion...")
            try:
                from llama_cpp import convert
                convert(MERGED_PATH, gguf_f16)
                print(f"[INFO] F16 GGUF saved to {gguf_f16}")
            except Exception as e:
                print(f"[WARN] All GGUF conversion methods failed: {e}")
    
    # Quantize to Q4_K_M if llama-quantize is available
    if Path(gguf_f16).exists():
        quantize_bin = f"{llama_cpp_path}/build/bin/llama-quantize"
        if Path(quantize_bin).exists():
            print("[INFO] Quantizing to Q4_K_M...")
            os.system(f"{quantize_bin} {gguf_f16} {gguf_q4} Q4_K_M")
            if Path(gguf_q4).exists():
                print(f"[INFO] Q4_K_M GGUF saved to {gguf_q4}")
            else:
                print("[WARN] Q4_K_M quantization failed, F16 will be used")
        else:
            print("[INFO] llama-quantize not built, skipping Q4_K_M")
            print("[INFO] You can build it with: cd llama.cpp && cmake -B build && cmake --build build")
    
    # ============ Step 5: Create Ollama Modelfile ============
    print("\n[STEP 5] Creating Ollama Modelfile...")
    
    # Use Q4_K_M if available, else F16
    if Path(gguf_q4).exists():
        gguf_file = "trader-v2-q4_k_m.gguf"
    elif Path(gguf_f16).exists():
        gguf_file = "trader-v2-f16.gguf"
    else:
        # If GGUF conversion failed, use the Ollama/HF model directly
        gguf_file = None
        print("[WARN] No GGUF file available. Will use HuggingFace model path in Modelfile.")
    
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
        # Fallback: use HuggingFace path
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
    
    # ============ Instructions ============
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 60)
    
    if gguf_file and Path(f"{GGUF_PATH}/{gguf_file}").exists():
        print(f"\nTo create Ollama model:")
        print(f"  cd {GGUF_PATH}")
        print(f"  ollama create trader-v2 -f Modelfile")
        print(f"\nTo test:")
        print(f"  ollama run trader-v2")
    elif Path(MERGED_PATH).exists():
        print(f"\nMerged model saved to: {MERGED_PATH}")
        print(f"\nTo create Ollama model from HuggingFace format:")
        print(f"  cd {GGUF_PATH}")
        print(f"  ollama create trader-v2 -f Modelfile")
    
    # Save conversion info
    conversion_info = {
        "base_model": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "merged_path": MERGED_PATH,
        "gguf_f16": str(Path(gguf_f16)) if Path(gguf_f16).exists() else "NOT_CREATED",
        "gguf_q4_k_m": str(Path(gguf_q4).exists()) if Path(gguf_q4).exists() else "NOT_CREATED",
        "modelfile": modelfile_path,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{GGUF_PATH}/conversion_info.json", "w") as f:
        json.dump(conversion_info, f, indent=2)


if __name__ == "__main__":
    main()
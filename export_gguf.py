#!/usr/bin/env python3
"""Export fine-tuned LoRA adapter merged with base model to GGUF format.

Steps:
  1. Load base model (16-bit) + LoRA adapter
  2. Merge LoRA weights into base model
  3. Save merged model as safetensors
  4. Convert to GGUF F16 via llama.cpp
  5. Quantize to Q5_K_M
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = BASE_DIR / "output" / "lora_adapter"
MERGED_DIR = BASE_DIR / "output" / "merged_model"
GGUF_DIR = BASE_DIR / "output" / "gguf"
BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"
MAX_SEQ_LENGTH = 8192

LLAMA_CPP = Path.home() / "workspace" / "tj" / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP / "build" / "bin" / "llama-quantize"

QUANTIZATION = "Q5_K_M"


def main():
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    f16_gguf = GGUF_DIR / "model-f16.gguf"
    quantized_gguf = GGUF_DIR / f"model-{QUANTIZATION}.gguf"

    # Step 1-3: Load, merge, save (on CPU - 8B model in fp16 needs ~16GB RAM)
    if not MERGED_DIR.exists():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model (16-bit, CPU): unsloth/Qwen3-8B")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Qwen3-8B",
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-8B")

        print(f"Loading adapter: {ADAPTER_DIR}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), device_map="cpu")

        print("Merging LoRA weights...")
        model = model.merge_and_unload()

        print(f"Saving merged model to {MERGED_DIR}")
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"Merged model saved to {MERGED_DIR}")

        del model

    else:
        print(f"Merged model already exists at {MERGED_DIR}, skipping merge step")

    # Step 4: Convert to GGUF F16
    if not f16_gguf.exists():
        print(f"Converting to GGUF F16...")
        cmd = [
            sys.executable, str(CONVERT_SCRIPT),
            str(MERGED_DIR),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ]
        subprocess.run(cmd, check=True)
        print(f"F16 GGUF saved to {f16_gguf}")
    else:
        print(f"F16 GGUF already exists, skipping conversion")

    # Step 5: Quantize
    if not quantized_gguf.exists():
        print(f"Quantizing to {QUANTIZATION}...")
        cmd = [
            str(QUANTIZE_BIN),
            str(f16_gguf),
            str(quantized_gguf),
            QUANTIZATION,
        ]
        subprocess.run(cmd, check=True)
        print(f"Quantized GGUF saved to {quantized_gguf}")
    else:
        print(f"Quantized GGUF already exists")

    # Print file sizes
    for f in [f16_gguf, quantized_gguf]:
        if f.exists():
            size_gb = f.stat().st_size / (1024**3)
            print(f"  {f.name}: {size_gb:.2f} GB")

    print("Export complete!")


if __name__ == "__main__":
    main()

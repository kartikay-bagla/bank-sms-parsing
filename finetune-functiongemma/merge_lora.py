#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for GGUF conversion.

Usage:
    python merge_lora.py --adapter_path ./checkpoints --output_dir ./merged_model
"""

import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model", default="google/functiongemma-270m-it", help="Base model name"
    )
    parser.add_argument(
        "--adapter_path", default="./checkpoints", help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output_dir",
        default="./merged_model",
        help="Output directory for merged model",
    )

    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="cpu",  # Use CPU to avoid VRAM issues
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done! Now convert to GGUF:")
    print(
        f"  python convert_hf_to_gguf.py {args.output_dir} "
        f"--outfile functiongemma-sms.gguf"
    )


if __name__ == "__main__":
    main()

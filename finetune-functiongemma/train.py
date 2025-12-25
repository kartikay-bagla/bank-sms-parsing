#!/usr/bin/env python3
"""
Fine-tune FunctionGemma for SMS transaction extraction.

Uses Hugging Face TRL's SFTTrainer with optional LoRA support.
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_dataset_from_jsonl(path: str | Path) -> Dataset:
    """Load a JSONL file where each line is a conversation dict."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)


def is_amd_gpu() -> bool:
    """Check if running on AMD GPU (ROCm)."""
    if not torch.cuda.is_available():
        return False
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        return any(
            x in gpu_name for x in ["amd", "radeon", "mi100", "mi200", "mi250", "mi300"]
        )
    except Exception:
        # Check for HIP (ROCm's CUDA-like interface)
        return hasattr(torch.version, "hip") and torch.version.hip is not None


def get_training_args(args) -> SFTConfig:
    """Create training configuration."""

    amd_gpu = is_amd_gpu()

    # Detect dtype - AMD GPUs with ROCm support bf16 on RDNA3/CDNA2+
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16, use_fp16 = True, False
        else:
            use_bf16, use_fp16 = False, True
    else:
        use_bf16, use_fp16 = False, False

    # Optimizer selection
    # - adamw_torch_fused can have issues on ROCm, use regular adamw_torch
    # - For NVIDIA, fused is faster
    if torch.cuda.is_available() and not amd_gpu:
        optimizer = "adamw_torch_fused"
    else:
        optimizer = "adamw_torch"

    if amd_gpu:
        print("🔴 AMD GPU detected - using ROCm-compatible settings")

    return SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        packing=False,  # Don't pack multiple samples
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=optimizer,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/training_logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


def setup_lora(model, args):
    """Configure LoRA for parameter-efficient fine-tuning."""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune FunctionGemma for SMS transaction extraction"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing train.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--output_dir", default="./checkpoints", help="Output directory for model"
    )

    # Model arguments
    parser.add_argument(
        "--base_model",
        default="google/functiongemma-270m-it",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Hub arguments
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        default=None,
        help="Model ID on Hub (e.g., 'username/model-name')",
    )

    args = parser.parse_args()

    # Load data
    data_dir = Path(args.data_dir)
    print(f"Loading datasets from {data_dir}...")

    train_dataset = load_dataset_from_jsonl(data_dir / "train.jsonl")
    eval_dataset = load_dataset_from_jsonl(data_dir / "test.jsonl")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.base_model}")

    # Note: Flash attention is not available for AMD GPUs
    # Always use "eager" attention which works on all platforms
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",  # "flash_attention_2" only works on NVIDIA
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    if is_amd_gpu():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Apply LoRA if requested
    if args.use_lora:
        print("\nApplying LoRA configuration...")
        model = setup_lora(model, args)

    # Preview formatted example
    print("\n--- Sample formatted input ---")
    sample = train_dataset[0]
    formatted = tokenizer.apply_chat_template(
        sample["messages"],
        tools=sample["tools"],
        add_generation_prompt=False,
        tokenize=False,
    )
    print(formatted[:1000] + "..." if len(formatted) > 1000 else formatted)

    # Training config
    training_args = get_training_args(args)

    # Custom data collator that handles the tools field
    def formatting_func(example):
        """Format a single example using the chat template with tools."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tools=example["tools"],
            add_generation_prompt=False,
            tokenize=False,
        )
        return text

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Train!
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save the tools config alongside the model
    tools_config = {
        "system_prompt": sample["messages"][0]["content"],
        "tools": sample["tools"],
    }
    with open(Path(args.output_dir) / "tools_config.json", "w") as f:
        json.dump(tools_config, f, indent=2)

    print(f"\nTraining complete! Model saved to: {args.output_dir}")
    print(f"View training logs: tensorboard --logdir {args.output_dir}/training_logs")

    if args.push_to_hub:
        print("\nPushing to Hugging Face Hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()

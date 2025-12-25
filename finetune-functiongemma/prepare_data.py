#!/usr/bin/env python3
"""
Prepare distilled SMS dataset for FunctionGemma fine-tuning.

Transforms the output from a larger model into the conversational
tool-calling format expected by FunctionGemma.
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter


# Tool definitions matching what we'll use at inference time
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_transaction",
            "description": (
                "Extract transaction details from a banking SMS message "
                "for hledger entry creation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "The source account or payment method "
                            "(e.g., 'ICICI Bank XX811', 'HDFC FASTag')"
                        ),
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency code (e.g., 'INR', 'USD')",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transaction amount as a number",
                    },
                    "date": {
                        "type": "string",
                        "description": "Transaction date in YYYY-MM-DD format",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The payee or recipient of the transaction",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["debit", "credit"],
                        "description": (
                            "Whether money was debited (sent) or credited (received)"
                        ),
                    },
                },
                "required": [
                    "source",
                    "currency",
                    "amount",
                    "date",
                    "destination",
                    "type",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_message",
            "description": (
                "Skip messages that are not financial transactions "
                "(spam, OTPs, promotional, etc.)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Brief explanation of why this message is not a transaction"
                        ),
                    }
                },
                "required": ["reason"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a financial transaction extractor. Analyze SMS messages and:
1. If the message describes a completed financial transaction (money sent, \
received, debited, or credited), use extract_transaction to capture the details.
2. If the message is not a transaction (OTP, promotional, application status, \
payment request, etc.), use skip_message.

Only extract actual completed transactions with concrete amounts, \
not payment requests or pending transactions."""


def create_conversation(sms_body: str, output: dict) -> dict | None:
    """Convert a single example to FunctionGemma conversation format."""

    classification = output.get("classification")

    if classification == "error":
        return None  # Skip errored entries

    if classification == "relevant":
        # Transaction message -> extract_transaction tool
        tool_args = {
            "source": output.get("source", "Unknown"),
            "currency": output.get("currency", "INR"),
            "amount": output.get("amount", 0),
            "date": output.get("date"),  # Can be null
            "destination": output.get("destination", "Unknown"),
            "type": output.get("type", "debit"),
        }
        tool_name = "extract_transaction"

    elif classification == "irrelevant":
        # Non-transaction -> skip_message tool
        tool_args = {"reason": output.get("reason", "Not a financial transaction")}
        tool_name = "skip_message"

    else:
        return None  # Unknown classification

    return {
        "messages": [
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sms_body},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_args},
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }


def load_and_filter_data(
    input_path: str, min_confidence: int = 0
) -> tuple[list[dict], Counter]:
    """Load JSONL data and filter by confidence score."""

    data = []
    stats = Counter()

    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            output = item.get("output", {})
            classification = output.get("classification", "unknown")
            score = output.get("score", 0)

            stats[f"total_{classification}"] += 1

            # Filter by confidence
            if score < min_confidence:
                stats["filtered_low_confidence"] += 1
                continue

            # Skip errors
            if classification == "error":
                stats["filtered_error"] += 1
                continue

            sms_body = item.get("source", {}).get("body", "")
            if not sms_body:
                stats["filtered_no_body"] += 1
                continue

            conversation = create_conversation(sms_body, output)
            if conversation:
                data.append(
                    {
                        "conversation": conversation,
                        "classification": classification,
                        "original_index": item.get("index"),
                        "confidence": score,
                    }
                )
                stats[f"kept_{classification}"] += 1

    return data, stats


def split_data(
    data: list[dict], test_size: float, seed: int = 42
) -> tuple[list, list]:
    """Split data into train/test, stratified by classification."""

    random.seed(seed)

    # Separate by classification
    relevant = [d for d in data if d["classification"] == "relevant"]
    irrelevant = [d for d in data if d["classification"] == "irrelevant"]

    random.shuffle(relevant)
    random.shuffle(irrelevant)

    # Split each class
    rel_split = int(len(relevant) * (1 - test_size))
    irr_split = int(len(irrelevant) * (1 - test_size))

    train = relevant[:rel_split] + irrelevant[:irr_split]
    test = relevant[rel_split:] + irrelevant[irr_split:]

    random.shuffle(train)
    random.shuffle(test)

    return train, test


def save_dataset(data: list[dict], output_path: Path | str):
    """Save dataset in JSONL format (just the conversation part)."""

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item["conversation"]) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SMS dataset for FunctionGemma fine-tuning"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL file from distillation"
    )
    parser.add_argument(
        "--output_dir", default="./data", help="Output directory for processed data"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data for testing (0.0-1.0)",
    )
    parser.add_argument(
        "--min_confidence",
        type=int,
        default=90,
        help="Minimum confidence score (0-100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input}...")
    data, stats = load_and_filter_data(args.input, args.min_confidence)

    print("\n--- Dataset Statistics ---")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    print(f"\nTotal usable examples: {len(data)}")

    # Split
    train_data, test_data = split_data(data, args.test_size, args.seed)

    train_rel = sum(1 for d in train_data if d["classification"] == "relevant")
    train_irr = sum(1 for d in train_data if d["classification"] == "irrelevant")
    print(f"\nTrain set: {len(train_data)} examples")
    print(f"  - Relevant (transactions): {train_rel}")
    print(f"  - Irrelevant (skip): {train_irr}")

    test_rel = sum(1 for d in test_data if d["classification"] == "relevant")
    test_irr = sum(1 for d in test_data if d["classification"] == "irrelevant")
    print(f"\nTest set: {len(test_data)} examples")
    print(f"  - Relevant (transactions): {test_rel}")
    print(f"  - Irrelevant (skip): {test_irr}")

    # Save
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    save_dataset(train_data, train_path)
    save_dataset(test_data, test_path)

    print(f"\nSaved train data to: {train_path}")
    print(f"Saved test data to: {test_path}")

    # Also save the tool definitions for inference
    tools_path = output_dir / "tools.json"
    with open(tools_path, "w") as f:
        json.dump({"system_prompt": SYSTEM_PROMPT, "tools": TOOLS}, f, indent=2)
    print(f"Saved tool definitions to: {tools_path}")

    # Save a sample for verification
    sample_path = output_dir / "sample.json"
    with open(sample_path, "w") as f:
        json.dump(train_data[0]["conversation"], f, indent=2)
    print(f"Saved sample conversation to: {sample_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Select the best model output from multi-model classification results.

Priority order: opus > sonnet > gpt-oss-120b > haiku
"""

import json
from pathlib import Path

MODEL_PRIORITY = ["opus", "sonnet", "gpt-oss-120b", "haiku"]


def select_best_output(input_file: str, output_file: str | None = None) -> None:
    """Select the best model output for each record."""
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path.with_suffix(".final.jsonl")

    results = []

    with open(input_path, "r") as f:
        for line in f:
            record = json.loads(line)
            outputs = record.get("output", {})

            # Find best available model
            best_model = None
            best_output = None
            for model in MODEL_PRIORITY:
                if model in outputs:
                    best_model = model
                    best_output = outputs[model]
                    break

            if best_output is None:
                print(f"Warning: No valid model output for index {record.get('index')}")
                continue

            final_record = {
                "index": record["index"],
                "source": record["source"],
                "model": best_model,
                "output": best_output,
            }
            results.append(final_record)

    # Sort by index
    results.sort(key=lambda x: x["index"])

    # Write output
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"Processed {len(results)} records")
    print(f"Output written to {output_path}")

    # Print model distribution
    model_counts: dict[str, int] = {}
    for record in results:
        model = record["model"]
        model_counts[model] = model_counts.get(model, 0) + 1

    print("\nModel distribution:")
    for model in MODEL_PRIORITY:
        if model in model_counts:
            print(f"  {model}: {model_counts[model]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select best model output")
    parser.add_argument(
        "--input",
        default="output/transaction_messages.output.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL file (default: input.final.jsonl)",
    )
    args = parser.parse_args()

    select_best_output(args.input, args.output)

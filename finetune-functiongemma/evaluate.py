#!/usr/bin/env python3
"""
Evaluate fine-tuned FunctionGemma model on test data.

Measures:
- Tool selection accuracy (extract_transaction vs skip_message)
- Field extraction accuracy for transactions
- Overall success rate
"""

import json
import argparse
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test_data(path: str) -> list[dict]:
    """Load test JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_tool_call(output: str) -> dict | None:
    """Parse the model's tool call from its output."""

    # Look for FunctionGemma's tool call format
    # <start_function_call>call:tool_name{args}<end_function_call>
    pattern = r"<start_function_call>call:(\w+)\{(.+?)\}<end_function_call>"
    match = re.search(pattern, output, re.DOTALL)

    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2)

    # Parse the arguments (FunctionGemma uses a custom format with <escape> tags)
    # e.g., {source:<escape>ICICI Bank<escape>,amount:<escape>450<escape>}
    args = {}

    # Extract key-value pairs
    kv_pattern = r"(\w+):<escape>(.+?)<escape>"
    for kv_match in re.finditer(kv_pattern, args_str):
        key = kv_match.group(1)
        value = kv_match.group(2)

        # Try to convert numbers
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        args[key] = value

    return {"name": tool_name, "arguments": args}


def get_expected_tool_call(sample: dict) -> dict:
    """Extract the expected tool call from a sample."""
    tool_call = sample["messages"][2]["tool_calls"][0]
    return {
        "name": tool_call["function"]["name"],
        "arguments": tool_call["function"]["arguments"],
    }


def evaluate_model(
    model,
    tokenizer,
    test_data: list[dict],
    tools: list[dict],
    verbose: bool = False,
) -> dict:
    """Run evaluation and return metrics."""

    results = {
        "total": len(test_data),
        "tool_correct": 0,
        "tool_wrong": 0,
        "no_tool_called": 0,
        "field_accuracy": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_expected_tool": defaultdict(lambda: {"correct": 0, "total": 0}),
        "examples": [],
    }

    for idx, sample in enumerate(test_data):
        # Prepare input (system + user message only)
        messages = [
            sample["messages"][0],  # system/developer
            sample["messages"][1],  # user (SMS)
        ]

        expected = get_expected_tool_call(sample)

        # Generate
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs.to(model.device),
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Greedy decoding for reproducibility
            )

        # Decode only the generated part
        input_len = len(inputs["input_ids"][0])
        generated = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        # Parse the tool call
        predicted = parse_tool_call(generated)

        # Evaluate
        example_result = {
            "idx": idx,
            "sms": sample["messages"][1]["content"][:100] + "...",
            "expected_tool": expected["name"],
            "predicted_tool": predicted["name"] if predicted else None,
            "correct": False,
        }

        results["by_expected_tool"][expected["name"]]["total"] += 1

        if predicted is None:
            results["no_tool_called"] += 1
            status = "❌ no tool"
        elif predicted["name"] != expected["name"]:
            results["tool_wrong"] += 1
            status = f"❌ wrong tool ({predicted['name']})"
        else:
            results["tool_correct"] += 1
            results["by_expected_tool"][expected["name"]]["correct"] += 1
            example_result["correct"] = True
            status = "✅"

            # For correct tool calls, check field accuracy
            if expected["name"] == "extract_transaction":
                for field in ["source", "amount", "date", "destination", "type"]:
                    results["field_accuracy"][field]["total"] += 1
                    expected_val = expected["arguments"].get(field)
                    predicted_val = predicted["arguments"].get(field)

                    # Fuzzy match for strings, exact for numbers
                    if isinstance(expected_val, (int, float)):
                        match = expected_val == predicted_val
                    elif expected_val is None:
                        match = predicted_val is None
                    else:
                        # Case-insensitive substring match for strings
                        exp_lower = str(expected_val).lower()
                        pred_lower = str(predicted_val).lower()
                        match = exp_lower in pred_lower or pred_lower in exp_lower

                    if match:
                        results["field_accuracy"][field]["correct"] += 1

        results["examples"].append(example_result)

        if verbose or not example_result["correct"]:
            print(f"\n[{idx + 1}/{len(test_data)}] {status}")
            print(f"  SMS: {sample['messages'][1]['content'][:80]}...")
            print(f"  Expected: {expected['name']}")
            if predicted:
                print(f"  Predicted: {predicted['name']}")
                if verbose and predicted["name"] == expected["name"]:
                    print(f"  Args: {predicted['arguments']}")

    return results


def print_report(results: dict):
    """Print evaluation summary."""

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    total = results["total"]
    correct = results["tool_correct"]
    pct = 100 * correct / total

    print(f"\n📊 Overall Tool Selection Accuracy: {correct}/{total} ({pct:.1f}%)")
    print(f"   - Correct tool: {results['tool_correct']}")
    print(f"   - Wrong tool: {results['tool_wrong']}")
    print(f"   - No tool called: {results['no_tool_called']}")

    print("\n📋 By Expected Tool:")
    for tool, stats in results["by_expected_tool"].items():
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"   - {tool}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    if results["field_accuracy"]:
        print("\n📝 Field Extraction Accuracy (correct extract_transaction calls):")
        for field, stats in results["field_accuracy"].items():
            if stats["total"] > 0:
                acc = 100 * stats["correct"] / stats["total"]
                print(f"   - {field}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned FunctionGemma model"
    )
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_file", required=True, help="Path to test JSONL file")
    parser.add_argument(
        "--tools_config", default=None, help="Path to tools config JSON (optional)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples to evaluate"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print all examples, not just errors"
    )
    parser.add_argument("--output", default=None, help="Save detailed results to JSON")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    test_data = load_test_data(args.test_file)

    if args.max_samples:
        test_data = test_data[: args.max_samples]

    print(f"Evaluating on {len(test_data)} samples...")

    # Get tools from config or test data
    if args.tools_config:
        with open(args.tools_config) as f:
            config = json.load(f)
            tools = config["tools"]
    else:
        tools = test_data[0]["tools"]

    # Run evaluation
    results = evaluate_model(model, tokenizer, test_data, tools, verbose=args.verbose)

    # Print report
    print_report(results)

    # Save detailed results
    if args.output:
        # Remove non-serializable parts
        output_results = {
            "total": results["total"],
            "tool_correct": results["tool_correct"],
            "tool_wrong": results["tool_wrong"],
            "no_tool_called": results["no_tool_called"],
            "accuracy": results["tool_correct"] / results["total"],
            "field_accuracy": {
                k: dict(v) for k, v in results["field_accuracy"].items()
            },
            "by_expected_tool": {
                k: dict(v) for k, v in results["by_expected_tool"].items()
            },
        }

        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

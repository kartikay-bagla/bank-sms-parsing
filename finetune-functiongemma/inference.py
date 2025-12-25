#!/usr/bin/env python3
"""
Run inference with the fine-tuned FunctionGemma model.

Accepts SMS messages and outputs structured transaction data or skip decision.
"""

import json
import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default tools config (can be overridden by loading from model dir)
DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_transaction",
            "description": "Extract transaction details from a banking SMS message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "currency": {"type": "string"},
                    "amount": {"type": "number"},
                    "date": {"type": "string"},
                    "destination": {"type": "string"},
                    "type": {"type": "string", "enum": ["debit", "credit"]},
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
            "description": "Skip messages that are not financial transactions.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

DEFAULT_SYSTEM_PROMPT = """\
You are a financial transaction extractor. Analyze SMS messages and:
1. If the message describes a completed financial transaction, \
use extract_transaction.
2. If the message is not a transaction (OTP, promotional, etc.), \
use skip_message."""


def parse_tool_call(output: str) -> dict | None:
    """Parse FunctionGemma's tool call format."""
    pattern = r"<start_function_call>call:(\w+)\{(.+?)\}<end_function_call>"
    match = re.search(pattern, output, re.DOTALL)

    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2)

    args = {}
    kv_pattern = r"(\w+):<escape>(.+?)<escape>"
    for kv_match in re.finditer(kv_pattern, args_str):
        key = kv_match.group(1)
        value = kv_match.group(2)

        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        args[key] = value

    return {"tool": tool_name, "arguments": args}


def to_hledger_entry(result: dict, sms: str | None = None) -> str:
    """Convert extracted transaction to hledger format."""

    if result["tool"] == "skip_message":
        return f"; Skipped: {result['arguments'].get('reason', 'Not a transaction')}"

    args = result["arguments"]

    date = args.get("date", datetime.now().strftime("%Y-%m-%d"))
    payee = args.get("destination", "Unknown")
    amount = args.get("amount", 0)
    currency = args.get("currency", "INR")
    tx_type = args.get("type", "debit")
    source = args.get("source", "assets:bank:unknown")

    # Clean up account names
    source_account = source.lower().replace(" ", ":").replace("xx", "")
    if "icici" in source_account:
        source_account = "assets:bank:icici"
    elif "hdfc" in source_account:
        source_account = "assets:bank:hdfc"
    elif "sbi" in source_account:
        source_account = "assets:bank:sbi"
    elif "fastag" in source_account.lower():
        source_account = "assets:fastag"
    else:
        source_account = "assets:bank:unknown"

    # Format amount with currency symbol
    if currency == "INR":
        amount_str = f"₹{amount:,.2f}"
    else:
        amount_str = f"{amount:,.2f} {currency}"

    # Build the entry
    if tx_type == "debit":
        entry = f"""{date} {payee}
    expenses:uncategorized    {amount_str}
    {source_account}"""
    else:  # credit
        entry = f"""{date} {payee}
    {source_account}    {amount_str}
    income:uncategorized"""

    # Add original SMS as comment
    if sms:
        comment = sms.replace("\n", " ")[:80]
        entry = f"; {comment}\n{entry}"

    return entry


class TransactionExtractor:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Try to load tools config
        config_path = Path(model_path) / "tools_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.tools = config.get("tools", DEFAULT_TOOLS)
                self.system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        else:
            self.tools = DEFAULT_TOOLS
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

        print(f"Model loaded on {self.model.device}")

    def extract(self, sms: str) -> dict:
        """Extract transaction info from an SMS."""

        messages = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": sms},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs.to(self.model.device),
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        input_len = len(inputs["input_ids"][0])
        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        result = parse_tool_call(generated)

        if result is None:
            return {
                "tool": "error",
                "arguments": {"reason": "Failed to parse tool call"},
                "raw_output": generated,
            }

        return result

    def process_batch(self, messages: list[str], output_format: str = "json") -> list:
        """Process multiple SMS messages."""
        results = []

        for i, sms in enumerate(messages):
            result = self.extract(sms)
            result["sms"] = sms

            if output_format == "hledger":
                result["hledger"] = to_hledger_entry(result, sms)

            results.append(result)
            print(f"Processed {i + 1}/{len(messages)}", file=sys.stderr)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract transactions from SMS using fine-tuned model"
    )
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument(
        "--input",
        default=None,
        help="SMS text or path to file with SMS messages (one per line)",
    )
    parser.add_argument(
        "--format", choices=["json", "hledger"], default="json", help="Output format"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    extractor = TransactionExtractor(args.model_path)

    if args.interactive:
        print("\n🏦 Transaction Extractor (interactive mode)")
        print("Enter SMS messages to extract (Ctrl+D to exit)\n")

        while True:
            try:
                sms = input("SMS> ").strip()
                if not sms:
                    continue

                result = extractor.extract(sms)

                print("\n📝 Result:")
                if args.format == "hledger":
                    print(to_hledger_entry(result, sms))
                else:
                    print(json.dumps(result, indent=2))
                print()

            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    elif args.input:
        # Check if input is a file
        if Path(args.input).exists():
            with open(args.input) as f:
                messages = [line.strip() for line in f if line.strip()]
        else:
            messages = [args.input]

        results = extractor.process_batch(messages, args.format)

        if args.format == "hledger":
            for r in results:
                print(r["hledger"])
                print()
        else:
            print(json.dumps(results, indent=2))

    else:
        # Read from stdin
        messages = [line.strip() for line in sys.stdin if line.strip()]

        if not messages:
            print(
                "No input provided. "
                "Use --input, --interactive, or pipe SMS via stdin."
            )
            sys.exit(1)

        results = extractor.process_batch(messages, args.format)

        if args.format == "hledger":
            for r in results:
                print(r["hledger"])
                print()
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

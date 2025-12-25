#!/usr/bin/env python3
"""
Transaction Message Classifier

Processes transaction messages through multiple LLM models using PydanticAI
and outputs structured classification results.

Output format for each message:
- relevant: {source, currency, amount, date, destination, score}
- irrelevant: {reason, score}

Supported providers:
- anthropic: Direct Anthropic API (ANTHROPIC_API_KEY)
- openai: Direct OpenAI API (OPENAI_API_KEY)
- azure: Azure AI Foundry (requires endpoint + api_key in config)
- google-gla: Google Gemini (GOOGLE_API_KEY)
- ollama: Local Ollama instance
"""

import csv
import json
import os
import random
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.cerebras import CerebrasProvider


# --- Output Schema ---


class RelevantTransaction(BaseModel):
    """A relevant financial transaction with extracted details."""

    classification: Literal["relevant"] = "relevant"
    source: str = Field(
        description="Bank/service name and account identifier, e.g., 'ICICI Bank XX811'"
    )
    currency: str = Field(description="Currency code, e.g., 'INR', 'USD'")
    amount: float = Field(description="Numeric amount of the transaction")
    date: str | None = Field(
        description="Transaction date in ISO format YYYY-MM-DD if available"
    )
    destination: str = Field(
        description="Payee/merchant name with reference ID if available, "
        "e.g., 'SWIGGYINSTAMART (UPI:357857902861)'"
    )
    type: Literal["debit", "credit"] = Field(
        description="Transaction type: debit/credit"
    )
    score: int = Field(ge=1, le=100, description="Confidence score 1-100")


class IrrelevantMessage(BaseModel):
    """A message that is not a financial transaction."""

    classification: Literal["irrelevant"] = "irrelevant"
    reason: str = Field(description="Brief explanation why not a transaction")
    score: int = Field(ge=1, le=100, description="Confidence score 1-100")


TransactionClassification = RelevantTransaction | IrrelevantMessage


# --- System Prompt ---

SYSTEM_PROMPT = """You are a financial transaction classifier. Analyze the given \
message and classify it.

If the message describes a FINANCIAL TRANSACTION (money being sent, received, \
debited, credited, paid, etc.), extract:
- source: bank/service name and account identifier (e.g., 'ICICI Bank XX811')
- currency: currency code (e.g., 'INR', 'USD')
- amount: numeric amount
- date: transaction date in ISO format YYYY-MM-DD if available
- destination: payee/merchant name with reference ID if available
- type: debit/credit
- score: your confidence 1-100

If the message is NOT a financial transaction (promotional, informational, OTP, \
application status, baggage info, etc.), provide:
- reason: brief explanation why this is not a transaction
- score: your confidence 1-100

IMPORTANT:
- Only classify as "relevant" if actual money movement is described
- Credit card applications, document signing requests, flight bookings without \
payment details are "irrelevant"
- The score represents your confidence in the classification"""


ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


haiku = AnthropicModel(
    "claude-haiku-4-5",
    provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL),
)
sonnet = AnthropicModel(
    "claude-sonnet-4-5",
    provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL),
)
opus = AnthropicModel(
    "claude-opus-4-5",
    provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL),
)

gpt_oss_120b = CerebrasModel(
    "gpt-oss-120b",
    provider=CerebrasProvider(api_key=os.environ.get("CEREBRAS_API_KEY", "")),
)


MODELS = [
    ("haiku", haiku),
    ("sonnet", sonnet),
    ("opus", opus),
    ("gpt-oss-120b", gpt_oss_120b),
]


def create_agent(model: AnthropicModel) -> Agent[None, TransactionClassification]:
    """Create a PydanticAI agent for transaction classification."""
    return Agent(
        model=model,
        output_type=TransactionClassification,  # type: ignore
        system_prompt=SYSTEM_PROMPT,
    )


async def classify_with_model(
    agent: Agent[None, TransactionClassification],
    message: str,
    model_name: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
) -> dict:
    """Classify a message with retry logic."""
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            result = await agent.run(f"Classify this message:\n\n{message}")
            return result.output.model_dump()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check for rate limiting
            if "rate" in error_str or "429" in error_str:
                print(
                    f"  [{model_name}] Rate limited, waiting {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            elif "500" in error_str or "502" in error_str or "503" in error_str:
                print(
                    f"  [{model_name}] Server error, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            else:
                print(
                    f"  [{model_name}] Error: {e}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

            # Exponential backoff with jitter
            await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
            delay = min(delay * 2, max_delay)

    # All retries exhausted
    return {
        "classification": "error",
        "error": str(last_error),
        "score": 0,
    }


async def process_single_message(
    idx: int,
    row: dict,
    agents: dict[str, Agent[None, TransactionClassification]],
    semaphore: asyncio.Semaphore,
    total: int,
) -> dict:
    """Process a single message through all models in parallel."""
    async with semaphore:
        body = row.get("body", "")
        timestamp = row.get("date", "")

        # Convert timestamp to readable format
        try:
            ts_int = int(timestamp)
            if ts_int > 1e12:  # milliseconds
                ts_int = ts_int // 1000
            readable_time = datetime.fromtimestamp(ts_int).isoformat()
        except (ValueError, TypeError, OSError):
            readable_time = timestamp

        result: dict = {
            "index": idx,
            "source": {
                "body": body,
                "time": readable_time,
                "original_timestamp": timestamp,
            },
            "output": {},
        }

        print(f"[{idx + 1}/{total}] Processing: {body[:50]}...")

        # Call all models in parallel for this message
        async def call_model(
            name: str, agent: Agent[None, TransactionClassification]
        ) -> tuple[str, dict, float]:
            start_time = time.perf_counter()
            classification = await classify_with_model(
                agent=agent,
                message=body,
                model_name=name,
            )
            elapsed_time = time.perf_counter() - start_time
            return name, classification, elapsed_time

        tasks = [call_model(name, agent) for name, agent in agents.items()]
        model_results = await asyncio.gather(*tasks)

        for model_name, classification, elapsed_time in model_results:
            result["output"][model_name] = classification
            print(
                f"  [{idx + 1}] {model_name} -> "
                f"{classification.get('classification', 'error')} ({elapsed_time:.2f}s)"
            )

        return result


async def process_messages(input_file_path: str, concurrency: int = 10) -> None:
    """Process all messages from the CSV and classify with all models."""
    input_file = Path(input_file_path)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / (input_file.stem + ".output.jsonl")

    # Create agents for each configured model
    agents: dict[str, Agent[None, TransactionClassification]] = {}
    for model_name, model in MODELS:
        try:
            agents[model_name] = create_agent(model)
        except Exception as e:
            print(f"Failed to configure {model_name}: {e}")

    if not agents:
        print("No models configured, exiting.")
        return

    # Read existing output to support resumption
    processed_indices: set[int] = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_indices.add(record.get("index"))
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: {len(processed_indices)} messages already processed")

    # Read input CSV
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    to_process = [
        (idx, row) for idx, row in enumerate(rows) if idx not in processed_indices
    ]
    print(
        f"Processing {len(to_process)} of {total} messages (concurrency={concurrency})"
    )

    # Semaphore to limit concurrent messages
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks for all messages
    tasks = [
        process_single_message(idx, row, agents, semaphore, total)
        for idx, row in to_process
    ]

    # Process all messages concurrently (limited by semaphore)
    # Write results as they complete
    with open(output_file, "a") as out_f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print(f"\nDone! Results written to {output_file}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify transaction messages using LLMs"
    )
    parser.add_argument(
        "--input-file",
        default="input/transaction_messages.csv",
        help="Path to input CSV file (default: input/transaction_messages.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first 5 messages without calling LLM",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of messages to process concurrently (default: 10)",
    )

    args = parser.parse_args()

    if args.dry_run:
        input_file = Path(args.input_file)
        with open(input_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                print(f"\n--- Message {i + 1} ---")
                print(f"Body: {row.get('body', '')[:200]}...")
                print(f"Timestamp: {row.get('date', '')}")

    asyncio.run(process_messages(args.input_file, concurrency=args.concurrency))


if __name__ == "__main__":
    main()

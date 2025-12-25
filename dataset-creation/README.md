# Dataset Creation

Create training datasets for model fine-tuning by classifying SMS messages using multiple LLM providers.

## Overview

This component processes raw SMS messages through multiple LLM models to generate labeled training data. It supports concurrent classification via multiple providers (Anthropic, OpenAI, Google, Cerebras, local models) and includes tools for batch processing with Fireworks AI.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_BASE_URL` | For Azure | Azure AI Foundry Anthropic endpoint |
| `ANTHROPIC_API_KEY` | For Azure | Azure AI Foundry API key |
| `CEREBRAS_API_KEY` | For Cerebras | Cerebras API key |
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key |
| `GOOGLE_API_KEY` | For Google | Google AI API key |

## Scripts

### filter_sms.py

Filter raw SMS exports to identify transaction-related messages.

```bash
uv run python filter_sms.py
```

**Input**: JSON SMS export in `source-sms/` (from apps like SMS Backup & Restore)

**Output**: Two CSV files in `input/`:
- `transaction_messages.csv` - Messages likely containing financial transactions
- `non_transaction_messages.csv` - Other messages

**Filtering logic**:
- Must contain transaction keywords (debit, credit, spent, payment, transfer, etc.)
- Must contain at least 2 numbers
- Excludes messages containing "OTP"

### classify_transactions.py

Classify SMS messages using PydanticAI with multiple LLM providers.

```bash
uv run python classify_transactions.py --input-file input/transaction_messages.csv
```

Options:
- `--input-file` - Input CSV file (default: `input/transaction_messages.csv`)
- `--concurrency` - Number of messages to process concurrently (default: 10)
- `--dry-run` - Print first 5 messages without calling LLM

Supports resumption - already processed messages are skipped on re-run.

### select_best_output.py

Select best model output when multiple models classify the same message.

```bash
uv run python select_best_output.py --input output/transaction_messages.output.jsonl
```

Model priority: opus > sonnet > gpt-oss-120b > haiku

## Input Format

CSV file with columns:
- `body` - SMS message text
- `date` - Timestamp (Unix epoch)

## Output Format

JSONL files with structure:
```json
{
  "index": 0,
  "source": {
    "body": "HDFC Bank Acct XX1234 debited for Rs 500.00 on 15-Jan-25; AMAZON PAY credited. UPI:123456789012.",
    "time": "2025-01-15T10:00:00",
    "original_timestamp": "1736930400000"
  },
  "output": {
    "model_name": {
      "classification": "relevant",
      "source": "HDFC Bank XX1234",
      "currency": "INR",
      "amount": 500.0,
      "date": "2025-01-15",
      "destination": "AMAZON PAY (UPI:123456789012)",
      "type": "debit",
      "score": 95
    }
  }
}
```

## Workflow

1. Place SMS CSV in `input/`
2. Run `classify_transactions.py` with desired models
3. Run `select_best_output.py` to pick best per message
4. Use output as training data for fine-tuning

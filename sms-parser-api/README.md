# Deployment

FastAPI server that processes bank SMS messages and imports transactions to Actual Budget.

## Overview

This component provides a REST API that:
1. Receives SMS messages via HTTP POST
2. Parses them using the fine-tuned FunctionGemma model (via LM Studio)
3. Imports extracted transactions to [Actual Budget](https://actualbudget.org/)

## Prerequisites

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running with the fine-tuned FunctionGemma model
- [Actual Budget](https://actualbudget.org/) with [actual-http-api](https://github.com/jhonderson/actual-http-api) wrapper

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
# Edit .env with your settings
```

See `.env.example` for all available configuration options.

### Testing Mode

Set `DISABLE_ACTUAL_BUDGET=true` in your `.env` to disable Actual Budget integration.
The `/process-sms` endpoint will still parse SMS messages but return results without importing.

## Running

```bash
# From project root
uvicorn deployment.main:app --reload

# Or run directly
python -m deployment.main
```

Server starts at `http://localhost:8000`.

## API Endpoints

### POST /process-sms

Process an SMS message and import transaction to Actual Budget.

```bash
curl -X POST http://localhost:8000/process-sms \
  -H "Content-Type: application/json" \
  -d '{"message": "ICICI Bank Acct XX811 debited Rs 450.00 on 15-Jan-24; ZOMATO credited."}'
```

**Request**:
```json
{
  "message": "Your SMS text here",
  "account_id": "optional-override-account-id"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Transaction imported successfully",
  "transaction": {
    "source": "ICICI Bank XX811",
    "currency": "INR",
    "amount": 450.0,
    "date": "2024-01-15",
    "destination": "ZOMATO",
    "type": "debit"
  }
}
```

### GET /accounts

List all accounts in the default budget.

### GET /health

Health check endpoint.

## Testing

Test the API using data from the fine-tuning test set:

```bash
# Make sure DISABLE_ACTUAL_BUDGET=true in .env
# Start the server
uvicorn deployment.main:app --reload

# Run tests (in another terminal)
uv run python -m deployment.test_api

# With options
uv run python -m deployment.test_api --verbose --max-samples 10
```

Options:
- `--api-url` - API base URL (default: http://localhost:8000)
- `--test-file` - Path to test JSONL (default: finetune-functiongemma/data/test.jsonl)
- `--max-samples` - Limit number of test samples
- `--verbose` - Print details for each test

## Directory Structure

```
deployment/
├── main.py              # FastAPI app and routes
├── config.py            # Environment settings
├── models.py            # Pydantic request/response models
├── test_api.py          # API test script
├── llm/
│   ├── prompts.py       # FunctionGemma prompt templates
│   └── parser.py        # LLM response parsing
└── clients/
    └── actual_budget.py # Actual Budget API client
```

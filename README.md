# Bank SMS Parsing

Extract and import bank transactions from SMS messages using a fine-tuned FunctionGemma model.

## Pipeline

```
SMS Messages (JSON export)
         │
         ▼
┌──────────────────┐
│ dataset-creation │  Filter SMS → Classify via LLMs
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│ finetune-functiongemma │  Train FunctionGemma-270M
└────────┬───────────────┘
         │
         ▼
┌────────────────┐
│ sms-parser-api │  FastAPI → llama.cpp → Actual Budget
└────────────────┘
```

## Components

| Folder | Description |
|--------|-------------|
| [dataset-creation](./dataset-creation) | Filter SMS and create training datasets using multiple LLM providers |
| [finetune-functiongemma](./finetune-functiongemma) | Fine-tune FunctionGemma for transaction extraction |
| [sms-parser-api](./sms-parser-api) | FastAPI server to process SMS and import to Actual Budget |

## Docker Compose Options

Three docker-compose configurations are available:

| File | Description | Use Case |
|------|-------------|----------|
| `docker-compose.yml` | Builds API locally | Development, customization |
| `docker-compose.dockerhub.yml` | Pulls pre-built image | Quick setup, production |
| `docker-compose.full.yml` | Complete stack with Actual Budget | Self-hosted everything |

### Quick Start (Pre-built Image)

```bash
# 1. Download the model from HuggingFace
huggingface-cli download kartikaybagla/functiongemma-bank-sms-parser \
    functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --local-dir ./models

# 2. Configure environment
cp .env.example .env
# Edit .env with your Actual Budget API settings

# 3. Start services
docker-compose -f docker-compose.dockerhub.yml up -d
```

API available at `http://localhost:8000`

### Full Stack (Self-Hosted Actual Budget)

Complete setup with Actual Budget, HTTP API, model server, and SMS parser:

```bash
# 1. Download the model
huggingface-cli download kartikaybagla/functiongemma-bank-sms-parser \
    functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --local-dir ./models

# 2. Start Actual Budget first
docker-compose -f docker-compose.full.yml up -d actual-budget

# 3. Go to http://localhost:5006 and set up your budget
#    - Create a password
#    - Create/import a budget
#    - Note down your budget sync ID

# 4. Configure environment
cp .env.full.example .env.full
# Edit .env.full:
#   - ACTUAL_SERVER_PASSWORD: password you just created
#   - ACTUAL_API_KEY: generate with `openssl rand -hex 32`
#   - DEFAULT_BUDGET_SYNC_ID: from Actual Budget settings
#   - DEFAULT_ACCOUNT_ID: account to import transactions to

# 5. Start all services
docker-compose -f docker-compose.full.yml --env-file .env.full up -d
```

Services:
- Actual Budget: `http://localhost:5006`
- Actual HTTP API: `http://localhost:5007` (docs at `/api-docs/`)
- Model Server: `http://localhost:1234`
- SMS Parser API: `http://localhost:8000`

### Local Build (Development)

```bash
# Build and run from source
docker-compose up -d
```

## Requirements (Development)

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- PyTorch (for fine-tuning only)

## Development Setup

```bash
# Install dependencies
uv sync

# Filter SMS messages
cd dataset-creation
uv run python filter_sms.py

# Start API server (requires model running via LM Studio or llama.cpp)
cd sms-parser-api
uv run uvicorn main:app --reload
```

### PyTorch for Fine-tuning

```bash
# NVIDIA (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# AMD (ROCm 6.2):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

See [finetune-functiongemma/README.md](./finetune-functiongemma/README.md) for training details.

## Links

- [Model on HuggingFace](https://huggingface.co/kartikaybagla/functiongemma-bank-sms-parser)
- [Docker Image on Docker Hub](https://hub.docker.com/r/kartikaybagla/bank-sms-api)
- [Actual Budget](https://actualbudget.org/)
- [Actual HTTP API](https://github.com/jhonderson/actual-http-api)

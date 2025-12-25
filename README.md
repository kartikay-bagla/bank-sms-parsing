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

## Quick Start with Docker

The easiest way to run the project is with Docker Compose:

```bash
# 1. Download the model from HuggingFace
huggingface-cli download kartikaybagla/functiongemma-bank-sms-parser \
    functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --local-dir ./models

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your Actual Budget API settings

# 3. Start services
docker-compose up -d
```

The API will be available at `http://localhost:8000`.

### Using the Published Docker Image

Instead of building locally, you can use the published image from Docker Hub:

```bash
docker pull kartikaybagla/bank-sms-api:latest
```

Or edit `docker-compose.yml` to uncomment the `image:` line under the `api` service.

## Requirements (Development)

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- PyTorch (for fine-tuning only - requires manual installation)

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

PyTorch requires platform-specific installation:

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

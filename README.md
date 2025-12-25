# Finance Function Gemma

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
┌────────────┐
│ deployment │  FastAPI → LM Studio → Actual Budget
└────────────┘
```

## Components

| Folder | Description |
|--------|-------------|
| [dataset-creation](./dataset-creation) | Filter SMS and create training datasets using multiple LLM providers |
| [finetune-functiongemma](./finetune-functiongemma) | Fine-tune FunctionGemma for transaction extraction |
| [deployment](./deployment) | FastAPI server to process SMS and import to Actual Budget |

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- PyTorch (for fine-tuning only - requires manual installation)

## Quick Start

```bash
# Install dependencies
uv sync

# Filter SMS messages
cd dataset-creation
uv run python filter_sms.py

# Start deployment server
uvicorn deployment.main:app --reload
```

### PyTorch for Fine-tuning

PyTorch requires platform-specific installation and must be installed manually before fine-tuning:

```bash
# NVIDIA (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# AMD (ROCm 6.2):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

See [finetune-functiongemma/README.md](./finetune-functiongemma/README.md) for details.

# Fine-tune FunctionGemma

Fine-tune Google's FunctionGemma-270M model for SMS transaction extraction using tool calling.

## Overview

This component trains a small language model to:
1. **`extract_transaction`** - Parse banking SMS messages and extract structured transaction data
2. **`skip_message`** - Identify and skip non-transaction messages (spam, OTPs, promotions)

The fine-tuned model can run locally on consumer hardware via LM Studio.

## Prerequisites

- Python 3.11+ with virtual environment
- GPU with 8GB+ VRAM (16GB+ recommended)
- [Hugging Face account](https://huggingface.co/) with FunctionGemma access

### Supported GPUs

| Platform | GPUs | Notes |
|----------|------|-------|
| NVIDIA | RTX 30xx/40xx, A100 | CUDA, flash attention supported |
| AMD | RX 6000/7000, MI100+ | ROCm required, ~20-30% slower |
| CPU | Any | Very slow, not recommended |

## Setup

> **Important**: PyTorch must be installed manually before other dependencies. It requires platform-specific installation (CUDA for NVIDIA, ROCm for AMD) and cannot be managed by uv/pip automatically.

```bash
# 1. Install PyTorch FIRST (choose one based on your GPU)
# NVIDIA (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# AMD (ROCm 6.2):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 2. Install other dependencies
uv sync

# 3. Authenticate with Hugging Face
huggingface-cli login
```

Accept the [FunctionGemma license](https://huggingface.co/google/functiongemma-270m-it) before training.

## Scripts

### prepare_data.py

Convert classified SMS data to FunctionGemma training format.

```bash
python prepare_data.py --input <distilled_output.jsonl> --output_dir ./data
```

Options:
- `--input` - Input JSONL from dataset-creation (required)
- `--output_dir` - Output directory (default: `./data`)
- `--test_size` - Test set fraction (default: 0.1)
- `--min_confidence` - Minimum confidence score filter (default: 90)

### train.py

Fine-tune the model using Hugging Face TRL.

```bash
python train.py --data_dir ./data --output_dir ./checkpoints
```

Key options:
- `--epochs` - Training epochs (default: 5)
- `--batch_size` - Per-device batch size (default: 4)
- `--learning_rate` - Learning rate (default: 5e-5)
- `--use_lora` - Enable LoRA for memory-efficient training
- `--gradient_checkpointing` - Reduce memory usage
- `--push_to_hub` - Upload to Hugging Face Hub

### evaluate.py

Evaluate model performance on test set.

```bash
python evaluate.py --model_path ./checkpoints --test_file ./data/test.jsonl
```

### inference.py

Run inference on individual messages.

```bash
python inference.py --model_path ./checkpoints --input "ICICI Bank Acct XX811 debited Rs 450.00"
```

### merge_lora.py

Merge LoRA adapters into base model for deployment.

```bash
python merge_lora.py --base_model google/functiongemma-270m-it --lora_path ./checkpoints --output_dir ./merged_model
```

## Output

After training:
- `./checkpoints/` - Model weights and tokenizer
- `./checkpoints/training_logs/` - TensorBoard logs
- `./checkpoints/tools_config.json` - Tool definitions for inference

View training progress:
```bash
tensorboard --logdir ./checkpoints/training_logs
```

## Export to GGUF

Convert the fine-tuned model to GGUF format for deployment with llama.cpp server.

### Prerequisites

- Git
- CMake and C++ compiler (`build-essential` on Linux, Xcode on macOS)
- Python 3.11+ (same environment as training)

### Clone and build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build (CPU-only)
cmake -B build
cmake --build build --config Release -j$(nproc)
```

### Install conversion dependencies

```bash
pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

### Convert to GGUF

Convert the merged HuggingFace model to GGUF format:

```bash
# From the llama.cpp directory
python convert_hf_to_gguf.py \
    /path/to/finetune-functiongemma/merged_model \
    --outfile ./functiongemma-270m-bank-sms-parser-f16.gguf \
    --outtype f16
```

> Note: Use `--outtype bf16` if the model uses bfloat16 weights.

### Quantize

Create smaller, faster models with quantization:

```bash
# Q4_K_M - Recommended: ~150MB, good quality
./build/bin/llama-quantize functiongemma-270m-bank-sms-parser-f16.gguf \
    functiongemma-270m-bank-sms-parser-Q4_K_M.gguf Q4_K_M

# Q5_K_M - Higher quality: ~180MB
./build/bin/llama-quantize functiongemma-270m-bank-sms-parser-f16.gguf \
    functiongemma-270m-bank-sms-parser-Q5_K_M.gguf Q5_K_M

# Q8_0 - Near-lossless: ~280MB
./build/bin/llama-quantize functiongemma-270m-bank-sms-parser-f16.gguf \
    functiongemma-270m-bank-sms-parser-Q8_0.gguf Q8_0
```

### Recommended quantization

| Quantization | Size | Use case |
|--------------|------|----------|
| **Q4_K_M** | ~150MB | Production - best size/quality tradeoff |
| Q5_K_M | ~180MB | If Q4_K_M shows quality issues |
| Q8_0 | ~280MB | Testing/validation against full precision |

### Publish to HuggingFace

Upload GGUF files to HuggingFace for distribution:

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Create the repository (first time only)
huggingface-cli repo create kartikaybagla/functiongemma-bank-sms-parser --type model

# Upload GGUF files
huggingface-cli upload kartikaybagla/functiongemma-bank-sms-parser \
    ./functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --repo-type model

# Upload multiple quantizations
huggingface-cli upload kartikaybagla/functiongemma-bank-sms-parser \
    ./functiongemma-270m-bank-sms-parser-Q5_K_M.gguf \
    ./functiongemma-270m-bank-sms-parser-Q8_0.gguf \
    --repo-type model
```

Add a model card (README.md) to the HuggingFace repo describing:
- Model purpose and training data
- Quantization options and recommended usage
- Example inference code

### Deployment

**Option 1: Download from HuggingFace (recommended)**

```bash
# Download the recommended Q4_K_M quantization
huggingface-cli download kartikaybagla/functiongemma-bank-sms-parser \
    functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --local-dir ./models
```

**Option 2: Use locally converted model**

Copy the GGUF file to the `models/` directory in the project root:

```bash
cp functiongemma-270m-bank-sms-parser-Q4_K_M.gguf /path/to/bank-sms-parsing/models/
```

**Start the services:**

```bash
docker-compose up -d
```

## Troubleshooting

**Out of Memory**: Reduce `--batch_size` to 1-2, add `--use_lora` and `--gradient_checkpointing`

**AMD GPU not detected**: Verify ROCm installation with `rocm-smi` and add user to video/render groups

**Poor results**: Increase training data, train for more epochs, ensure balanced class distribution

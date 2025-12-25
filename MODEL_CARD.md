---
license: gemma
base_model: google/functiongemma-270m-it
tags:
  - function-calling
  - finance
  - sms-parsing
  - transaction-extraction
  - gguf
  - llama-cpp
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# FunctionGemma Bank SMS Parser

A fine-tuned [FunctionGemma-270M-IT](https://huggingface.co/google/functiongemma-270m-it) model for extracting structured transaction data from bank SMS messages.

## Model Description

This model is trained to perform two functions:

1. **`extract_transaction`** - Parse banking SMS and extract structured fields:
   - `source`: Bank or sender name
   - `currency`: Currency code (INR, USD, etc.)
   - `amount`: Transaction amount (number)
   - `date`: Transaction date
   - `destination`: Recipient or merchant
   - `type`: "debit" or "credit"

2. **`skip_message`** - Identify non-transaction messages:
   - OTPs and verification codes
   - Promotional messages
   - Payment requests (not completed transactions)
   - Account alerts without transactions

## Quantization Options

| File | Quantization | Size | Description |
|------|--------------|------|-------------|
| `functiongemma-270m-bank-sms-parser-Q4_K_M.gguf` | Q4_K_M | ~242MB | **Recommended** - Best size/quality tradeoff |
| `functiongemma-270m-bank-sms-parser-Q5_K_M.gguf` | Q5_K_M | ~248MB | Higher quality if Q4 shows issues |
| `functiongemma-270m-bank-sms-parser-Q8_0.gguf` | Q8_0 | ~280MB | Near-lossless, for validation |

## Usage

### With llama.cpp server

```bash
# Download model
huggingface-cli download kartikaybagla/functiongemma-bank-sms-parser \
    functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --local-dir ./models

# Run server
llama-server --model ./models/functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 --ctx-size 2048
```

### With Docker

```bash
docker run -p 8080:8080 -v ./models:/models \
    ghcr.io/ggml-org/llama.cpp:server \
    --model /models/functiongemma-270m-bank-sms-parser-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080
```

### API Request

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<bos><start_of_turn>developer\nYou are a financial transaction extractor. Analyze SMS messages and:\n1. If the message describes a completed financial transaction (money sent, received, debited, or credited), use extract_transaction to capture the details.\n2. If the message is not a transaction (OTP, promotional, application status, payment request, etc.), use skip_message.\n\nOnly extract actual completed transactions with concrete amounts, not payment requests or pending transactions.<start_function_declaration>declaration:extract_transaction{description:<escape>Extract transaction details from a banking SMS message<escape>,parameters:{properties:{source:{type:<escape>STRING<escape>},currency:{type:<escape>STRING<escape>},amount:{type:<escape>NUMBER<escape>},date:{type:<escape>STRING<escape>},destination:{type:<escape>STRING<escape>},type:{type:<escape>STRING<escape>}},required:[<escape>source<escape>,<escape>currency<escape>,<escape>amount<escape>,<escape>date<escape>,<escape>destination<escape>,<escape>type<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><start_function_declaration>declaration:skip_message{description:<escape>Skip messages that are not financial transactions<escape>,parameters:{properties:{reason:{type:<escape>STRING<escape>}},required:[<escape>reason<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nICICI Bank Acct XX123 debited Rs 450.00 on 15-Jan-25; UPI to SWIGGY. UPI Ref: 123456789012<end_of_turn>\n<start_of_turn>model\n",
    "max_tokens": 200,
    "stop": ["<end_function_call>"]
  }'
```

### Example Output

**Input SMS:**
```
ICICI Bank Acct XX123 debited Rs 450.00 on 15-Jan-25; UPI to SWIGGY. UPI Ref: 123456789012
```

**Model Output:**
```
<start_function_call>extract_transaction{"source": "ICICI Bank", "currency": "INR", "amount": 450.00, "date": "15-Jan-25", "destination": "SWIGGY", "type": "debit"}<end_function_call>
```

**Input SMS (non-transaction):**
```
Your OTP for login is 482910. Valid for 5 minutes. Do not share.
```

**Model Output:**
```
<start_function_call>skip_message{"reason": "OTP verification code"}<end_function_call>
```

## Training

- **Base Model**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- **Training Framework**: Hugging Face TRL (SFTTrainer)
- **Training Data**: Classified bank SMS messages from Indian banks
- **Fine-tuning Method**: LoRA

## Intended Use

This model is designed for:
- Personal finance automation
- Importing transactions into budgeting apps (e.g., Actual Budget)
- SMS-based expense tracking

## Limitations

- Primarily trained on Indian bank SMS formats (ICICI, HDFC, SBI, etc.)
- May not generalize well to banks from other countries
- Requires the specific prompt format shown above
- Not suitable for security-critical applications without additional validation

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from the base model.

## Links

- [Project Repository](https://github.com/kartikaybagla/bank-sms-parsing)
- [Base Model](https://huggingface.co/google/functiongemma-270m-it)

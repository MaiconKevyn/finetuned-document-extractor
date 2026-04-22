# DocTune — Fine-tuned Document Extractor

A fine-tuning pipeline that teaches a small LLM (Qwen2.5-1.5B) to extract structured fields from noisy payroll documents — and measures how much the fine-tuning actually helps.

---

## Benchmark: Base vs Fine-tuned

Evaluated on **100 held-out samples** with OCR noise. Both models run in 4-bit quantization (NF4) on a single GPU.

### Overall

| Metric | Base Model | Fine-tuned (DocTune) | Delta |
|---|---|---|---|
| Valid JSON Rate | 99.0% | **100.0%** | +1.0pp |
| Avg Field Accuracy | 63.86% | **93.71%** | **+29.85pp** |
| Avg Latency / sample | 3.006s | 3.986s | +32.6% |

### Per-field Accuracy

| Field | Base Model | Fine-tuned | Delta |
|---|---|---|---|
| `employee_name` | 79% | 92% | +13pp |
| `gross_pay` | 56% | 92% | +36pp |
| `tax` | 58% | 94% | +36pp |
| `deductions` | 58% | 91% | +33pp |
| `net_pay` | 60% | **99%** | +39pp |
| `pay_period` | 82% | **99%** | +17pp |
| `invoice_number` | 54% | 89% | +35pp |

The base model already produces valid JSON consistently (99%), but struggles with numeric fields and precise field mapping. Fine-tuning closes that gap by ~30 points across all fields with no architecture changes — just supervised examples.

The latency cost (~1s per call) comes from loading the LoRA adapter weights on top of the base model, which is expected and acceptable for this use case.

---

## How it works

```
generate_dataset.py   →   finetune.py   →   evaluate.py   →   src/api/main.py
   (synthetic data)       (LoRA / SFT)     (benchmark)        (FastAPI serving)
```

---

## Fine-tuning Details

### Method: QLoRA (Quantized Low-Rank Adaptation)

Instead of updating all 1.5B parameters, QLoRA freezes the base model in 4-bit and trains only small low-rank adapter matrices injected into each layer. This makes fine-tuning feasible on consumer hardware (8GB VRAM) with minimal accuracy loss.

### Quantization (BitsAndBytes)

| Parameter | Value | Why |
|---|---|---|
| `load_in_4bit` | `True` | Reduces base model VRAM from ~3GB (FP16) to ~0.9GB |
| `bnb_4bit_quant_type` | `nf4` | NormalFloat4 — better distribution for neural network weights than standard INT4 |
| `bnb_4bit_compute_dtype` | `float16` | Dequantizes to FP16 during forward pass for numerical stability |
| `bnb_4bit_use_double_quant` | `True` | Quantizes the quantization constants themselves, saving an extra ~0.4GB |
| `torch_dtype` (model load) | `float32` | Forced to FP32 due to RTX 2070 lacking BFloat16 support — avoids GradScaler instability during training |

### LoRA Configuration

| Parameter | Value | Why |
|---|---|---|
| `r` (rank) | `16` | Controls adapter size. r=16 adds ~13M trainable params — enough capacity for a structured extraction task without overfitting on 900 samples |
| `lora_alpha` | `32` | Scaling factor (α/r = 2.0). Higher ratio amplifies adapter contribution relative to frozen weights |
| `lora_dropout` | `0.05` | Light regularization to prevent overfitting on the small dataset |
| `bias` | `none` | Bias terms not trained — standard for QLoRA |
| `target_modules` | all projections | Adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) + `gate_proj`, `up_proj`, `down_proj` (MLP). Full coverage yields better task alignment than attention-only |

### Training Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| `per_device_train_batch_size` | `1` | Maximum that fits in 8GB VRAM without FP16 GradScaler |
| `gradient_accumulation_steps` | `8` | Effective batch size = 8. Simulates larger batches without extra VRAM |
| `learning_rate` | `1e-4` | Standard QLoRA LR. Lower than typical SFT (2e-4) for stability without mixed precision |
| `num_train_epochs` | `3` | Sufficient convergence on 900 samples; more epochs risk overfitting |
| `warmup_steps` | `10` | Gradual LR ramp-up to avoid early instability |
| `optimizer` | `paged_adamw_32bit` | Paged optimizer offloads optimizer states to CPU RAM when GPU is under pressure, preventing OOM |
| `fp16 / bf16` | `False / False` | Disabled — RTX 2070 doesn't support BFloat16 and FP16 GradScaler caused gradient underflow during testing |
| `max_length` | `512` | Covers all document templates + JSON output with margin |
| `eval_strategy` | `steps` (every 100) | Monitors val loss during training to detect overfitting early |

### Prompt Format

```
### Instruction:
Extract the following fields from the document text into a JSON format:
employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number.

### Input:
<noisy document text>

### Response:
{"employee_name": "...", "gross_pay": ..., ...}
```

Alpaca-style instruction format. The model learns to associate the `### Response:` token boundary with structured JSON output, which is why the base model (already instruction-tuned) gets to 99% valid JSON even without fine-tuning — it understands the format. Fine-tuning teaches it the specific field semantics and extraction precision.

### Dataset

- **1,000 synthetic payslips** generated with Faker + 5 document templates
- **OCR noise**: ~2% of non-digit characters randomly corrupted, random spurious line breaks inserted 50% of the time
- **Split**: 900 train / 100 validation (10% held-out)
- **Labels**: exact JSON via Pydantic model serialization — no label noise

### Hardware

Trained on a single **RTX 2070 8GB** consumer GPU. Peak VRAM usage ~6.5GB during training.

---

## Quickstart

**Requirements:** Docker, NVIDIA GPU with CUDA 12.4, nvidia-container-toolkit.

```bash
# Clone and enter the repo
git clone <repo-url>
cd finetuned-document-extractor

# Start the API (models are mounted as a volume, not baked into the image)
docker compose up

# Health check
curl http://localhost:8000/health

# Run extraction
curl -s -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Employee: Jane Doe\nInvoice #: 84201\nPeriod: March 2025\nGross: $6200.00\nTax Amount: $1240.00\nDeductions: $300.00\nTotal Net: $4660.00"}'
```

Expected response:
```json
{
  "data": {
    "employee_name": "Jane Doe",
    "gross_pay": 6200.0,
    "tax": 1240.0,
    "deductions": 300.0,
    "net_pay": 4660.0,
    "pay_period": "March 2025",
    "invoice_number": "84201"
  },
  "raw_response": "..."
}
```

---

## Reproduce the pipeline

```bash
# Install local deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic dataset (1000 samples)
python scripts/generate_dataset.py

# 2. Validate data quality before training
python scripts/check_data_quality.py
# Checks: schema, completeness ≥ 95%, exact duplicates, empty inputs

# 3. Fine-tune (requires GPU)
python scripts/finetune.py

# 4. Run benchmark (base vs fine-tuned)
python scripts/evaluate.py
# Results saved to results/artifact_results.json and results/training_run.json

# 5. (Optional) Merge LoRA adapter into base model for PEFT-free serving
python scripts/merge_adapter.py \
  --base models/Qwen2.5-1.5B-Instruct \
  --adapter models/doctune-qwen-1.5b-lora \
  --output models/doctune-qwen-1.5b-merged
```

### Running tests

```bash
pytest tests/ -v
```

Tests cover the full API contract (health, input validation, extraction response, JSON parsing logic) using mocked model/tokenizer — no GPU required.

---

## Project structure

```
.
├── data/                        # Generated datasets (gitignored except .jsonl)
├── models/                      # Base model + LoRA adapter (mounted as volume, not in image)
├── results/
│   ├── artifact_results.json    # Per-field accuracy, base vs fine-tuned
│   └── training_run.json        # Full training run record (hyperparams + benchmark)
├── scripts/
│   ├── generate_dataset.py      # Synthetic payslip generator with OCR noise
│   ├── check_data_quality.py    # Data quality gate (run before fine-tuning)
│   ├── finetune.py              # QLoRA fine-tuning with SFTTrainer
│   ├── evaluate.py              # Benchmark: base model vs fine-tuned
│   ├── merge_adapter.py         # Merge LoRA adapter into base for standalone serving
│   └── test_model_load.py       # Smoke test for model loading outside Docker
├── src/api/
│   └── main.py                  # FastAPI app — lifespan model loading, GPU lock, /extract
├── tests/
│   └── test_api.py              # 17 pytest tests (no GPU required)
├── .env.example                 # Environment variable reference
├── docker-compose.yml           # Local serving with GPU and model volume mount
└── Dockerfile                   # Python 3.11 + CUDA 12.4 + torch 2.5.1 (no models baked in)
```

---

## Stack

| Layer | Technology |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | QLoRA via PEFT + TRL SFTTrainer |
| Quantization | bitsandbytes NF4 4-bit |
| API | FastAPI + Uvicorn |
| Serving | Docker Compose + NVIDIA Container Toolkit |
| Testing | pytest + httpx (no GPU required) |
| Python | 3.11 |
| CUDA | 12.4 |

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

**Dataset generation:** 1,000 synthetic payslips across 5 document templates with simulated OCR noise (character corruption, spurious line breaks). Labels are exact JSON.

**Fine-tuning:** QLoRA (4-bit NF4 base + LoRA adapters, r=16, α=32) via `SFTTrainer`. Targets all attention and MLP projection layers. Trained for 3 epochs on a single RTX 2070 8GB.

**Serving:** FastAPI with an async GPU lock (one inference at a time), 4-bit inference via bitsandbytes, LoRA adapter loaded via PEFT.

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

# 2. Fine-tune (requires GPU)
python scripts/finetune.py

# 3. Run benchmark (base vs fine-tuned)
python scripts/evaluate.py
# Results saved to results/artifact_results.json
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
| Python | 3.11 |
| CUDA | 12.4 |

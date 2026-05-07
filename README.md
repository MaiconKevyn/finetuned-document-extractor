# DocTune — Fine-tuned Document Extractor

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Qwen-2.5--1.5B--Instruct-red.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Fine-tuning](https://img.shields.io/badge/QLoRA-LoRA%20adapters-orange.svg)](https://github.com/huggingface/peft)
[![Serving](https://img.shields.io/badge/FastAPI-inference-009688.svg)](https://fastapi.tiangolo.com/)

DocTune is an AI Engineering portfolio project focused on one concrete research question:

> Can a small open model become a reliable structured payroll-document extractor after targeted QLoRA fine-tuning, and can the result be served with production-minded evaluation, monitoring, and API contracts?

The project trains a LoRA adapter on top of `Qwen/Qwen2.5-1.5B-Instruct` to extract seven fields from noisy payslip-like documents:

```text
employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number
```

## Why This Project Exists

Most document-extraction demos stop at prompting a large commercial model. DocTune is designed to show the full AI Engineering path:

1. Build a controlled synthetic dataset with OCR-like noise.
2. Establish prompt-only baselines.
3. Fine-tune a compact model with QLoRA on consumer hardware.
4. Evaluate with field accuracy, JSON validity, hallucination checks, business-rule checks, latency percentiles, and failure logs.
5. Serve the model behind a FastAPI contract with request IDs, validation flags, batch extraction, and Prometheus-style metrics.

## Current Result Snapshot

The current `data/test.jsonl` report compares fine-tuning against stronger prompt-only baselines. Few-shot examples are selected deterministically from `data/train.jsonl` and balanced across templates.

| System | Valid JSON | Avg Field Accuracy | Business Rule | p95 Latency |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B 0-shot | 100.0% | 75.43% | 74.0% | 3.521s |
| Qwen2.5-1.5B 3-shot | 100.0% | 92.29% | 95.0% | 3.173s |
| Qwen2.5-1.5B 5-shot | 100.0% | 92.71% | 96.0% | 5.606s |
| Qwen2.5-1.5B 10-shot | 100.0% | 94.71% | 95.0% | 6.239s |
| DocTune LoRA | 100.0% | 98.71% | 98.0% | 8.396s |

The stronger 10-shot baseline is valid and useful: it closes much of the gap without training. The adapter still wins on the held-out synthetic test set by about 4 percentage points of average field accuracy and is especially stronger on invoice numbers, pay periods, and payroll arithmetic consistency.

The OOD benchmark has been expanded from an 8-record smoke set to a 200-record categorized `data/golden.jsonl`. The old golden metrics should no longer be treated as current. A limited smoke run is checked in at `results/golden_smoke_eval.md`; the full 200-record golden benchmark should be regenerated as a longer GPU job before making a new model decision.

The repository now uses a `800/100/100` train/validation/test split. Regenerate the full current report with:

```bash
python scripts/eval_suite.py \
  --model-id models/Qwen2.5-1.5B-Instruct \
  --adapter-path models/doctune-qwen-1.5b-lora
```

This writes:

- `results/eval_report.json`
- `results/eval_report.md`

The full report should include 0-shot, 3-shot, 5-shot, 10-shot, few-shot + postprocess, and fine-tuned runs on both `data/test.jsonl` and `data/golden.jsonl`.

## Fine-tuning Method

DocTune uses QLoRA:

| Component | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter method | PEFT LoRA |
| Quantization | NF4 4-bit via BitsAndBytes |
| LoRA rank | 16 in the baseline adapter |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Hardware target | NVIDIA RTX 2070 8GB |

The training script is parameterized for research runs:

```bash
python scripts/finetune.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --lora-r 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --seed 42
```

## Fine-tuning Research Pack

To make the project read as AI Research instead of a single lucky run, DocTune includes an ablation plan:

```bash
python scripts/finetuning_research_pack.py plan
```

The generated plan covers:

| Experiment | Values |
|---|---|
| LoRA rank | `4`, `8`, `16` |
| Learning rate | `5e-5`, `1e-4` |
| Winner seeds | `42`, `123`, `7` |
| Reported metrics | accuracy, JSON validity, business rule, hallucination, p95/p99 latency, adapter size, training time |

Decision rule: pick the smallest adapter within 1 percentage point of the best test accuracy unless latency, hallucination, or business-rule compliance regresses materially.

## Evaluation Design

DocTune separates evaluation into:

| Dataset | Purpose |
|---|---|
| `data/train.jsonl` | SFT training data |
| `data/val.jsonl` | validation / early stopping / hyperparameter selection |
| `data/test.jsonl` | release-quality held-out benchmark |
| `data/golden.jsonl` | 200 categorized OOD edge cases: missing fields, PT-BR/EU formats, heavy OCR noise, unseen templates |
| `data/ablations/*/train.jsonl` | controlled training mixtures for synthetic-only vs synthetic + golden-like data |

Core metrics:

- valid JSON rate
- per-field accuracy
- average field accuracy
- payroll arithmetic check: `gross_pay - tax - deductions ~= net_pay`
- hallucination rate for null ground-truth fields
- template, OOD category, and noise breakdown
- p50/p95/p99 latency
- failure examples persisted through `data/failure_log.jsonl`

## Serving Path

The FastAPI service exposes:

| Endpoint | Purpose |
|---|---|
| `POST /extract` | single-document extraction |
| `POST /extract/batch` | sequential batch extraction with one GPU lock |
| `GET /health` | service and CUDA availability |
| `GET /version` | model, adapter, and prompt version |
| `GET /monitoring/drift` | Evidently drift report over request metadata |
| `GET /metrics` | Prometheus-compatible counters |

Response example:

```json
{
  "request_id": "req-123",
  "data": {
    "employee_name": "Jane Doe",
    "gross_pay": 6200.0,
    "tax": 1240.0,
    "deductions": 300.0,
    "net_pay": 4660.0,
    "pay_period": "March 2025",
    "invoice_number": "84201"
  },
  "raw_response": "...",
  "constrained": false,
  "flags": {
    "extraction_success": true,
    "valid_schema": true,
    "business_rule_valid": true,
    "confidence": "high",
    "failure_reason": null
  }
}
```

## Dataset

The dataset is fully synthetic and contains no real payroll data.

Current split:

| Split | Samples |
|---|---:|
| Train | 800 |
| Validation | 100 |
| Test | 100 |
| Golden | 8 |

Generation features:

- five templates: key-value, abbreviated labels, narrative, table-like, indented summary
- deterministic seed
- OCR-style character corruption
- random line breaks
- metadata fields: `template_id`, `noise_level`

Regenerate:

```bash
python scripts/generate_dataset.py
python scripts/check_data_quality.py
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU inference, place local assets under `models/`:

```text
models/Qwen2.5-1.5B-Instruct/
models/doctune-qwen-1.5b-lora/
```

Run the API:

```bash
docker compose up
```

Smoke test local model loading:

```bash
python scripts/smoke_model_load.py
```

## Testing

Unit tests are GPU-free and restricted to `tests/`:

```bash
./.venv/bin/python -m pytest tests -q
```

Data quality:

```bash
python scripts/check_data_quality.py
```

## Limitations

- The trained adapter is scoped to payroll-like text, not arbitrary documents.
- The baseline dataset is synthetic; the golden set intentionally probes out-of-distribution behavior.
- Current serving serializes GPU access to avoid CUDA OOM on an 8GB RTX 2070.
- Accuracy should be reported from regenerated `results/eval_report.*` before claiming a new model version.
- PT-BR/EU numeric formats are edge cases unless explicitly included in future training data.

## Next Experiments

1. Complete the ablation grid and publish `mean ± std` across three seeds.
2. Add dataset lineage hashes to `MODEL_CARD.md`.
3. Compare against commercial API baselines with `scripts/benchmark_apis.py`.
4. Add DPO/ORPO only if the failure log shows preference-style errors.
5. Promote a model version through MLflow registry and document rollback/canary criteria.

## Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, limitations, ethics, and artifact lineage.

## License

MIT. See [LICENSE](LICENSE).

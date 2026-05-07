# Model Card — DocTune (Qwen2.5-1.5B LoRA)

## Model Details

| | |
|---|---|
| **Base model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Fine-tuning method** | QLoRA (LoRA rank 16, NF4 4-bit quantization) |
| **Adapter type** | PEFT LoRA — does not modify base weights |
| **Task** | Structured field extraction from payroll documents |
| **Training hardware** | NVIDIA RTX 2070 8GB |
| **Training duration** | 3 epochs (~900 steps) |
| **Output format** | JSON |

## Intended Use

Extract seven structured fields from payroll document text:

```
employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number
```

The model is designed for English-language payslips in American numeric format (`$5000.00`, `March 2025`). It handles documents with OCR noise (character corruption, broken lines) across multiple layout styles (key-value, prose, table, indented summary).

**In scope:**
- English payslips and earnings statements with standard US formatting
- Documents with moderate OCR noise (up to ~5% character corruption)
- Multiple visual layouts for the same information

**Out of scope:**
- Non-English documents
- Brazilian/European number formats (`R$ 5.000,00`, `5.000,00`)
- Handwritten documents
- Multi-page documents or multi-employee batches
- Fields beyond the 7 defined above

## Training Data

1,000 synthetic payslips generated with Faker across 5 document templates, with simulated OCR noise applied to non-digit characters. The current repository split is 800/100/100 (train/validation/test), plus a small handcrafted golden set for edge cases. No real personal data was used.

See [The Dataset](README.md#the-dataset) section in the README for full details on generation methodology, templates, and noise design.

## Evaluation

The current benchmark is evaluated on `data/test.jsonl` with 100 held-out samples. Numeric fields are compared with absolute tolerance ±0.5 (accounts for 4-bit quantization drift). Prompt-only few-shot baselines use deterministic, template-balanced examples from `data/train.jsonl`. The OOD golden benchmark now contains 200 categorized records; regenerate current test and golden-set reports with `python scripts/eval_suite.py` before making a new model-selection decision.

### Current test-set comparison

| Metric | Base 0-shot | Base 3-shot | Base 5-shot | Base 10-shot | Fine-tuned |
|---|---:|---:|---:|---:|---:|
| Valid JSON Rate | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| Avg Field Accuracy | 75.43% | 92.29% | 92.71% | 94.71% | **98.71%** |
| Business Rule Compliance | 74.0% | 95.0% | 96.0% | 95.0% | **98.0%** |
| p95 Latency / sample | 3.521s | **3.173s** | 5.606s | 6.239s | 8.396s |

The last full test-set comparison above remains useful for the synthetic holdout. The old golden comparison was superseded by the expanded 200-record OOD set. A limited smoke run lives in `results/golden_smoke_eval.md`; a full golden benchmark should be run as a longer GPU job.

### Per-field accuracy (fine-tuned)

| Field | Accuracy |
|---|---|
| `employee_name` | 95% |
| `gross_pay` | 100% |
| `tax` | 100% |
| `deductions` | 99% |
| `net_pay` | 98% |
| `pay_period` | 100% |
| `invoice_number` | 99% |

## Limitations

**Numeric format sensitivity:** The model was trained on US number format (`5000.00`). Brazilian or European formats with period as thousand separator (`5.000,00`) will cause extraction errors on numeric fields.

**Quantization drift:** 4-bit NF4 quantization can shift numeric outputs by up to ~1.0 in absolute value. The evaluation uses a ±0.5 tolerance to account for this. Applications requiring cent-level precision should post-process or use the `raw_response` field.

**Hallucination risk:** The model may generate plausible but incorrect values for fields that are ambiguous or absent in the input. The API now returns validation flags and a coarse confidence label, but downstream applications should still treat low-confidence fields and missing-source fields with caution.

**Dataset size:** 800 training samples is small. The model generalizes well across the 5 trained templates, but the expanded 200-record OOD golden set is the next gate for judging robustness.

## Ethical Considerations

The model processes payroll data, which is personally identifiable information (PII). Deployments must:

- Not log raw request text containing employee names or salary data
- Not store extraction outputs beyond the immediate processing window
- Comply with applicable data protection regulations (LGPD, GDPR, etc.)
- Not use this model to process documents without authorization from the document owner

The training data is fully synthetic and contains no real personal information.

## How to Use

```python
# Via API (recommended)
import httpx
response = httpx.post("http://localhost:8000/extract", json={
    "text": "Employee: Jane Doe\nGross: $6200.00\nTax: $1240.00\nNet: $4660.00\n..."
})
print(response.json()["data"])

# Enable constrained generation (guarantees valid JSON)
# Set USE_CONSTRAINED_GENERATION=true in docker-compose.yml or .env
```

## Training Hyperparameters

See [Fine-tuning Details](README.md#fine-tuning-details) in the README for the full parameter table with rationale for each choice.

## Artifact Lineage

| Artifact | Location |
|---|---|
| Training script | `scripts/finetune.py` |
| Dataset generator | `scripts/generate_dataset.py` |
| Evaluation scripts | `scripts/evaluate.py`, `scripts/eval_suite.py` |
| Training run record | `results/training_run.json` |
| Benchmark results | `results/artifact_results.json` |
| LoRA adapter | `models/doctune-qwen-1.5b-lora/` |

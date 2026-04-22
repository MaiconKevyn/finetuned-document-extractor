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

1,000 synthetic payslips generated with Faker across 5 document templates, with simulated OCR noise applied to non-digit characters. Dataset was split 900/100 (train/val). No real personal data was used.

See [The Dataset](README.md#the-dataset) section in the README for full details on generation methodology, templates, and noise design.

## Evaluation

Evaluated on 100 held-out samples. Numeric fields compared with absolute tolerance ±0.5 (accounts for 4-bit quantization drift).

### Three-way comparison

| Metric | Base 0-shot | Base 3-shot | Fine-tuned |
|---|---|---|---|
| Valid JSON Rate | 99.0% | 99.0% | **100.0%** |
| Avg Field Accuracy | 63.86% | — | **93.71%** |
| Avg Latency / sample | 3.006s | — | 3.986s |

> Note: 3-shot baseline result to be updated after re-running evaluate.py with the updated metric.

### Per-field accuracy (fine-tuned)

| Field | Accuracy |
|---|---|
| `employee_name` | 92% |
| `gross_pay` | 92% |
| `tax` | 94% |
| `deductions` | 91% |
| `net_pay` | 99% |
| `pay_period` | 99% |
| `invoice_number` | 89% |

## Limitations

**Numeric format sensitivity:** The model was trained on US number format (`5000.00`). Brazilian or European formats with period as thousand separator (`5.000,00`) will cause extraction errors on numeric fields.

**Quantization drift:** 4-bit NF4 quantization can shift numeric outputs by up to ~1.0 in absolute value. The evaluation uses a ±0.5 tolerance to account for this. Applications requiring cent-level precision should post-process or use the `raw_response` field.

**Hallucination risk:** The model may generate plausible but incorrect values for fields that are ambiguous or absent in the input. The API does not currently score confidence — all fields are returned with equal weight. Downstream applications should treat low-confidence fields (e.g., fields not literally present in the input) with caution.

**Dataset size:** 900 training samples is small. The model generalizes well across the 5 trained templates but may degrade on novel layouts not seen during training.

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
| Evaluation script | `scripts/evaluate.py` |
| Training run record | `results/training_run.json` |
| Benchmark results | `results/artifact_results.json` |
| LoRA adapter | `models/doctune-qwen-1.5b-lora/` |

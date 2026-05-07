# DocTune Evaluation Report

- Generated at: `2026-05-07T03:01:38.740384+00:00`
- Prompt version: `v1`
- Base model: `models/Qwen2.5-1.5B-Instruct`
- Adapter: `models/doctune-qwen-1.5b-lora`
- Prompt-only baselines: `3-shot`
- Postprocess baselines: `3-shot`
- Few-shot source: `data/train.jsonl`
- Sample limit: `3`

## Executive Summary

| Dataset | System | Valid JSON | Avg Field Acc | Business Rule | Hallucination | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `data/golden.jsonl` | `baseline_3shot` | 100.0% | 100.0% | 100.0% | 0.0% | 5695.60 | 6154.60 | 6195.40 |
| `data/golden.jsonl` | `baseline_3shot_postprocess` | 100.0% | 100.0% | 100.0% | 0.0% | 5201.68 | 5353.66 | 5367.16 |
| `data/golden.jsonl` | `fine_tuned` | 100.0% | 81.0% | 66.7% | 0.0% | 8114.80 | 8619.38 | 8664.23 |

## Dataset: `data/golden.jsonl`


### baseline_3shot

| Field | Accuracy |
|---|---:|
| `employee_name` | 100.0% |
| `gross_pay` | 100.0% |
| `tax` | 100.0% |
| `deductions` | 100.0% |
| `net_pay` | 100.0% |
| `pay_period` | 100.0% |
| `invoice_number` | 100.0% |

#### Template Breakdown

| Template | Accuracy |
|---|---:|
| `golden_pt_br_currency` | 100.0% |

#### Category Breakdown

| Category | Accuracy |
|---|---:|
| `golden_pt_br_currency` | 100.0% |

#### Noise Breakdown

| Noise bucket | Accuracy |
|---|---:|
| `low` | 100.0% |

### baseline_3shot_postprocess

| Field | Accuracy |
|---|---:|
| `employee_name` | 100.0% |
| `gross_pay` | 100.0% |
| `tax` | 100.0% |
| `deductions` | 100.0% |
| `net_pay` | 100.0% |
| `pay_period` | 100.0% |
| `invoice_number` | 100.0% |

#### Template Breakdown

| Template | Accuracy |
|---|---:|
| `golden_pt_br_currency` | 100.0% |

#### Category Breakdown

| Category | Accuracy |
|---|---:|
| `golden_pt_br_currency` | 100.0% |

#### Noise Breakdown

| Noise bucket | Accuracy |
|---|---:|
| `low` | 100.0% |

### fine_tuned

| Field | Accuracy |
|---|---:|
| `employee_name` | 100.0% |
| `gross_pay` | 66.7% |
| `tax` | 100.0% |
| `deductions` | 100.0% |
| `net_pay` | 100.0% |
| `pay_period` | 0.0% |
| `invoice_number` | 100.0% |

#### Template Breakdown

| Template | Accuracy |
|---|---:|
| `golden_pt_br_currency` | 81.0% |

#### Category Breakdown

| Category | Accuracy |
|---|---:|
| `golden_pt_br_currency` | 81.0% |

#### Noise Breakdown

| Noise bucket | Accuracy |
|---|---:|
| `low` | 81.0% |

#### Failure Examples

| Sample | Reason | Field | Prediction | Ground truth |
|---:|---|---|---|---|
| 0 | field_mismatch | `pay_period` | `January 2023` | `Janeiro 2023` |
| 1 | field_mismatch | `pay_period` | `June 2024` | `Junho 2024` |
| 2 | field_mismatch | `gross_pay` | `4960.63` | `4860.63` |
| 2 | field_mismatch | `pay_period` | `September 2024` | `Setembro 2024` |

## Interpretation Notes

- `business_rule` checks whether `gross_pay - tax - deductions` approximately equals `net_pay`.
- `hallucination` is measured only when ground truth contains null fields.
- Numeric field matching uses the tolerance defined in `scripts/evaluate.py`.
- Generated reports should be committed only when produced from the current model and dataset hashes.

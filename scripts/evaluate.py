import json
import torch
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from src.utils import extract_json_from_text

# Fields whose values are numeric — compared with tolerance, not string equality
NUMERIC_FIELDS = {"gross_pay", "tax", "deductions", "net_pay"}
# Tolerance in absolute value (accounts for 4-bit quantization drift)
NUMERIC_TOLERANCE = 0.5

# Few-shot examples drawn from training distribution (never from val set)
FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "Employee: Maria Costa\nInvoice #: 55123\nPeriod: January 2024\n"
            "Gross: $4500.00\nTax Amount: $900.00\nDeductions: $150.00\nTotal Net: $3450.00"
        ),
        "output": (
            '{"employee_name":"Maria Costa","gross_pay":4500.0,"tax":900.0,'
            '"deductions":150.0,"net_pay":3450.0,"pay_period":"January 2024","invoice_number":"55123"}'
        ),
    },
    {
        "input": (
            "PAYSLIP\nName: James Turner\nID: 77842\nDates: August 2025\n"
            "Earnings: 7200.00\nTaxes: 1440.00\nOther: 320.00\nPayable: 5440.00"
        ),
        "output": (
            '{"employee_name":"James Turner","gross_pay":7200.0,"tax":1440.0,'
            '"deductions":320.0,"net_pay":5440.0,"pay_period":"August 2025","invoice_number":"77842"}'
        ),
    },
    {
        "input": (
            "Earnings Statement for Carol Mendes. Invoice 34901 for period June 2023. "
            "Your gross pay was 3100.00 with taxes of 465.00 and deductions of 200.00. "
            "Resulting net: 2435.00."
        ),
        "output": (
            '{"employee_name":"Carol Mendes","gross_pay":3100.0,"tax":465.0,'
            '"deductions":200.0,"net_pay":2435.0,"pay_period":"June 2023","invoice_number":"34901"}'
        ),
    },
]

INSTRUCTION = (
    "Extract the following fields from the document text into a JSON format: "
    "employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number."
)



def values_match(field: str, pred_val, gt_val) -> bool:
    """Compare predicted vs ground-truth value with numeric tolerance for float fields."""
    if field in NUMERIC_FIELDS:
        try:
            return abs(float(pred_val) - float(gt_val)) <= NUMERIC_TOLERANCE
        except (ValueError, TypeError):
            return False
    return str(pred_val).strip().lower() == str(gt_val).strip().lower()


def build_prompt(instruction: str, input_text: str, n_shots: int = 0) -> str:
    """Build Alpaca-style prompt, optionally with few-shot examples prepended."""
    shots = FEW_SHOT_EXAMPLES[:n_shots]
    prefix = ""
    for ex in shots:
        prefix += (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}\n\n"
        )
    return f"{prefix}### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"


def calculate_metrics(predictions, ground_truths):
    total = len(predictions)
    valid_json_count = 0
    field_scores = {f: 0 for f in ground_truths[0].keys()}
    failures = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        if pred is None:
            failures.append({"sample_idx": i, "reason": "invalid_json", "pred": None, "gt": gt})
            continue

        valid_json_count += 1
        for field in gt.keys():
            pred_val = pred.get(field)
            gt_val = gt.get(field)
            if values_match(field, pred_val, gt_val):
                field_scores[field] += 1
            else:
                failures.append({
                    "sample_idx": i,
                    "reason": "field_mismatch",
                    "field": field,
                    "pred": pred_val,
                    "gt": gt_val,
                })

    return {
        "valid_json_rate": round(valid_json_count / total, 4) if total > 0 else 0,
        "field_accuracy": {k: round(v / total, 4) for k, v in field_scores.items()},
        "avg_field_accuracy": round(
            sum(field_scores.values()) / (total * len(ground_truths[0])), 4
        ) if total > 0 else 0,
        "_failures": failures,
    }


def error_analysis(metrics: dict, label: str) -> None:
    failures = metrics.pop("_failures", [])
    if not failures:
        print(f"\n[{label}] No failures.")
        return

    invalid_json = [f for f in failures if f["reason"] == "invalid_json"]
    field_failures = [f for f in failures if f["reason"] == "field_mismatch"]

    print(f"\n[{label}] Error Analysis")
    print(f"  Invalid JSON responses : {len(invalid_json)}")
    print(f"  Field mismatches       : {len(field_failures)}")

    # Count failures by field
    by_field: dict[str, int] = {}
    for f in field_failures:
        by_field[f["field"]] = by_field.get(f["field"], 0) + 1

    if by_field:
        print("  Failures by field:")
        for field, count in sorted(by_field.items(), key=lambda x: -x[1]):
            print(f"    {field:20s} {count} failures")

    # Sample of 3 worst-field failures
    if field_failures:
        print("  Sample mismatches (up to 3):")
        for ex in field_failures[:3]:
            print(f"    field={ex['field']}  pred={ex['pred']!r}  gt={ex['gt']!r}")


def run_evaluation(
    model_id: str,
    adapter_path: str | None = None,
    test_file: str = "data/val.jsonl",
    n_shots: int = 0,
):
    label = (
        f"Fine-tuned ({n_shots}-shot)" if adapter_path
        else f"Baseline ({n_shots}-shot)"
    )
    print(f"\n--- Evaluating: {label} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # float16 is safe for inference-only (no backward pass → no GradScaler underflow).
    # finetune.py uses float32 because RTX 2070 GradScaler had stability issues with fp16 training.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    with open(test_file) as f:
        samples = [json.loads(line) for line in f]

    predictions, ground_truths = [], []

    start_time = time.time()
    for sample in tqdm(samples):
        prompt = build_prompt(sample["instruction"], sample["input"], n_shots=n_shots)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_text = response.split("### Response:\n")[-1]
        predictions.append(extract_json_from_text(prediction_text))
        ground_truths.append(json.loads(sample["output"]))

    duration = time.time() - start_time
    metrics = calculate_metrics(predictions, ground_truths)
    metrics["avg_latency_sec"] = round(duration / len(samples), 4)

    error_analysis(metrics, label)

    del model
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    model_base = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = "models/doctune-qwen-1.5b-lora"

    baseline_0shot = run_evaluation(model_base, n_shots=0)
    baseline_3shot = run_evaluation(model_base, n_shots=3)
    ft_res = run_evaluation(model_base, adapter_path=adapter, n_shots=0)

    artifact = {
        "project": "DocTune",
        "date": "2026-04-21",
        "note": "Numeric fields compared with abs tolerance 0.5 (accounts for 4-bit drift)",
        "comparison": {
            "baseline_0shot": baseline_0shot,
            "baseline_3shot": baseline_3shot,
            "fine_tuned": ft_res,
        },
    }

    with open("results/artifact_results.json", "w") as f:
        json.dump(artifact, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"Baseline  0-shot  — valid JSON: {baseline_0shot['valid_json_rate']*100:.1f}%  "
          f"avg accuracy: {baseline_0shot['avg_field_accuracy']*100:.1f}%")
    print(f"Baseline  3-shot  — valid JSON: {baseline_3shot['valid_json_rate']*100:.1f}%  "
          f"avg accuracy: {baseline_3shot['avg_field_accuracy']*100:.1f}%")
    print(f"Fine-tuned 0-shot — valid JSON: {ft_res['valid_json_rate']*100:.1f}%  "
          f"avg accuracy: {ft_res['avg_field_accuracy']*100:.1f}%")

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test envs
    def tqdm(iterable):
        return iterable

from src.prompts import EXTRACTION_INSTRUCTION as INSTRUCTION, PROMPT_VERSION, build_alpaca_prompt
from src.utils import extract_json_from_text

NUMERIC_FIELDS = {"gross_pay", "tax", "deductions", "net_pay"}
NUMERIC_TOLERANCE = 0.5
MISSING_PREDICTION_VALUES = {None, "", "null", "None"}

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


def values_match(field: str, pred_val, gt_val) -> bool:
    if gt_val is None:
        return pred_val in MISSING_PREDICTION_VALUES

    if field in NUMERIC_FIELDS:
        try:
            return abs(float(pred_val) - float(gt_val)) <= NUMERIC_TOLERANCE
        except (ValueError, TypeError):
            return False

    return str(pred_val).strip().lower() == str(gt_val).strip().lower()


def build_prompt(instruction: str, input_text: str, n_shots: int = 0) -> str:
    prefix = "".join(
        build_alpaca_prompt(instruction, ex["input"], ex["output"]) + "\n\n"
        for ex in FEW_SHOT_EXAMPLES[:n_shots]
    )
    return prefix + build_alpaca_prompt(instruction, input_text)


def bucket_noise_level(noise_level: float | None) -> str:
    if noise_level is None:
        return "unknown"
    if noise_level <= 0.01:
        return "low"
    if noise_level <= 0.03:
        return "medium"
    return "high"


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def business_rule_holds(record: dict | None) -> bool:
    if not record:
        return False

    try:
        gross = float(record["gross_pay"])
        tax = float(record["tax"])
        deductions = float(record["deductions"])
        net = float(record["net_pay"])
    except (KeyError, TypeError, ValueError):
        return False

    return abs(gross - tax - deductions - net) < 1.0


def compute_latency_stats(latencies_sec: list[float]) -> dict[str, float]:
    latencies_ms = [lat * 1000 for lat in latencies_sec]
    if not latencies_ms:
        return {}

    return {
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "latency_ms_p50": round(compute_percentile(latencies_ms, 50), 2),
        "latency_ms_p95": round(compute_percentile(latencies_ms, 95), 2),
        "latency_ms_p99": round(compute_percentile(latencies_ms, 99), 2),
        "latency_ms_max": round(max(latencies_ms), 2),
    }


def _build_breakdown(metrics: dict[str, dict[str, int]]) -> dict[str, float]:
    return {
        key: round(values["correct"] / values["total"], 4)
        for key, values in metrics.items()
        if values["total"] > 0
    }


def calculate_metrics(
    predictions,
    ground_truths,
    sample_metadata: list[dict] | None = None,
    latencies_sec: list[float] | None = None,
):
    total = len(predictions)
    if total == 0:
        return {
            "valid_json_rate": 0,
            "field_accuracy": {},
            "avg_field_accuracy": 0,
            "_failures": [],
        }

    field_names = list(ground_truths[0].keys())
    valid_json_count = 0
    field_scores = {f: 0 for f in field_names}
    failures = []

    template_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    noise_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    has_template_metadata = bool(sample_metadata and any(m.get("template_id") for m in sample_metadata))
    has_noise_metadata = bool(sample_metadata and any("noise_level" in m for m in sample_metadata))

    hallucination_count = 0
    hallucination_opportunities = 0
    business_rule_passes = 0

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        metadata = sample_metadata[i] if sample_metadata else {}
        sample_correct = 0

        if pred is not None:
            valid_json_count += 1

        if business_rule_holds(pred):
            business_rule_passes += 1

        for field in field_names:
            pred_val = pred.get(field) if pred else None
            gt_val = gt.get(field)

            if gt_val is None:
                hallucination_opportunities += 1
                if pred_val not in MISSING_PREDICTION_VALUES:
                    hallucination_count += 1

            if values_match(field, pred_val, gt_val):
                field_scores[field] += 1
                sample_correct += 1
            else:
                failures.append(
                    {
                        "sample_idx": i,
                        "reason": "field_mismatch" if pred is not None else "invalid_json",
                        "field": field,
                        "pred": pred_val,
                        "gt": gt_val,
                        "template_id": metadata.get("template_id"),
                        "noise_level": metadata.get("noise_level"),
                    }
                )

        if pred is None:
            failures.append(
                {
                    "sample_idx": i,
                    "reason": "invalid_json",
                    "pred": None,
                    "gt": gt,
                    "template_id": metadata.get("template_id"),
                    "noise_level": metadata.get("noise_level"),
                }
            )

        if has_template_metadata:
            template_id = metadata.get("template_id", "unknown")
            template_breakdown[template_id]["correct"] += sample_correct
            template_breakdown[template_id]["total"] += len(field_names)

        if has_noise_metadata:
            noise_bucket = bucket_noise_level(metadata.get("noise_level"))
            noise_breakdown[noise_bucket]["correct"] += sample_correct
            noise_breakdown[noise_bucket]["total"] += len(field_names)

    metrics = {
        "valid_json_rate": round(valid_json_count / total, 4),
        "field_accuracy": {k: round(v / total, 4) for k, v in field_scores.items()},
        "avg_field_accuracy": round(sum(field_scores.values()) / (total * len(field_names)), 4),
        "business_rule_compliance": round(business_rule_passes / total, 4),
        "hallucination_rate": round(hallucination_count / hallucination_opportunities, 4)
        if hallucination_opportunities
        else 0.0,
        "_failures": failures,
    }

    if has_template_metadata:
        metrics["accuracy_by_template"] = _build_breakdown(template_breakdown)
    if has_noise_metadata:
        metrics["accuracy_by_noise_bucket"] = _build_breakdown(noise_breakdown)
    if latencies_sec is not None:
        metrics.update(compute_latency_stats(latencies_sec))

    return metrics


def error_analysis(metrics: dict, label: str) -> None:
    failures = metrics.get("_failures", [])
    if not failures:
        print(f"\n[{label}] No failures.")
        return

    invalid_json = [f for f in failures if f["reason"] == "invalid_json"]
    field_failures = [f for f in failures if f["reason"] == "field_mismatch"]

    print(f"\n[{label}] Error Analysis")
    print(f"  Invalid JSON responses : {len(invalid_json)}")
    print(f"  Field mismatches       : {len(field_failures)}")

    by_field: dict[str, int] = {}
    for failure in field_failures:
        field = failure["field"]
        by_field[field] = by_field.get(field, 0) + 1

    if by_field:
        print("  Failures by field:")
        for field, count in sorted(by_field.items(), key=lambda item: -item[1]):
            print(f"    {field:20s} {count} failures")

    print("  Sample mismatches (up to 3):")
    for example in failures[:3]:
        print(
            f"    field={example.get('field')} pred={example.get('pred')!r} "
            f"gt={example.get('gt')!r}"
        )


def append_failure_log(
    *,
    metrics: dict,
    label: str,
    dataset_path: str,
    log_path: str = "data/failure_log.jsonl",
    prompt_version: str = PROMPT_VERSION,
) -> None:
    failures = metrics.get("_failures", [])
    if not failures:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        for failure in failures:
            record = {
                "timestamp": timestamp,
                "dataset": dataset_path,
                "model": label,
                "prompt_version": prompt_version,
                **failure,
            }
            f.write(json.dumps(record) + "\n")


def run_evaluation(
    model_id: str,
    adapter_path: str | None = None,
    test_file: str = "data/test.jsonl",
    n_shots: int = 0,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    label = f"Fine-tuned ({n_shots}-shot)" if adapter_path else f"Baseline ({n_shots}-shot)"
    print(f"\n--- Evaluating: {label} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    predictions = []
    ground_truths = []
    sample_metadata = []
    latencies_sec = []

    for sample in tqdm(samples):
        prompt = build_prompt(sample["instruction"], sample["input"], n_shots=n_shots)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        started_at = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        latencies_sec.append(time.perf_counter() - started_at)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_text = response.split("### Response:\n")[-1]
        predictions.append(extract_json_from_text(prediction_text))
        ground_truths.append(json.loads(sample["output"]))
        sample_metadata.append(
            {
                "template_id": sample.get("template_id"),
                "noise_level": sample.get("noise_level"),
            }
        )

    metrics = calculate_metrics(
        predictions,
        ground_truths,
        sample_metadata=sample_metadata,
        latencies_sec=latencies_sec,
    )
    error_analysis(metrics, label)

    del model
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    model_base = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = "models/doctune-qwen-1.5b-lora"
    test_file = "data/test.jsonl" if os.path.exists("data/test.jsonl") else "data/val.jsonl"

    baseline_0shot = run_evaluation(model_base, test_file=test_file, n_shots=0)
    baseline_3shot = run_evaluation(model_base, test_file=test_file, n_shots=3)
    ft_res = run_evaluation(model_base, adapter_path=adapter, test_file=test_file, n_shots=0)

    artifact = {
        "project": "DocTune",
        "date": "2026-04-25",
        "prompt_version": PROMPT_VERSION,
        "dataset": test_file,
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
    print(
        f"Baseline  0-shot  — valid JSON: {baseline_0shot['valid_json_rate']*100:.1f}%  "
        f"avg accuracy: {baseline_0shot['avg_field_accuracy']*100:.1f}%"
    )
    print(
        f"Baseline  3-shot  — valid JSON: {baseline_3shot['valid_json_rate']*100:.1f}%  "
        f"avg accuracy: {baseline_3shot['avg_field_accuracy']*100:.1f}%"
    )
    print(
        f"Fine-tuned 0-shot — valid JSON: {ft_res['valid_json_rate']*100:.1f}%  "
        f"avg accuracy: {ft_res['avg_field_accuracy']*100:.1f}%"
    )

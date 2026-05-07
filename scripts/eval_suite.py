import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import PROMPT_VERSION, append_failure_log, run_evaluation


DEFAULT_DATASETS = ("data/test.jsonl", "data/golden.jsonl")
DEFAULT_BASELINE_SHOTS = (0, 3, 5, 10)
DEFAULT_POSTPROCESS_SHOTS = (3, 5, 10)


def _percent(value: float | None) -> str:
    if value is None:
        return "not run"
    return f"{value * 100:.1f}%"


def _metric(metrics: dict | None, key: str, default=0):
    if not metrics:
        return default
    return metrics.get(key, default)


def _failure_examples(metrics: dict | None, limit: int) -> list[dict]:
    if not metrics:
        return []
    return metrics.get("_failures", [])[:limit]


def build_markdown_report(artifact: dict, failure_limit: int = 5) -> str:
    lines = [
        "# DocTune Evaluation Report",
        "",
        f"- Generated at: `{artifact['generated_at']}`",
        f"- Prompt version: `{artifact['prompt_version']}`",
        f"- Base model: `{artifact['model_id']}`",
        f"- Adapter: `{artifact['adapter_path']}`",
    ]

    first_dataset = next(iter(artifact["datasets"].values()), None)
    if first_dataset:
        postprocess_shots = first_dataset.get("postprocess_shots", [])
        lines.extend(
            [
                f"- Prompt-only baselines: `{', '.join(str(s) + '-shot' for s in first_dataset['baseline_shots'])}`",
                f"- Postprocess baselines: `{', '.join(str(s) + '-shot' for s in postprocess_shots)}`",
                f"- Few-shot source: `{first_dataset['few_shot_file']}`",
            ]
        )
        if first_dataset.get("sample_limit") is not None:
            lines.append(f"- Sample limit: `{first_dataset['sample_limit']}`")

    lines.extend(
        [
            "",
            "## Executive Summary",
            "",
            "| Dataset | System | Valid JSON | Avg Field Acc | Business Rule | Hallucination | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for dataset_path, dataset_result in artifact["datasets"].items():
        for system, metrics in dataset_result["comparison"].items():
            lines.append(
                "| "
                f"`{dataset_path}` | `{system}` | "
                f"{_percent(_metric(metrics, 'valid_json_rate', None))} | "
                f"{_percent(_metric(metrics, 'avg_field_accuracy', None))} | "
                f"{_percent(_metric(metrics, 'business_rule_compliance', None))} | "
                f"{_percent(_metric(metrics, 'hallucination_rate', None))} | "
                f"{_metric(metrics, 'latency_ms_p50', 0):.2f} | "
                f"{_metric(metrics, 'latency_ms_p95', 0):.2f} | "
                f"{_metric(metrics, 'latency_ms_p99', 0):.2f} |"
            )

    for dataset_path, dataset_result in artifact["datasets"].items():
        lines.extend(["", f"## Dataset: `{dataset_path}`", ""])

        for system, metrics in dataset_result["comparison"].items():
            lines.extend(["", f"### {system}", ""])

            field_accuracy = metrics.get("field_accuracy", {}) if metrics else {}
            if field_accuracy:
                lines.extend(["| Field | Accuracy |", "|---|---:|"])
                for field, value in field_accuracy.items():
                    lines.append(f"| `{field}` | {_percent(value)} |")

            template_breakdown = metrics.get("accuracy_by_template", {}) if metrics else {}
            if template_breakdown:
                lines.extend(["", "#### Template Breakdown", "", "| Template | Accuracy |", "|---|---:|"])
                for template, value in sorted(template_breakdown.items()):
                    lines.append(f"| `{template}` | {_percent(value)} |")

            category_breakdown = metrics.get("accuracy_by_category", {}) if metrics else {}
            if category_breakdown:
                lines.extend(["", "#### Category Breakdown", "", "| Category | Accuracy |", "|---|---:|"])
                for category, value in sorted(category_breakdown.items()):
                    lines.append(f"| `{category}` | {_percent(value)} |")

            noise_breakdown = metrics.get("accuracy_by_noise_bucket", {}) if metrics else {}
            if noise_breakdown:
                lines.extend(["", "#### Noise Breakdown", "", "| Noise bucket | Accuracy |", "|---|---:|"])
                for bucket, value in sorted(noise_breakdown.items()):
                    lines.append(f"| `{bucket}` | {_percent(value)} |")

            examples = _failure_examples(metrics, failure_limit)
            if examples:
                lines.extend(["", "#### Failure Examples", "", "| Sample | Reason | Field | Prediction | Ground truth |", "|---:|---|---|---|---|"])
                for failure in examples:
                    lines.append(
                        "| "
                        f"{failure.get('sample_idx')} | "
                        f"{failure.get('reason')} | "
                        f"`{failure.get('field')}` | "
                        f"`{failure.get('pred')}` | "
                        f"`{failure.get('gt')}` |"
                    )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `business_rule` checks whether `gross_pay - tax - deductions` approximately equals `net_pay`.",
            "- `hallucination` is measured only when ground truth contains null fields.",
            "- Numeric field matching uses the tolerance defined in `scripts/evaluate.py`.",
            "- Generated reports should be committed only when produced from the current model and dataset hashes.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_suite(
    *,
    model_id: str,
    adapter_path: str,
    datasets: list[str],
    baseline_shots: list[int],
    postprocess_shots: list[int],
    few_shot_file: str,
    failure_limit: int,
    sample_limit: int | None = None,
) -> dict:
    artifact = {
        "project": "DocTune",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "datasets": {},
    }

    for dataset_path in datasets:
        comparison = {}

        for n_shots in baseline_shots:
            metrics = run_evaluation(
                model_id,
                test_file=dataset_path,
                n_shots=n_shots,
                few_shot_file=few_shot_file,
                sample_limit=sample_limit,
            )
            label = f"baseline_{n_shots}shot"
            append_failure_log(metrics=metrics, label=label, dataset_path=dataset_path)
            comparison[label] = metrics

        for n_shots in postprocess_shots:
            metrics = run_evaluation(
                model_id,
                test_file=dataset_path,
                n_shots=n_shots,
                few_shot_file=few_shot_file,
                postprocess=True,
                sample_limit=sample_limit,
            )
            label = f"baseline_{n_shots}shot_postprocess"
            append_failure_log(metrics=metrics, label=label, dataset_path=dataset_path)
            comparison[label] = metrics

        fine_tuned = run_evaluation(
            model_id,
            adapter_path=adapter_path,
            test_file=dataset_path,
            n_shots=0,
            sample_limit=sample_limit,
        )
        append_failure_log(metrics=fine_tuned, label="fine_tuned", dataset_path=dataset_path)
        comparison["fine_tuned"] = fine_tuned

        artifact["datasets"][dataset_path] = {
            "comparison": comparison,
            "baseline_shots": baseline_shots,
            "postprocess_shots": postprocess_shots,
            "few_shot_file": few_shot_file,
            "failure_examples_per_system": failure_limit,
            "sample_limit": sample_limit,
        }

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DocTune evaluation suite on test and golden sets.")
    parser.add_argument("--model-id", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="models/doctune-qwen-1.5b-lora")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--baseline-shots", nargs="+", type=int, default=list(DEFAULT_BASELINE_SHOTS))
    parser.add_argument("--postprocess-shots", nargs="+", type=int, default=list(DEFAULT_POSTPROCESS_SHOTS))
    parser.add_argument("--few-shot-file", default="data/train.jsonl")
    parser.add_argument("--output-json", default="results/eval_report.json")
    parser.add_argument("--output-md", default="results/eval_report.md")
    parser.add_argument("--failure-limit", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per dataset for smoke runs.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    artifact = run_suite(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        datasets=args.datasets,
        baseline_shots=args.baseline_shots,
        postprocess_shots=args.postprocess_shots,
        few_shot_file=args.few_shot_file,
        failure_limit=args.failure_limit,
        sample_limit=args.limit,
    )

    Path(args.output_json).write_text(json.dumps(artifact, indent=2))
    Path(args.output_md).write_text(build_markdown_report(artifact, args.failure_limit))


if __name__ == "__main__":
    main()

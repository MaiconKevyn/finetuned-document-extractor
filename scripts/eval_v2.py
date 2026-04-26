import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import PROMPT_VERSION, append_failure_log, run_evaluation


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _build_markdown_report(dataset_path: str, comparison: dict[str, dict]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Dataset: `{dataset_path}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Summary",
        "",
        "| System | Valid JSON | Avg Field Acc | Business Rule | Hallucination | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for label, metrics in comparison.items():
        lines.append(
            "| "
            f"{label} | "
            f"{_format_percent(metrics['valid_json_rate'])} | "
            f"{_format_percent(metrics['avg_field_accuracy'])} | "
            f"{_format_percent(metrics['business_rule_compliance'])} | "
            f"{_format_percent(metrics['hallucination_rate'])} | "
            f"{metrics.get('latency_ms_p50', 0):.2f} | "
            f"{metrics.get('latency_ms_p95', 0):.2f} | "
            f"{metrics.get('latency_ms_p99', 0):.2f} |"
        )

    for label, metrics in comparison.items():
        if metrics.get("accuracy_by_template"):
            lines.extend(
                [
                    "",
                    f"## Per-template breakdown — {label}",
                    "",
                    "| Template | Accuracy |",
                    "|---|---|",
                ]
            )
            for template_id, accuracy in sorted(metrics["accuracy_by_template"].items()):
                lines.append(f"| {template_id} | {_format_percent(accuracy)} |")

        if metrics.get("accuracy_by_noise_bucket"):
            lines.extend(
                [
                    "",
                    f"## Noise breakdown — {label}",
                    "",
                    "| Noise bucket | Accuracy |",
                    "|---|---|",
                ]
            )
            for bucket, accuracy in sorted(metrics["accuracy_by_noise_bucket"].items()):
                lines.append(f"| {bucket} | {_format_percent(accuracy)} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the enhanced evaluation workflow.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="models/doctune-qwen-1.5b-lora")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--output-json", default="results/eval_report.json")
    parser.add_argument("--output-md", default="results/eval_report.md")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    baseline_0shot = run_evaluation(args.model_id, test_file=args.test_file, n_shots=0)
    append_failure_log(metrics=baseline_0shot, label="baseline_0shot", dataset_path=args.test_file)
    baseline_3shot = run_evaluation(args.model_id, test_file=args.test_file, n_shots=3)
    append_failure_log(metrics=baseline_3shot, label="baseline_3shot", dataset_path=args.test_file)
    fine_tuned = run_evaluation(
        args.model_id,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        n_shots=0,
    )
    append_failure_log(metrics=fine_tuned, label="fine_tuned", dataset_path=args.test_file)

    artifact = {
        "project": "DocTune",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.test_file,
        "prompt_version": PROMPT_VERSION,
        "comparison": {
            "baseline_0shot": baseline_0shot,
            "baseline_3shot": baseline_3shot,
            "fine_tuned": fine_tuned,
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(artifact, f, indent=2)

    with open(args.output_md, "w") as f:
        f.write(_build_markdown_report(args.test_file, artifact["comparison"]))


if __name__ == "__main__":
    main()

import argparse
import json
import statistics
from pathlib import Path


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def adapter_output(config_name: str) -> str:
    return f"models/research/{config_name}"


def build_train_command(run: dict, base_model: str) -> str:
    return (
        "python scripts/finetune.py "
        f"--model-id {base_model} "
        f"--output-dir {adapter_output(run['name'])} "
        f"--lora-r {run['lora_r']} "
        f"--learning-rate {run['learning_rate']} "
        f"--lora-dropout {run['lora_dropout']} "
        f"--epochs {run['epochs']} "
        f"--seed {run['seed']} "
        f"--run-name doctune-{run['name']}"
    )


def build_eval_command(run_name: str, base_model: str) -> str:
    return (
        "python scripts/eval_suite.py "
        f"--model-id {base_model} "
        f"--adapter-path {adapter_output(run_name)} "
        f"--output-json results/research/{run_name}_eval.json "
        f"--output-md results/research/{run_name}_eval.md"
    )


def write_plan(config: dict, output_path: str) -> None:
    lines = [
        "# DocTune Fine-tuning Research Pack",
        "",
        f"Objective: {config['objective']}",
        "",
        "## Datasets",
        "",
        "| Split | Path |",
        "|---|---|",
    ]
    for split, path in config["datasets"].items():
        lines.append(f"| {split} | `{path}` |")

    lines.extend(
        [
            "",
            "## Ablation Grid",
            "",
            "| Run | LoRA r | LR | Dropout | Epochs | Seed | Train command | Eval command |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for run in config["ablation_grid"]:
        lines.append(
            "| "
            f"`{run['name']}` | {run['lora_r']} | {run['learning_rate']} | "
            f"{run['lora_dropout']} | {run['epochs']} | {run['seed']} | "
            f"`{build_train_command(run, config['base_model'])}` | "
            f"`{build_eval_command(run['name'], config['base_model'])}` |"
        )

    seed_sweep = config["winner_seed_sweep"]
    winner = next(run for run in config["ablation_grid"] if run["name"] == seed_sweep["base_config_name"])
    lines.extend(
        [
            "",
            "## Winner Seed Sweep",
            "",
            "| Run | Seed | Train command | Eval command |",
            "|---|---:|---|---|",
        ]
    )
    for seed in seed_sweep["seeds"]:
        run = {**winner, "name": f"{winner['name']}_seed{seed}", "seed": seed}
        lines.append(
            "| "
            f"`{run['name']}` | {seed} | "
            f"`{build_train_command(run, config['base_model'])}` | "
            f"`{build_eval_command(run['name'], config['base_model'])}` |"
        )

    lines.extend(
        [
            "",
            "## Required Metrics",
            "",
            ", ".join(f"`{metric}`" for metric in config["metrics"]),
            "",
            "## Decision Rule",
            "",
            "Pick the smallest adapter whose test-set accuracy is within 1 percentage point of the best run, unless latency or hallucination metrics regress materially.",
        ]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")


def _nested_get(obj: dict, path: list[str], default=None):
    current = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def summarize_runs(results_dir: str) -> dict:
    summary = []
    for path in sorted(Path(results_dir).glob("*_eval.json")):
        artifact = json.loads(path.read_text())
        comparison = _nested_get(artifact, ["datasets", "data/test.jsonl", "comparison", "fine_tuned"], {})
        summary.append(
            {
                "run": path.name.removesuffix("_eval.json"),
                "avg_field_accuracy": comparison.get("avg_field_accuracy"),
                "valid_json_rate": comparison.get("valid_json_rate"),
                "business_rule_compliance": comparison.get("business_rule_compliance"),
                "hallucination_rate": comparison.get("hallucination_rate"),
                "latency_ms_p95": comparison.get("latency_ms_p95"),
            }
        )

    accuracies = [row["avg_field_accuracy"] for row in summary if row["avg_field_accuracy"] is not None]
    return {
        "runs": summary,
        "aggregate": {
            "runs_with_accuracy": len(accuracies),
            "mean_avg_field_accuracy": statistics.mean(accuracies) if accuracies else None,
            "std_avg_field_accuracy": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else None,
        },
    }


def write_summary(summary: dict, output_json: str, output_md: str) -> None:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(summary, indent=2))

    lines = [
        "# Fine-tuning Research Summary",
        "",
        "## Aggregate",
        "",
        f"- Runs with accuracy: `{summary['aggregate']['runs_with_accuracy']}`",
        f"- Mean avg field accuracy: `{summary['aggregate']['mean_avg_field_accuracy']}`",
        f"- Std avg field accuracy: `{summary['aggregate']['std_avg_field_accuracy']}`",
        "",
        "## Runs",
        "",
        "| Run | Avg Field Acc | Valid JSON | Business Rule | Hallucination | p95 latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        lines.append(
            "| "
            f"`{row['run']}` | {row['avg_field_accuracy']} | {row['valid_json_rate']} | "
            f"{row['business_rule_compliance']} | {row['hallucination_rate']} | {row['latency_ms_p95']} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or summarize DocTune fine-tuning ablations.")
    parser.add_argument("--config", default="configs/finetuning_research_pack.json")
    parser.add_argument("--plan-output", default="results/finetuning_research_plan.md")
    parser.add_argument("--results-dir", default="results/research")
    parser.add_argument("--summary-json", default="results/finetuning_research_summary.json")
    parser.add_argument("--summary-md", default="results/finetuning_research_summary.md")
    parser.add_argument("command", choices=["plan", "summarize"])
    args = parser.parse_args()

    if args.command == "plan":
        write_plan(load_config(args.config), args.plan_output)
        return

    write_summary(
        summarize_runs(args.results_dir),
        args.summary_json,
        args.summary_md,
    )


if __name__ == "__main__":
    main()

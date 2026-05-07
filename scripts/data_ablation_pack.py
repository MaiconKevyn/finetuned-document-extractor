import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_golden_set import build_golden_records, write_jsonl


DEFAULT_CONFIG = "configs/data_ablation_pack.json"


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_records(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def build_ablation_datasets(config: dict) -> dict:
    base_train = load_jsonl(config["base_train_file"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    pool = build_golden_records(
        records_per_category=config["golden_like_pool"]["records_per_category"],
        seed=config["golden_like_pool"]["seed"],
    )
    pool_path = output_root / "golden_like_pool.jsonl"
    write_jsonl(str(pool_path), pool)

    manifest = {
        "base_train_file": config["base_train_file"],
        "val_file": config["val_file"],
        "test_file": config["test_file"],
        "golden_eval_file": config["golden_eval_file"],
        "golden_like_pool": str(pool_path),
        "variants": [],
    }

    for variant in config["variants"]:
        name = variant["name"]
        golden_like_count = variant["golden_like_count"]
        records = base_train + pool[:golden_like_count]
        variant_dir = output_root / name
        train_path = variant_dir / "train.jsonl"
        write_records(str(train_path), records)
        manifest["variants"].append(
            {
                "name": name,
                "train_file": str(train_path),
                "train_samples": len(records),
                "synthetic_samples": len(base_train),
                "golden_like_samples": golden_like_count,
                "val_file": config["val_file"],
                "adapter_path": f"{config['adapter_root']}/{name}",
                "eval_json": f"{config['results_root']}/{name}_eval.json",
                "eval_md": f"{config['results_root']}/{name}_eval.md",
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def train_command(config: dict, variant: dict) -> str:
    return (
        "python scripts/finetune.py "
        f"--model-id {config['base_model']} "
        f"--train-file {variant['train_file']} "
        f"--val-file {variant['val_file']} "
        f"--output-dir {variant['adapter_path']} "
        f"--lora-r {config['training']['lora_r']} "
        f"--learning-rate {config['training']['learning_rate']} "
        f"--epochs {config['training']['epochs']} "
        f"--seed {config['training']['seed']} "
        f"--run-name doctune-{variant['name']}"
    )


def eval_command(config: dict, variant: dict) -> str:
    return (
        "python scripts/eval_suite.py "
        f"--model-id {config['base_model']} "
        f"--adapter-path {variant['adapter_path']} "
        f"--datasets {config['test_file']} {config['golden_eval_file']} "
        f"--output-json {variant['eval_json']} "
        f"--output-md {variant['eval_md']}"
    )


def write_plan(config: dict, manifest: dict | None, output_path: str) -> None:
    variants = manifest["variants"] if manifest else [
        {
            "name": variant["name"],
            "train_file": f"{config['output_root']}/{variant['name']}/train.jsonl",
            "train_samples": 800 + variant["golden_like_count"],
            "synthetic_samples": 800,
            "golden_like_samples": variant["golden_like_count"],
            "val_file": config["val_file"],
            "adapter_path": f"{config['adapter_root']}/{variant['name']}",
            "eval_json": f"{config['results_root']}/{variant['name']}_eval.json",
            "eval_md": f"{config['results_root']}/{variant['name']}_eval.md",
        }
        for variant in config["variants"]
    ]

    lines = [
        "# DocTune Data Ablation Pack",
        "",
        "## Objective",
        "",
        config["objective"],
        "",
        "## Leakage Control",
        "",
        f"- OOD evaluation set: `{config['golden_eval_file']}`",
        f"- Golden-like training pool: `{config['output_root']}/golden_like_pool.jsonl`",
        "- The golden-like training pool is generated with a different seed and must not replace the OOD evaluation set.",
        "",
        "## Variants",
        "",
        "| Variant | Synthetic | Golden-like | Train samples | Train command | Eval command |",
        "|---|---:|---:|---:|---|---|",
    ]
    for variant in variants:
        lines.append(
            "| "
            f"`{variant['name']}` | {variant['synthetic_samples']} | "
            f"{variant['golden_like_samples']} | {variant['train_samples']} | "
            f"`{train_command(config, variant)}` | "
            f"`{eval_command(config, variant)}` |"
        )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Promote a new adapter only if golden/OOD accuracy improves materially without losing more than 1 percentage point on `data/test.jsonl`.",
            "- If postprocess few-shot remains better on golden/OOD, collect more representative labels before another main training run.",
            "- Keep the current adapter as rollback until a candidate clears both test and golden gates.",
        ]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")


def summarize(results_root: str, output_json: str, output_md: str) -> dict:
    rows = []
    for path in sorted(Path(results_root).glob("*_eval.json")):
        artifact = json.loads(path.read_text())
        datasets = artifact.get("datasets", {})
        test = datasets.get("data/test.jsonl", {}).get("comparison", {}).get("fine_tuned", {})
        golden = datasets.get("data/golden.jsonl", {}).get("comparison", {}).get("fine_tuned", {})
        rows.append(
            {
                "run": path.name.removesuffix("_eval.json"),
                "test_avg_field_accuracy": test.get("avg_field_accuracy"),
                "test_business_rule": test.get("business_rule_compliance"),
                "golden_avg_field_accuracy": golden.get("avg_field_accuracy"),
                "golden_business_rule": golden.get("business_rule_compliance"),
                "golden_hallucination": golden.get("hallucination_rate"),
            }
        )

    summary = {"runs": rows}
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Data Ablation Summary",
        "",
        "| Run | Test Field Acc | Test Business | Golden Field Acc | Golden Business | Golden Hallucination |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['run']}` | {row['test_avg_field_accuracy']} | {row['test_business_rule']} | "
            f"{row['golden_avg_field_accuracy']} | {row['golden_business_rule']} | {row['golden_hallucination']} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, plan, or summarize DocTune data ablations.")
    parser.add_argument("command", choices=["build", "plan", "summarize"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--plan-output", default="results/data_ablation_plan.md")
    parser.add_argument("--summary-json", default="results/data_ablation_summary.json")
    parser.add_argument("--summary-md", default="results/data_ablation_summary.md")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(config["output_root"]) / "manifest.json"

    if args.command == "build":
        manifest = build_ablation_datasets(config)
        write_plan(config, manifest, args.plan_output)
        return

    if args.command == "plan":
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        write_plan(config, manifest, args.plan_output)
        return

    summarize(config["results_root"], args.summary_json, args.summary_md)


if __name__ == "__main__":
    main()

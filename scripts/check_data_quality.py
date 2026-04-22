"""
Data quality gate — run before fine-tuning to catch dataset issues early.
Exits with code 1 if any critical check fails.
"""
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {"instruction", "input", "output"}
OUTPUT_FIELDS = {"employee_name", "gross_pay", "tax", "deductions", "net_pay", "pay_period", "invoice_number"}
COMPLETENESS_THRESHOLD = 0.95


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_file(path: Path) -> list[str]:
    errors = []
    records = load_jsonl(path)
    total = len(records)

    if total == 0:
        return [f"{path.name}: file is empty"]

    # Schema: required top-level keys
    missing_schema = sum(1 for r in records if not REQUIRED_FIELDS.issubset(r.keys()))
    if missing_schema:
        errors.append(f"{path.name}: {missing_schema}/{total} records missing required keys {REQUIRED_FIELDS}")

    # Output completeness: all label fields present and non-null
    null_outputs = 0
    missing_output_fields = 0
    for r in records:
        if not r.get("output"):
            null_outputs += 1
            continue
        try:
            parsed = json.loads(r["output"])
            missing = OUTPUT_FIELDS - set(parsed.keys())
            if missing:
                missing_output_fields += 1
        except (json.JSONDecodeError, TypeError):
            null_outputs += 1

    completeness = 1 - (null_outputs / total)
    if completeness < COMPLETENESS_THRESHOLD:
        errors.append(
            f"{path.name}: output completeness {completeness:.1%} < {COMPLETENESS_THRESHOLD:.0%} threshold"
            f" ({null_outputs} null/unparseable outputs)"
        )
    if missing_output_fields:
        errors.append(f"{path.name}: {missing_output_fields} records have incomplete output JSON fields")

    # Exact duplicates (input text)
    inputs = [r.get("input", "") for r in records]
    duplicates = total - len(set(inputs))
    if duplicates > 0:
        errors.append(f"{path.name}: {duplicates} exact duplicate inputs found")

    # Empty inputs
    empty_inputs = sum(1 for r in records if not r.get("input", "").strip())
    if empty_inputs:
        errors.append(f"{path.name}: {empty_inputs} records with empty input text")

    print(f"  {path.name}: {total} records | completeness {completeness:.1%} | duplicates {duplicates} | empty inputs {empty_inputs}")
    return errors


def main():
    data_dir = Path("data")
    targets = list(data_dir.glob("*.jsonl"))

    if not targets:
        print("No .jsonl files found in data/")
        sys.exit(1)

    print(f"Checking {len(targets)} dataset file(s)...\n")

    all_errors = []
    for path in sorted(targets):
        all_errors.extend(check_file(path))

    print()
    if all_errors:
        print(f"FAILED — {len(all_errors)} issue(s) found:")
        for err in all_errors:
            print(f"  ✗ {err}")
        sys.exit(1)
    else:
        print("PASSED — all data quality checks OK")


if __name__ == "__main__":
    main()

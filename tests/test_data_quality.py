"""
Unit tests for check_data_quality.py utility functions — no GPU required.

Uses pytest tmp_path to create temporary JSONL files.
Covers: valid dataset, null output, empty input, duplicates,
        missing required fields, completeness below threshold.
"""
import json
import pytest
from pathlib import Path

from scripts.check_data_quality import check_file, load_jsonl, COMPLETENESS_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_OUTPUT = json.dumps({
    "employee_name": "Jane Doe",
    "gross_pay": 5000.0,
    "tax": 750.0,
    "deductions": 200.0,
    "net_pay": 4050.0,
    "pay_period": "March 2025",
    "invoice_number": "84201",
})

def _record(instruction="Extract fields.", input_text="Employee: Jane Doe", output=_VALID_OUTPUT):
    return {"instruction": instruction, "input": input_text, "output": output}

def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path):
        f = _write_jsonl(tmp_path / "data.jsonl", [_record(), _record(input_text="Doc 2")])
        records = load_jsonl(f)
        assert len(records) == 2

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(_record()) + "\n\n" + json.dumps(_record(input_text="Doc 2")))
        records = load_jsonl(f)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# check_file — valid dataset (no errors)
# ---------------------------------------------------------------------------

class TestCheckFileValid:
    def test_valid_dataset_returns_no_errors(self, tmp_path):
        f = _write_jsonl(tmp_path / "train.jsonl", [_record(input_text=f"Doc {i}") for i in range(10)])
        errors = check_file(f)
        assert errors == []

    def test_single_valid_record_passes(self, tmp_path):
        f = _write_jsonl(tmp_path / "val.jsonl", [_record()])
        errors = check_file(f)
        assert errors == []


# ---------------------------------------------------------------------------
# check_file — empty file
# ---------------------------------------------------------------------------

class TestCheckFileEmpty:
    def test_empty_file_returns_error(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        errors = check_file(f)
        assert len(errors) == 1
        assert "empty" in errors[0]


# ---------------------------------------------------------------------------
# check_file — schema violations
# ---------------------------------------------------------------------------

class TestCheckFileSchema:
    def test_missing_instruction_field_reported(self, tmp_path):
        bad = {"input": "Doc text", "output": _VALID_OUTPUT}
        f = _write_jsonl(tmp_path / "bad.jsonl", [bad])
        errors = check_file(f)
        assert any("required keys" in e for e in errors)

    def test_missing_output_field_reported(self, tmp_path):
        bad = {"instruction": "Do this.", "input": "Doc text"}
        f = _write_jsonl(tmp_path / "bad.jsonl", [bad])
        errors = check_file(f)
        assert any("required keys" in e or "completeness" in e for e in errors)

    def test_missing_input_field_reported(self, tmp_path):
        bad = {"instruction": "Do this.", "output": _VALID_OUTPUT}
        f = _write_jsonl(tmp_path / "bad.jsonl", [bad])
        errors = check_file(f)
        assert any("required keys" in e for e in errors)


# ---------------------------------------------------------------------------
# check_file — output completeness
# ---------------------------------------------------------------------------

class TestCheckFileCompleteness:
    def test_null_output_triggers_completeness_failure(self, tmp_path):
        records = [_record(input_text=f"Doc {i}") for i in range(10)]
        records[0] = _record(input_text="Doc bad", output=None)
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert any("completeness" in e for e in errors)

    def test_unparseable_output_json_counts_as_null(self, tmp_path):
        records = [_record(input_text=f"Doc {i}") for i in range(10)]
        records[0] = _record(input_text="Doc bad", output="not valid json")
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert any("completeness" in e for e in errors)

    def test_completeness_below_threshold_fails(self, tmp_path):
        # 5 valid + 5 null → 50% completeness < 95%
        good = [_record(input_text=f"Good {i}") for i in range(5)]
        bad = [_record(input_text=f"Bad {i}", output=None) for i in range(5)]
        f = _write_jsonl(tmp_path / "data.jsonl", good + bad)
        errors = check_file(f)
        assert any("completeness" in e for e in errors)

    def test_output_with_missing_label_field_reported(self, tmp_path):
        incomplete_output = json.dumps({"employee_name": "Jane"})  # missing 6 fields
        records = [_record(input_text=f"Doc {i}", output=incomplete_output) for i in range(10)]
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert any("incomplete output" in e for e in errors)


# ---------------------------------------------------------------------------
# check_file — duplicates
# ---------------------------------------------------------------------------

class TestCheckFileDuplicates:
    def test_exact_duplicate_inputs_reported(self, tmp_path):
        records = [_record(input_text="Same doc text") for _ in range(3)]
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert any("duplicate" in e for e in errors)

    def test_no_false_positive_on_similar_but_different_inputs(self, tmp_path):
        records = [_record(input_text=f"Doc {i}") for i in range(5)]
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert not any("duplicate" in e for e in errors)


# ---------------------------------------------------------------------------
# check_file — empty inputs
# ---------------------------------------------------------------------------

class TestCheckFileEmptyInputs:
    def test_empty_string_input_reported(self, tmp_path):
        records = [_record(input_text=f"Doc {i}") for i in range(4)]
        records.append(_record(input_text=""))
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert any("empty input" in e for e in errors)

    def test_whitespace_only_input_reported(self, tmp_path):
        records = [_record(input_text=f"Doc {i}") for i in range(4)]
        records.append(_record(input_text="   "))
        f = _write_jsonl(tmp_path / "data.jsonl", records)
        errors = check_file(f)
        assert any("empty input" in e for e in errors)

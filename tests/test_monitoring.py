"""
Unit tests for src/monitoring.py — no GPU required.

Covers: log_request (append, never raises), _load_reference_features,
_load_current_features, run_drift_report (no_data, insufficient_data paths).
Full drift report (evidently path) requires ≥30 samples + pandas and is
tested via integration path.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.monitoring import (
    log_request,
    run_drift_report,
    _load_reference_features,
    _load_current_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_OUTPUT = json.dumps({
    "employee_name": "Jane Doe", "gross_pay": 5000.0, "tax": 750.0,
    "deductions": 200.0, "net_pay": 4050.0, "pay_period": "March 2025",
    "invoice_number": "84201",
})

def _write_request_log(path: Path, n: int) -> None:
    records = [
        json.dumps({"ts": "2026-04-23T00:00:00+00:00", "text_length": 100 + i, "field_count": 7, "extraction_success": True})
        for i in range(n)
    ]
    path.write_text("\n".join(records))

def _write_train_jsonl(path: Path, n: int = 10) -> None:
    records = [
        json.dumps({"instruction": "Extract.", "input": f"Doc {i} " * 10, "output": _VALID_OUTPUT})
        for i in range(n)
    ]
    path.write_text("\n".join(records))


# ---------------------------------------------------------------------------
# log_request
# ---------------------------------------------------------------------------

class TestLogRequest:
    def test_creates_log_file(self, tmp_path):
        log_path = tmp_path / "req.jsonl"
        with patch("src.monitoring.REQUEST_LOG", log_path):
            log_request("Employee: Jane", {"employee_name": "Jane"})
        assert log_path.exists()

    def test_appends_on_multiple_calls(self, tmp_path):
        log_path = tmp_path / "req.jsonl"
        with patch("src.monitoring.REQUEST_LOG", log_path):
            log_request("Doc 1", {"k": "v"})
            log_request("Doc 2", None)
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_records_correct_fields(self, tmp_path):
        log_path = tmp_path / "req.jsonl"
        with patch("src.monitoring.REQUEST_LOG", log_path):
            log_request("Hello world", {"a": 1, "b": 2})
        rec = json.loads(log_path.read_text().strip())
        assert rec["text_length"] == 11
        assert rec["field_count"] == 2
        assert rec["extraction_success"] is True

    def test_null_extraction_logged_correctly(self, tmp_path):
        log_path = tmp_path / "req.jsonl"
        with patch("src.monitoring.REQUEST_LOG", log_path):
            log_request("doc text", None)
        rec = json.loads(log_path.read_text().strip())
        assert rec["field_count"] == 0
        assert rec["extraction_success"] is False

    def test_never_raises_on_bad_path(self):
        with patch("src.monitoring.REQUEST_LOG", Path("/nonexistent/path/log.jsonl")):
            log_request("text", None)  # must not raise


# ---------------------------------------------------------------------------
# _load_reference_features
# ---------------------------------------------------------------------------

class TestLoadReferenceFeatures:
    def test_returns_empty_if_file_missing(self, tmp_path):
        with patch("src.monitoring.REFERENCE_DATA", tmp_path / "missing.jsonl"):
            rows = _load_reference_features()
        assert rows == []

    def test_returns_features_for_each_record(self, tmp_path):
        ref = tmp_path / "train.jsonl"
        _write_train_jsonl(ref, n=5)
        with patch("src.monitoring.REFERENCE_DATA", ref):
            rows = _load_reference_features()
        assert len(rows) == 5
        assert all("text_length" in r and "field_count" in r for r in rows)

    def test_text_length_matches_input(self, tmp_path):
        ref = tmp_path / "train.jsonl"
        rec = {"instruction": "X", "input": "A" * 42, "output": _VALID_OUTPUT}
        ref.write_text(json.dumps(rec))
        with patch("src.monitoring.REFERENCE_DATA", ref):
            rows = _load_reference_features()
        assert rows[0]["text_length"] == 42

    def test_field_count_matches_output_json(self, tmp_path):
        ref = tmp_path / "train.jsonl"
        rec = {"instruction": "X", "input": "doc", "output": _VALID_OUTPUT}
        ref.write_text(json.dumps(rec))
        with patch("src.monitoring.REFERENCE_DATA", ref):
            rows = _load_reference_features()
        assert rows[0]["field_count"] == 7

    def test_skips_blank_lines(self, tmp_path):
        ref = tmp_path / "train.jsonl"
        rec = json.dumps({"instruction": "X", "input": "doc", "output": _VALID_OUTPUT})
        ref.write_text(f"{rec}\n\n{rec}")
        with patch("src.monitoring.REFERENCE_DATA", ref):
            rows = _load_reference_features()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# _load_current_features
# ---------------------------------------------------------------------------

class TestLoadCurrentFeatures:
    def test_returns_empty_if_log_missing(self, tmp_path):
        with patch("src.monitoring.REQUEST_LOG", tmp_path / "missing.jsonl"):
            rows = _load_current_features()
        assert rows == []

    def test_returns_all_logged_requests(self, tmp_path):
        log = tmp_path / "req.jsonl"
        _write_request_log(log, n=10)
        with patch("src.monitoring.REQUEST_LOG", log):
            rows = _load_current_features()
        assert len(rows) == 10

    def test_features_have_correct_keys(self, tmp_path):
        log = tmp_path / "req.jsonl"
        _write_request_log(log, n=3)
        with patch("src.monitoring.REQUEST_LOG", log):
            rows = _load_current_features()
        assert all("text_length" in r and "field_count" in r for r in rows)


# ---------------------------------------------------------------------------
# run_drift_report — no_data and insufficient_data paths
# ---------------------------------------------------------------------------

class TestRunDriftReport:
    def test_returns_no_data_when_log_empty(self, tmp_path):
        with (
            patch("src.monitoring.REQUEST_LOG", tmp_path / "missing.jsonl"),
            patch("src.monitoring.REFERENCE_DATA", tmp_path / "missing_ref.jsonl"),
        ):
            result = run_drift_report()
        assert result["status"] == "no_data"

    def test_returns_insufficient_data_when_fewer_than_30_requests(self, tmp_path):
        log = tmp_path / "req.jsonl"
        _write_request_log(log, n=15)
        with (
            patch("src.monitoring.REQUEST_LOG", log),
            patch("src.monitoring.REFERENCE_DATA", tmp_path / "missing_ref.jsonl"),
        ):
            result = run_drift_report()
        assert result["status"] == "insufficient_data"
        assert result["logged_requests"] == 15

    def test_returns_ok_with_enough_requests(self, tmp_path):
        log = tmp_path / "req.jsonl"
        ref = tmp_path / "train.jsonl"
        _write_request_log(log, n=50)
        _write_train_jsonl(ref, n=100)
        with (
            patch("src.monitoring.REQUEST_LOG", log),
            patch("src.monitoring.REFERENCE_DATA", ref),
        ):
            result = run_drift_report()
        assert result["status"] == "ok"
        assert "drift_detected" in result
        assert result["logged_requests"] == 50
        assert result["reference_samples"] == 100


# ---------------------------------------------------------------------------
# /monitoring/drift endpoint
# ---------------------------------------------------------------------------

class TestDriftEndpoint:
    def test_drift_endpoint_returns_200_with_no_data(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

"""
Request logging and input drift monitoring.

Two responsibilities:
  1. log_request() — appends every /extract call to a JSONL log (text length,
     field count, timestamp). Lightweight, always-on.
  2. run_drift_report() — compares the current request log against the reference
     dataset (train.jsonl) using Evidently DataDriftPreset. Returns a dict
     summary; call from GET /monitoring/drift.

Drift signals:
  - text_length   : distribution shift may indicate documents from a new source
  - field_count   : shift in non-null extracted fields may indicate model degradation
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REQUEST_LOG = Path(os.getenv("REQUEST_LOG_PATH", "data/request_log.jsonl"))
REFERENCE_DATA = Path(os.getenv("REFERENCE_DATA_PATH", "data/train.jsonl"))


def log_request(text: str, extracted_fields: Optional[dict]) -> None:
    """Append one request record to the request log. Never raises."""
    try:
        REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "text_length": len(text),
            "field_count": len(extracted_fields) if extracted_fields else 0,
            "extraction_success": extracted_fields is not None,
        }
        with REQUEST_LOG.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # monitoring must never crash the serving path


def _load_reference_features() -> list[dict]:
    """Compute text_length and field_count from the training JSONL."""
    if not REFERENCE_DATA.exists():
        return []
    rows = []
    for line in REFERENCE_DATA.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            fields = json.loads(rec.get("output", "{}"))
            field_count = len(fields)
        except Exception:
            field_count = 0
        rows.append({
            "text_length": len(rec.get("input", "")),
            "field_count": field_count,
        })
    return rows


def _load_current_features() -> list[dict]:
    """Load logged request features."""
    if not REQUEST_LOG.exists():
        return []
    rows = []
    for line in REQUEST_LOG.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append({
            "text_length": rec.get("text_length", 0),
            "field_count": rec.get("field_count", 0),
        })
    return rows


def run_drift_report() -> dict:
    """
    Compare current request distribution against training reference.
    Returns a dict with drift flags and summary stats.
    Requires evidently>=0.7 and at least 30 logged requests for reliable results.
    """
    import pandas as pd
    from evidently import Dataset, DataDefinition, ColumnType
    from evidently.presets import DataDriftPreset
    from evidently import Report

    reference_rows = _load_reference_features()
    current_rows = _load_current_features()

    if not current_rows:
        return {"status": "no_data", "message": "No requests logged yet."}

    if len(current_rows) < 30:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(current_rows)} requests logged. Need ≥ 30 for reliable drift detection.",
            "logged_requests": len(current_rows),
        }

    ref_df = pd.DataFrame(reference_rows)
    cur_df = pd.DataFrame(current_rows)

    definition = DataDefinition(
        numerical_columns=["text_length", "field_count"],
    )

    ref_dataset = Dataset.from_pandas(ref_df, data_definition=definition)
    cur_dataset = Dataset.from_pandas(cur_df, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)
    result_dict = result.dict()

    # Extract top-level drift summary
    drift_detected = False
    column_summaries = {}
    try:
        for metric in result_dict.get("metrics", []):
            if "column_name" in metric.get("result", {}):
                col = metric["result"]["column_name"]
                drifted = metric["result"].get("drift_detected", False)
                drift_detected = drift_detected or drifted
                column_summaries[col] = {
                    "drift_detected": drifted,
                    "drift_score": metric["result"].get("drift_score"),
                    "stattest": metric["result"].get("stattest_name"),
                }
    except Exception:
        pass

    return {
        "status": "ok",
        "drift_detected": drift_detected,
        "logged_requests": len(current_rows),
        "reference_samples": len(reference_rows),
        "columns": column_summaries,
    }

"""
Unit tests for evaluate.py utility functions — no GPU required.

Covers: values_match, build_prompt, calculate_metrics.
extract_json_from_text is tested in test_api.py (now lives in src/utils.py).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from scripts.evaluate import (
    values_match,
    build_prompt,
    calculate_metrics,
    INSTRUCTION,
    FEW_SHOT_EXAMPLES,
    NUMERIC_FIELDS,
)


# ---------------------------------------------------------------------------
# values_match
# ---------------------------------------------------------------------------

class TestValuesMatch:
    def test_numeric_field_exact_match(self):
        assert values_match("gross_pay", 5000.0, 5000.0)

    def test_numeric_field_within_tolerance(self):
        assert values_match("gross_pay", 5000.3, 5000.0)

    def test_numeric_field_at_tolerance_boundary(self):
        assert values_match("tax", 900.5, 900.0)

    def test_numeric_field_exceeds_tolerance(self):
        assert not values_match("tax", 901.0, 900.0)

    def test_numeric_field_string_representation(self):
        # model may return string instead of float
        assert values_match("net_pay", "4050.0", 4050.0)

    def test_numeric_field_invalid_value_returns_false(self):
        assert not values_match("gross_pay", "not_a_number", 5000.0)

    def test_string_field_exact_match(self):
        assert values_match("employee_name", "Jane Doe", "Jane Doe")

    def test_string_field_case_insensitive(self):
        assert values_match("employee_name", "jane doe", "Jane Doe")

    def test_string_field_strips_whitespace(self):
        assert values_match("pay_period", " March 2025 ", "March 2025")

    def test_string_field_mismatch(self):
        assert not values_match("employee_name", "Wrong Name", "Jane Doe")

    def test_invoice_number_as_string(self):
        assert values_match("invoice_number", "84201", "84201")

    def test_all_numeric_fields_covered(self):
        assert NUMERIC_FIELDS == {"gross_pay", "tax", "deductions", "net_pay"}


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_0_shot_has_no_examples(self):
        prompt = build_prompt(INSTRUCTION, "some doc text", n_shots=0)
        # Should have exactly one occurrence of ### Instruction:
        assert prompt.count("### Instruction:") == 1
        assert "### Input:\nsome doc text" in prompt
        assert "### Response:\n" in prompt

    def test_3_shot_prepends_three_examples(self):
        prompt = build_prompt(INSTRUCTION, "some doc text", n_shots=3)
        assert prompt.count("### Instruction:") == 4  # 3 shots + 1 real
        assert prompt.count("### Response:") == 4

    def test_1_shot_uses_first_example(self):
        prompt = build_prompt(INSTRUCTION, "my doc", n_shots=1)
        assert FEW_SHOT_EXAMPLES[0]["input"] in prompt
        assert FEW_SHOT_EXAMPLES[0]["output"] in prompt

    def test_real_input_always_at_end(self):
        prompt = build_prompt(INSTRUCTION, "UNIQUE_MARKER_TEXT", n_shots=2)
        assert prompt.endswith("### Input:\nUNIQUE_MARKER_TEXT\n\n### Response:\n")

    def test_shots_exceeding_available_examples_clips(self):
        # Only 3 examples defined — asking for 10 should not raise
        prompt = build_prompt(INSTRUCTION, "doc", n_shots=10)
        assert prompt.count("### Instruction:") == 4  # 3 available + 1


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    FIELDS = ["employee_name", "gross_pay", "tax", "deductions", "net_pay", "pay_period", "invoice_number"]

    def _make_gt(self, **overrides):
        base = {
            "employee_name": "Jane Doe",
            "gross_pay": 5000.0,
            "tax": 750.0,
            "deductions": 200.0,
            "net_pay": 4050.0,
            "pay_period": "March 2025",
            "invoice_number": "84201",
        }
        base.update(overrides)
        return base

    def test_perfect_predictions_give_1_0_accuracy(self):
        gt = self._make_gt()
        metrics = calculate_metrics([gt.copy()], [gt])
        assert metrics["avg_field_accuracy"] == 1.0
        assert metrics["valid_json_rate"] == 1.0

    def test_all_none_predictions_give_0_accuracy(self):
        gt = self._make_gt()
        metrics = calculate_metrics([None], [gt])
        assert metrics["valid_json_rate"] == 0.0
        assert metrics["avg_field_accuracy"] == 0.0

    def test_partial_match_accuracy(self):
        gt = self._make_gt()
        pred = gt.copy()
        pred["employee_name"] = "Wrong Name"  # 1 field wrong out of 7
        metrics = calculate_metrics([pred], [gt])
        expected = 6 / 7
        assert abs(metrics["avg_field_accuracy"] - expected) < 0.01

    def test_numeric_tolerance_counts_as_correct(self):
        gt = self._make_gt(gross_pay=5000.0)
        pred = self._make_gt(gross_pay=5000.4)  # within ±0.5
        metrics = calculate_metrics([pred], [gt])
        assert metrics["field_accuracy"]["gross_pay"] == 1.0

    def test_numeric_over_tolerance_counts_as_wrong(self):
        gt = self._make_gt(gross_pay=5000.0)
        pred = self._make_gt(gross_pay=5001.0)  # outside ±0.5
        metrics = calculate_metrics([pred], [gt])
        assert metrics["field_accuracy"]["gross_pay"] == 0.0

    def test_failures_list_populated_for_invalid_json(self):
        gt = self._make_gt()
        metrics = calculate_metrics([None], [gt])
        assert "_failures" in metrics
        assert any(f["reason"] == "invalid_json" for f in metrics["_failures"])

    def test_failures_list_populated_for_field_mismatch(self):
        gt = self._make_gt()
        pred = self._make_gt(employee_name="Wrong")
        metrics = calculate_metrics([pred], [gt])
        assert any(f["reason"] == "field_mismatch" for f in metrics["_failures"])

    def test_latency_not_added_by_calculate_metrics(self):
        gt = self._make_gt()
        metrics = calculate_metrics([gt.copy()], [gt])
        assert "avg_latency_sec" not in metrics

    def test_multiple_samples_average_correctly(self):
        gt = self._make_gt()
        # 1 perfect + 1 all-wrong
        wrong = {k: "bad" for k in gt}
        metrics = calculate_metrics([gt.copy(), wrong], [gt, gt])
        assert 0.0 < metrics["avg_field_accuracy"] < 1.0

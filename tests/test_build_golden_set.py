import json
from collections import Counter

from scripts.build_golden_set import build_golden_records


class TestBuildGoldenSet:
    def test_builds_categorized_ood_records(self):
        records = build_golden_records(records_per_category=2, seed=123)
        categories = Counter(record["golden_category"] for record in records)
        assert len(records) == 20
        assert len(categories) == 10
        assert all(count == 2 for count in categories.values())

    def test_records_use_full_extraction_schema(self):
        record = build_golden_records(records_per_category=1, seed=123)[0]
        output = json.loads(record["output"])
        assert set(output) == {
            "employee_name",
            "gross_pay",
            "tax",
            "deductions",
            "net_pay",
            "pay_period",
            "invoice_number",
        }
        assert record["difficulty"] == "ood"
        assert record["template_id"] == record["golden_category"]

import json

from scripts.build_golden_set import build_golden_records


class TestBuildGoldenSet:
    def test_builds_multiple_records(self):
        records = build_golden_records()
        assert len(records) >= 5

    def test_records_include_required_metadata(self):
        record = build_golden_records()[0]
        assert "template_id" in record
        assert "noise_level" in record

    def test_at_least_one_record_contains_null_ground_truth_field(self):
        records = build_golden_records()
        parsed_outputs = [json.loads(record["output"]) for record in records]
        assert any(any(value is None for value in output.values()) for output in parsed_outputs)

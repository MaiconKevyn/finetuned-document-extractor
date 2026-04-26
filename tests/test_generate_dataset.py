import json

from scripts.generate_dataset import (
    PayStub,
    add_noise,
    generate_paystub_text,
    split_dataset_stratified,
)


def _record(template_id: str, idx: int) -> dict:
    return {
        "instruction": "Extract",
        "input": f"doc-{template_id}-{idx}",
        "output": json.dumps({"employee_name": "Jane"}),
        "template_id": template_id,
        "noise_level": 0.02,
    }


class TestGeneratePaystubText:
    def test_returns_known_template_id(self):
        data = PayStub(
            employee_name="Jane Doe",
            gross_pay=5000.0,
            tax=750.0,
            deductions=200.0,
            net_pay=4050.0,
            pay_period="March 2025",
            invoice_number="84201",
        )
        template_id, text = generate_paystub_text(data)
        assert template_id in {"key_value", "abbreviated", "narrative", "table", "indented"}
        assert "Jane Doe" in text


class TestAddNoise:
    def test_returns_noise_level_in_range(self):
        noisy_text, noise_level = add_noise("Employee: Jane Doe\nGross: $5000.00")
        assert noisy_text
        assert 0.0 <= noise_level <= 1.0


class TestSplitDatasetStratified:
    def test_preserves_all_records(self):
        dataset = [_record("key_value", i) for i in range(10)] + [_record("table", i) for i in range(10)]
        train, val, test = split_dataset_stratified(dataset, train_ratio=0.8, val_ratio=0.1)
        assert len(train) + len(val) + len(test) == len(dataset)
        assert (len(train), len(val), len(test)) == (16, 2, 2)

    def test_keeps_each_template_in_all_splits_when_group_is_large_enough(self):
        dataset = []
        for template_id in ("key_value", "table", "narrative"):
            dataset.extend(_record(template_id, i) for i in range(20))

        train, val, test = split_dataset_stratified(dataset, train_ratio=0.8, val_ratio=0.1)

        for split in (train, val, test):
            present_templates = {record["template_id"] for record in split}
            assert {"key_value", "table", "narrative"}.issubset(present_templates)

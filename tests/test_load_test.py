import json

from scripts.load_test import load_inputs


def test_load_inputs_reads_input_field(tmp_path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"input": "doc 1"}),
                json.dumps({"input": "doc 2"}),
            ]
        )
    )
    assert load_inputs(str(dataset)) == ["doc 1", "doc 2"]

"""
Shared pytest fixtures — available to all test modules.
"""
import pytest
from unittest.mock import MagicMock, patch


_FAKE_RESPONSE = (
    "### Response:\n"
    '{"employee_name":"John Silva","gross_pay":5000.0,"tax":750.0,'
    '"deductions":200.0,"net_pay":4050.0,"pay_period":"March 2025",'
    '"invoice_number":"84201"}'
)


def _make_model_mock():
    mock = MagicMock()
    mock.generate.return_value = MagicMock(__getitem__=lambda self, i: [0])
    mock.eval.return_value = mock
    return mock


def _make_tokenizer_mock():
    mock = MagicMock()
    inputs = MagicMock()
    inputs.to.return_value = inputs
    mock.return_value = inputs
    mock.decode.return_value = _FAKE_RESPONSE
    return mock


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import src.api.main as main_module

    with (
        patch("src.api.main.os.path.exists", return_value=True),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_make_tokenizer_mock()),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_make_model_mock()),
        patch("peft.PeftModel.from_pretrained", return_value=_make_model_mock()),
    ):
        with TestClient(main_module.app) as c:
            yield c

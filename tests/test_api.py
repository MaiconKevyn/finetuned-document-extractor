"""
API tests — run without GPU using a mocked model.

The model/tokenizer are patched at module level so no CUDA or model files
are required. Tests focus on HTTP contract, input validation, and JSON
extraction logic, which are the behaviours that don't change with hardware.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
# client fixture is defined in conftest.py and shared across all test modules


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_response_has_gpu_field(self, client):
        data = client.get("/health").json()
        assert "gpu" in data
        assert isinstance(data["gpu"], bool)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_text_rejected(self, client):
        response = client.post("/extract", json={"text": ""})
        assert response.status_code == 422

    def test_whitespace_only_rejected(self, client):
        response = client.post("/extract", json={"text": "   "})
        assert response.status_code == 422

    def test_text_exceeding_50k_chars_rejected(self, client):
        response = client.post("/extract", json={"text": "a" * 50_001})
        assert response.status_code == 422

    def test_text_at_limit_accepted(self, client):
        response = client.post("/extract", json={"text": "a" * 50_000})
        assert response.status_code == 200

    def test_missing_body_rejected(self, client):
        response = client.post("/extract")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Extraction response contract
# ---------------------------------------------------------------------------

class TestExtractEndpoint:
    VALID_PAYLOAD = {
        "text": (
            "Employee: John Silva\nInvoice #: 84201\nPeriod: March 2025\n"
            "Gross: $5000.00\nTax Amount: $750.00\nDeductions: $200.00\nTotal Net: $4050.00"
        )
    }

    def test_returns_200(self, client):
        response = client.post("/extract", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_data_field(self, client):
        data = client.post("/extract", json=self.VALID_PAYLOAD).json()
        assert "data" in data

    def test_response_has_raw_response_field(self, client):
        data = client.post("/extract", json=self.VALID_PAYLOAD).json()
        assert "raw_response" in data
        assert isinstance(data["raw_response"], str)

    def test_data_contains_expected_fields(self, client):
        data = client.post("/extract", json=self.VALID_PAYLOAD).json()
        expected_fields = {
            "employee_name", "gross_pay", "tax",
            "deductions", "net_pay", "pay_period", "invoice_number",
        }
        assert expected_fields.issubset(set(data["data"].keys()))

    def test_response_includes_request_id(self, client):
        data = client.post("/extract", json=self.VALID_PAYLOAD).json()
        assert "request_id" in data
        assert isinstance(data["request_id"], str)

    def test_response_includes_quality_flags(self, client):
        data = client.post("/extract", json=self.VALID_PAYLOAD).json()
        assert data["flags"]["extraction_success"] is True
        assert data["flags"]["valid_schema"] is True
        assert data["flags"]["business_rule_valid"] is True
        assert data["flags"]["confidence"] == "high"

    def test_request_id_header_is_propagated(self, client):
        response = client.post(
            "/extract",
            headers={"x-request-id": "req-123"},
            json=self.VALID_PAYLOAD,
        )
        assert response.headers["x-request-id"] == "req-123"
        assert response.json()["request_id"] == "req-123"


# ---------------------------------------------------------------------------
# Model failure behaviour
# ---------------------------------------------------------------------------

class TestModelFailureBehaviour:
    """What happens when the model does not produce valid JSON."""

    def test_invalid_json_from_model_returns_200_with_null_data(self, client):
        """data must be null, not a 500, when the model returns garbage."""
        import src.api.main as main_module
        original_decode = main_module.tokenizer.decode

        main_module.tokenizer.decode = lambda *a, **kw: "### Response:\nsorry I cannot help with that"
        try:
            response = client.post("/extract", json={"text": "some document"})
            assert response.status_code == 200
            assert response.json()["data"] is None
            assert isinstance(response.json()["raw_response"], str)
        finally:
            main_module.tokenizer.decode = original_decode

    def test_partial_json_from_model_returns_null_data(self, client):
        """Truncated or malformed JSON must not crash the endpoint."""
        import src.api.main as main_module
        original_decode = main_module.tokenizer.decode

        main_module.tokenizer.decode = lambda *a, **kw: '### Response:\n{"employee_name": "truncated'
        try:
            response = client.post("/extract", json={"text": "some document"})
            assert response.status_code == 200
            assert response.json()["data"] is None
        finally:
            main_module.tokenizer.decode = original_decode

    def test_heavily_noisy_input_does_not_crash(self, client):
        """All-symbol input should be accepted and return a response."""
        noisy = "!@#$%^&*()_+ " * 100
        response = client.post("/extract", json={"text": noisy})
        assert response.status_code == 200
        assert "data" in response.json()

    def test_response_includes_constrained_flag(self, client):
        """Response schema must always include the constrained field."""
        response = client.post("/extract", json={"text": "Employee: Test"})
        assert "constrained" in response.json()
        assert isinstance(response.json()["constrained"], bool)

    def test_invalid_json_response_includes_failure_flags(self, client):
        import src.api.main as main_module
        original_decode = main_module.tokenizer.decode

        main_module.tokenizer.decode = lambda *a, **kw: "### Response:\nno json"
        try:
            response = client.post("/extract", json={"text": "some document"})
            flags = response.json()["flags"]
            assert flags["extraction_success"] is False
            assert flags["valid_schema"] is False
            assert flags["confidence"] == "low"
            assert flags["failure_reason"] == "invalid_json"
        finally:
            main_module.tokenizer.decode = original_decode


class TestBatchAndMetrics:
    def test_batch_endpoint_returns_one_result_per_text(self, client):
        response = client.post("/extract/batch", json={"texts": ["Employee: A", "Employee: B"]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert all("request_id" in item for item in data["results"])

    def test_metrics_endpoint_is_prometheus_text(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "doctune_extract_requests_total" in response.text
        assert response.headers["content-type"].startswith("text/plain")


# ---------------------------------------------------------------------------
# Startup failure
# ---------------------------------------------------------------------------

class TestStartupFailure:
    def test_missing_adapter_path_raises_on_load(self):
        """load_model() must raise RuntimeError when adapter path does not exist."""
        import src.api.main as main_module
        original_model = main_module.model

        main_module.model = None
        with patch("src.api.main.os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="Adapter path not found"):
                main_module.load_model()

        main_module.model = original_model


# ---------------------------------------------------------------------------
# JSON extraction logic (unit — no HTTP)
# ---------------------------------------------------------------------------

class TestExtractJsonFromText:
    def setup_method(self):
        from src.utils import extract_json_from_text
        self.fn = extract_json_from_text

    def test_valid_json_extracted(self):
        text = '{"employee_name": "Jane", "gross_pay": 3000.0}'
        result = self.fn(text)
        assert result == {"employee_name": "Jane", "gross_pay": 3000.0}

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"key": "value"} end.'
        result = self.fn(text)
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        result = self.fn("this is not json at all")
        assert result is None

    def test_malformed_json_returns_none(self):
        result = self.fn('{"key": "unclosed}')
        assert result is None

    def test_empty_string_returns_none(self):
        result = self.fn("")
        assert result is None

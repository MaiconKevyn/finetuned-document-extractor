import json

from scripts.benchmark_apis import (
    BenchmarkAPIError,
    OPENAI_DEFAULT_MODEL,
    build_messages,
    cache_key,
    build_markdown_report,
    estimate_cost_usd,
    parse_limit_type,
    parse_openai_error,
    parse_retry_after_seconds,
)


class TestBenchmarkHelpers:
    def test_build_messages_contains_instruction_and_document(self):
        messages = build_messages("Employee: Jane Doe")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Employee: Jane Doe" in messages[1]["content"]

    def test_estimate_cost_uses_known_pricing(self):
        cost = estimate_cost_usd(
            OPENAI_DEFAULT_MODEL,
            {"prompt_tokens": 1000, "completion_tokens": 500},
        )
        assert cost > 0
        assert round(cost, 6) == 0.00045

    def test_cache_key_changes_with_text(self):
        key_a = cache_key(model="gpt-4o-mini", prompt_version="v1", text="doc a")
        key_b = cache_key(model="gpt-4o-mini", prompt_version="v1", text="doc b")
        assert key_a != key_b

    def test_markdown_report_contains_core_metrics(self):
        report = build_markdown_report(
            "data/test.jsonl",
            "gpt-4o-mini",
            {
                "samples_evaluated": 2,
                "valid_json_rate": 1.0,
                "avg_field_accuracy": 0.9,
                "business_rule_compliance": 1.0,
                "hallucination_rate": 0.0,
                "latency_ms_p50": 123.0,
                "latency_ms_p95": 456.0,
                "cost_per_1k_requests_usd": 0.3,
                "accuracy_by_template": {"key_value": 0.95},
                "accuracy_by_noise_bucket": {"low": 0.95},
            },
        )
        assert "gpt-4o-mini" in report
        assert "Avg Field Accuracy" in report
        assert "key_value" in report
        assert "low" in report

    def test_parse_retry_after_seconds_extracts_float(self):
        message = "Please try again in 8.64s. Visit rate limits."
        assert parse_retry_after_seconds(message) == 8.64

    def test_parse_limit_type_detects_rpd(self):
        message = "Rate limit reached on requests per day (RPD): Limit 10000."
        assert parse_limit_type(message) == "RPD"

    def test_parse_openai_error_extracts_rate_limit_details(self):
        body = json.dumps(
            {
                "error": {
                    "message": (
                        "Rate limit reached for gpt-4o-mini on requests per day (RPD): "
                        "Limit 10000, Used 10000, Requested 1. Please try again in 8.64s."
                    ),
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            }
        )
        error = parse_openai_error(429, body)
        assert isinstance(error, BenchmarkAPIError)
        assert error.status_code == 429
        assert error.error_type == "rate_limit_exceeded"
        assert error.limit_type == "RPD"
        assert error.retry_after_seconds == 8.64

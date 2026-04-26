import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import PROMPT_VERSION, calculate_metrics
from src.prompts import EXTRACTION_INSTRUCTION

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_CAP_SECONDS = 15.0
EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "employee_name": {"type": ["string", "null"]},
        "gross_pay": {"type": ["number", "null"]},
        "tax": {"type": ["number", "null"]},
        "deductions": {"type": ["number", "null"]},
        "net_pay": {"type": ["number", "null"]},
        "pay_period": {"type": ["string", "null"]},
        "invoice_number": {"type": ["string", "null"]},
    },
    "required": [
        "employee_name",
        "gross_pay",
        "tax",
        "deductions",
        "net_pay",
        "pay_period",
        "invoice_number",
    ],
}


class BenchmarkAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        limit_type: str | None = None,
        retry_after_seconds: float | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.limit_type = limit_type
        self.retry_after_seconds = retry_after_seconds
        self.response_body = response_body


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(f"[openai-benchmark] {json.dumps(payload, sort_keys=True)}")


def build_messages(document_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You extract structured payroll fields from document text. "
                "Return only the requested JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{EXTRACTION_INSTRUCTION}\n\n"
                f"Document:\n{document_text}"
            ),
        },
    ]


def estimate_cost_usd(model: str, usage: dict) -> float:
    pricing = OPENAI_PRICING_PER_1M.get(model, OPENAI_PRICING_PER_1M[OPENAI_DEFAULT_MODEL])
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return round(
        (prompt_tokens / 1_000_000) * pricing["input"]
        + (completion_tokens / 1_000_000) * pricing["output"],
        8,
    )


def cache_key(*, model: str, prompt_version: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(prompt_version.encode())
    digest.update(text.encode())
    return digest.hexdigest()


def load_cache(path: str) -> dict:
    cache_file = Path(path)
    if not cache_file.exists():
        return {}
    return json.loads(cache_file.read_text())


def save_cache(path: str, cache: dict) -> None:
    cache_file = Path(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2))


def parse_retry_after_seconds(message: str) -> float | None:
    match = re.search(r"Please try again in ([0-9]+(?:\.[0-9]+)?)s", message)
    if not match:
        return None
    return float(match.group(1))


def parse_limit_type(message: str) -> str | None:
    upper = message.upper()
    for limit_type in ("RPD", "RPM", "TPM"):
        if limit_type in upper:
            return limit_type
    match = re.search(r"\(([^)]+)\)", message)
    if match:
        return match.group(1).upper()
    return None


def parse_openai_error(status_code: int, body: str) -> BenchmarkAPIError:
    error_type = None
    limit_type = None
    retry_after_seconds = None
    message = body

    try:
        parsed = json.loads(body)
        error = parsed.get("error", {})
        message = error.get("message", body)
        error_type = error.get("code") or error.get("type")
    except json.JSONDecodeError:
        pass

    retry_after_seconds = parse_retry_after_seconds(message)
    limit_type = parse_limit_type(message)
    return BenchmarkAPIError(
        f"OpenAI API request failed with status {status_code}: {message}",
        status_code=status_code,
        error_type=error_type,
        limit_type=limit_type,
        retry_after_seconds=retry_after_seconds,
        response_body=body,
    )


def call_openai_gpt_4o_mini(
    *,
    text: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 60,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_cap_seconds: float = DEFAULT_RETRY_CAP_SECONDS,
    sample_label: str | None = None,
) -> tuple[dict | None, str, dict, float]:
    payload = {
        "model": model,
        "messages": build_messages(text),
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "paystub_extraction",
                "strict": True,
                "schema": EXTRACTION_SCHEMA,
            },
        },
    }

    request = Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    attempt = 0
    while True:
        attempt += 1
        started_at = time.perf_counter()
        try:
            log_event(
                "request_start",
                provider="openai",
                model=model,
                sample=sample_label,
                attempt=attempt,
            )
            with urlopen(request, timeout=timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            parsed_error = parse_openai_error(exc.code, body)
            log_event(
                "request_error",
                provider="openai",
                model=model,
                sample=sample_label,
                attempt=attempt,
                status_code=parsed_error.status_code,
                error_type=parsed_error.error_type,
                limit_type=parsed_error.limit_type,
                retry_after_seconds=parsed_error.retry_after_seconds,
            )
            can_retry = (
                parsed_error.status_code == 429
                and parsed_error.retry_after_seconds is not None
                and parsed_error.retry_after_seconds <= retry_cap_seconds
                and attempt <= max_retries
            )
            if not can_retry:
                raise parsed_error from exc
            time.sleep(parsed_error.retry_after_seconds)
        except URLError as exc:
            log_event(
                "request_error",
                provider="openai",
                model=model,
                sample=sample_label,
                attempt=attempt,
                error_type="network_error",
                detail=str(exc),
            )
            raise BenchmarkAPIError(f"OpenAI API request failed: {exc}", error_type="network_error") from exc

    latency_seconds = time.perf_counter() - started_at
    raw_content = parsed["choices"][0]["message"]["content"]
    usage = parsed.get("usage", {})
    log_event(
        "request_success",
        provider="openai",
        model=model,
        sample=sample_label,
        latency_ms=round(latency_seconds * 1000, 2),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )
    structured = None
    try:
        structured = json.loads(raw_content)
    except json.JSONDecodeError:
        structured = None
    return structured, raw_content, usage, latency_seconds


def run_benchmark(
    *,
    dataset_path: str,
    model: str,
    cache_path: str,
    limit: int | None = None,
) -> dict:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    samples = load_jsonl(dataset_path)
    if limit is not None:
        samples = samples[:limit]

    cache = load_cache(cache_path)
    predictions = []
    ground_truths = []
    metadata = []
    latencies = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = 0.0

    log_event(
        "benchmark_start",
        provider="openai",
        model=model,
        dataset=dataset_path,
        samples=len(samples),
    )

    for idx, sample in enumerate(samples, start=1):
        sample_label = f"{idx}/{len(samples)}"
        key = cache_key(model=model, prompt_version=PROMPT_VERSION, text=sample["input"])
        cached = cache.get(key)
        if cached:
            log_event(
                "cache_hit",
                provider="openai",
                model=model,
                dataset=dataset_path,
                sample=sample_label,
            )
            structured = cached["prediction"]
            raw_content = cached["raw_content"]
            usage = cached["usage"]
            latency_seconds = cached["latency_seconds"]
        else:
            log_event(
                "cache_miss",
                provider="openai",
                model=model,
                dataset=dataset_path,
                sample=sample_label,
            )
            structured, raw_content, usage, latency_seconds = call_openai_gpt_4o_mini(
                text=sample["input"],
                api_key=api_key,
                model=model,
                sample_label=sample_label,
            )
            cache[key] = {
                "prediction": structured,
                "raw_content": raw_content,
                "usage": usage,
                "latency_seconds": latency_seconds,
            }

        predictions.append(structured)
        ground_truths.append(json.loads(sample["output"]))
        metadata.append(
            {
                "template_id": sample.get("template_id"),
                "noise_level": sample.get("noise_level"),
            }
        )
        latencies.append(latency_seconds)
        usage_totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_totals["completion_tokens"] += usage.get("completion_tokens", 0)
        usage_totals["total_tokens"] += usage.get("total_tokens", 0)
        total_cost += estimate_cost_usd(model, usage)

    save_cache(cache_path, cache)

    metrics = calculate_metrics(
        predictions,
        ground_truths,
        sample_metadata=metadata,
        latencies_sec=latencies,
    )
    metrics["usage"] = usage_totals
    metrics["total_cost_usd"] = round(total_cost, 6)
    metrics["cost_per_1k_requests_usd"] = round((total_cost / len(samples)) * 1000, 4) if samples else 0.0
    metrics["samples_evaluated"] = len(samples)
    log_event(
        "benchmark_complete",
        provider="openai",
        model=model,
        dataset=dataset_path,
        samples=len(samples),
        valid_json_rate=metrics["valid_json_rate"],
        avg_field_accuracy=metrics["avg_field_accuracy"],
        total_cost_usd=metrics["total_cost_usd"],
        cost_per_1k_requests_usd=metrics["cost_per_1k_requests_usd"],
    )
    return metrics


def build_markdown_report(dataset_path: str, model: str, metrics: dict) -> str:
    lines = [
        "# OpenAI Benchmark Comparison",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Model: `{model}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        f"- Samples: `{metrics['samples_evaluated']}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Valid JSON | {metrics['valid_json_rate'] * 100:.1f}% |",
        f"| Avg Field Accuracy | {metrics['avg_field_accuracy'] * 100:.1f}% |",
        f"| Business Rule Compliance | {metrics['business_rule_compliance'] * 100:.1f}% |",
        f"| Hallucination Rate | {metrics['hallucination_rate'] * 100:.1f}% |",
        f"| p50 latency (ms) | {metrics.get('latency_ms_p50', 0):.2f} |",
        f"| p95 latency (ms) | {metrics.get('latency_ms_p95', 0):.2f} |",
        f"| Cost / 1k requests (USD) | {metrics['cost_per_1k_requests_usd']:.4f} |",
        "",
        "## Per-template breakdown",
        "",
        "| Template | Accuracy |",
        "|---|---|",
    ]

    for template_id, accuracy in sorted(metrics.get("accuracy_by_template", {}).items()):
        lines.append(f"| {template_id} | {accuracy * 100:.1f}% |")

    lines.extend(
        [
            "",
            "## Noise breakdown",
            "",
            "| Noise bucket | Accuracy |",
            "|---|---|",
        ]
    )
    for bucket, accuracy in sorted(metrics.get("accuracy_by_noise_bucket", {}).items()):
        lines.append(f"| {bucket} | {accuracy * 100:.1f}% |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the evaluation set against OpenAI models.")
    parser.add_argument("--dataset", default="data/test.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_BENCHMARK_MODEL", OPENAI_DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-path", default="results/openai_benchmark_cache.json")
    parser.add_argument("--output-json", default="results/benchmark_comparison.json")
    parser.add_argument("--output-md", default="results/benchmark_comparison.md")
    args = parser.parse_args()

    metrics = run_benchmark(
        dataset_path=args.dataset,
        model=args.model,
        cache_path=args.cache_path,
        limit=args.limit,
    )

    artifact = {
        "provider": "openai",
        "model": args.model,
        "dataset": args.dataset,
        "prompt_version": PROMPT_VERSION,
        "metrics": metrics,
    }

    Path(args.output_json).write_text(json.dumps(artifact, indent=2))
    Path(args.output_md).write_text(build_markdown_report(args.dataset, args.model, metrics))


if __name__ == "__main__":
    main()

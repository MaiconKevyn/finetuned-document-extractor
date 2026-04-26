import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import compute_percentile


def load_inputs(dataset_path: str, limit: int | None = None) -> list[str]:
    with open(dataset_path) as f:
        texts = [json.loads(line)["input"] for line in f if line.strip()]
    return texts[:limit] if limit is not None else texts


def post_extract(host: str, text: str, timeout_seconds: int = 60) -> tuple[int, float]:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = Request(
        f"{host.rstrip('/')}/extract",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    with urlopen(request, timeout=timeout_seconds) as response:
        response.read()
        return response.status, time.perf_counter() - started_at


def run_load_test(host: str, texts: list[str], concurrency: int) -> dict:
    if not texts:
        raise ValueError("No inputs available for load test")

    latencies = []
    success = 0
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(post_extract, host, text) for text in texts]
        for future in as_completed(futures):
            status, latency = future.result()
            latencies.append(latency * 1000)
            if status == 200:
                success += 1

    duration = time.perf_counter() - started_at
    return {
        "requests": len(texts),
        "success_rate": round(success / len(texts), 4),
        "throughput_rps": round(len(texts) / duration, 2),
        "latency_ms_p50": round(compute_percentile(latencies, 50), 2),
        "latency_ms_p95": round(compute_percentile(latencies, 95), 2),
        "latency_ms_p99": round(compute_percentile(latencies, 99), 2),
        "latency_ms_max": round(max(latencies), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple concurrent load test for /extract.")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--dataset", default="data/test.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    texts = load_inputs(args.dataset, limit=args.limit)
    results = run_load_test(args.host, texts, concurrency=args.concurrency)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

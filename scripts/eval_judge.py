import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_apis import OPENAI_API_URL, cache_key, load_dotenv

DEFAULT_MODEL = "gpt-4o-mini"
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "match": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["match", "reason"],
}


def load_cache(path: str) -> dict:
    cache_file = Path(path)
    if not cache_file.exists():
        return {}
    return json.loads(cache_file.read_text())


def save_cache(path: str, cache: dict) -> None:
    cache_file = Path(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2))


def judge_name_match(
    *,
    predicted: str,
    ground_truth: str,
    original_text: str,
    model: str = DEFAULT_MODEL,
    cache_path: str = "results/judge_cache.json",
) -> dict:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    cache = load_cache(cache_path)
    key = cache_key(
        model=model,
        prompt_version="judge_v1",
        text=f"{predicted}\n{ground_truth}\n{original_text}",
    )
    if key in cache:
        return cache[key]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are evaluating whether two extracted employee names should count as a match. "
                    "Allow accent differences and harmless formatting differences, but do not allow different people."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Predicted: {predicted}\n"
                    f"Ground truth: {ground_truth}\n"
                    f"Original document text:\n{original_text}"
                ),
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_match",
                "strict": True,
                "schema": JUDGE_SCHEMA,
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
    try:
        with urlopen(request, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI judge request failed with status {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI judge request failed: {exc}") from exc

    result = json.loads(parsed["choices"][0]["message"]["content"])
    cache[key] = result
    save_cache(cache_path, cache)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge helper for employee_name mismatches.")
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(
        json.dumps(
            judge_name_match(
                predicted=args.predicted,
                ground_truth=args.ground_truth,
                original_text=args.text,
                model=args.model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

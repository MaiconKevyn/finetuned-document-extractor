"""
Model integrity verification — mitigates OWASP LLM03 (Supply Chain).

Usage:
    python scripts/verify_model_integrity.py --register   # first run: compute and save hashes
    python scripts/verify_model_integrity.py              # subsequent runs: verify against saved hashes

Hashes are stored in models/checksums.json. Commit this file to git
so any tampering with model weights is detectable via diff.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

CHECKSUM_FILE = Path("models/checksums.json")

# Files that matter for inference correctness — only the essential artifacts
TRACKED_PATHS = [
    "models/doctune-qwen-1.5b-lora/adapter_config.json",
    "models/doctune-qwen-1.5b-lora/adapter_model.safetensors",
    "models/doctune-qwen-1.5b-lora/tokenizer.json",
    "models/doctune-qwen-1.5b-lora/tokenizer_config.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
            h.update(chunk)
    return h.hexdigest()


def compute_hashes() -> dict[str, str]:
    result = {}
    for rel in TRACKED_PATHS:
        p = Path(rel)
        if not p.exists():
            print(f"  [SKIP] {rel} — file not found")
            continue
        digest = sha256(p)
        result[rel] = digest
        print(f"  [OK]   {rel}\n         sha256: {digest[:16]}...")
    return result


def register():
    print("Computing hashes for tracked model artifacts...")
    hashes = compute_hashes()
    if not hashes:
        print("No files found. Check that TRACKED_PATHS are correct.")
        sys.exit(1)
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_FILE.write_text(json.dumps(hashes, indent=2))
    print(f"\nSaved {len(hashes)} hashes to {CHECKSUM_FILE}")
    print("Commit models/checksums.json to git to enable future verification.")


def verify():
    if not CHECKSUM_FILE.exists():
        print(f"Checksum file not found: {CHECKSUM_FILE}")
        print("Run with --register first to initialize.")
        sys.exit(1)

    expected: dict[str, str] = json.loads(CHECKSUM_FILE.read_text())
    print(f"Verifying {len(expected)} tracked artifact(s)...")

    failures = []
    for rel, expected_hash in expected.items():
        p = Path(rel)
        if not p.exists():
            failures.append(f"  [MISSING]  {rel}")
            continue
        actual = sha256(p)
        if actual == expected_hash:
            print(f"  [PASS] {rel}")
        else:
            failures.append(
                f"  [FAIL] {rel}\n"
                f"         expected: {expected_hash[:16]}...\n"
                f"         actual:   {actual[:16]}..."
            )

    print()
    if failures:
        print(f"INTEGRITY CHECK FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f)
        sys.exit(1)
    else:
        print("All artifacts verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify model artifact integrity.")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Compute and save hashes (run once after training).",
    )
    args = parser.parse_args()

    if args.register:
        register()
    else:
        verify()

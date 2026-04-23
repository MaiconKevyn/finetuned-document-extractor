"""
Utility for re-splitting an existing JSONL dataset with a custom ratio.

Usage:
    python scripts/split_data.py                          # default: train.jsonl → 90/10
    python scripts/split_data.py --input data/train.jsonl --ratio 0.8
    python scripts/split_data.py --input data/my_data.jsonl --train out_train.jsonl --val out_val.jsonl

This script is a standalone utility — it is NOT part of the main pipeline.
generate_dataset.py already produces train.jsonl and val.jsonl with a 90/10 split.
Use this only when you need to re-split existing data with a different ratio.
"""
import argparse
import random


def split_dataset(input_file: str, train_file: str, val_file: str, split_ratio: float = 0.9) -> None:
    with open(input_file) as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_point = int(len(lines) * split_ratio)
    train_data = lines[:split_point]
    val_data = lines[split_point:]

    with open(train_file, "w") as f:
        f.writelines(train_data)

    with open(val_file, "w") as f:
        f.writelines(val_data)

    print(f"Split complete: {len(train_data)} train / {len(val_data)} val  (ratio={split_ratio})")
    print(f"  → {train_file}")
    print(f"  → {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-split a JSONL dataset.")
    parser.add_argument("--input",  default="data/train.jsonl",  help="Source JSONL file")
    parser.add_argument("--train",  default="data/train.jsonl",  help="Output train file")
    parser.add_argument("--val",    default="data/val.jsonl",    help="Output val file")
    parser.add_argument("--ratio",  type=float, default=0.9,     help="Train split ratio (default: 0.9)")
    args = parser.parse_args()

    split_dataset(args.input, args.train, args.val, args.ratio)

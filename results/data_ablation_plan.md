# DocTune Data Ablation Pack

## Objective

Measure whether adding representative golden-like OOD examples to the SFT data improves golden/OOD extraction without regressing the held-out synthetic test set.

## Leakage Control

- OOD evaluation set: `data/golden.jsonl`
- Golden-like training pool: `data/ablations/golden_like_pool.jsonl`
- The golden-like training pool is generated with a different seed and must not replace the OOD evaluation set.

## Variants

| Variant | Synthetic | Golden-like | Train samples | Train command | Eval command |
|---|---:|---:|---:|---|---|
| `synthetic_800` | 800 | 0 | 800 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --train-file data/ablations/synthetic_800/train.jsonl --val-file data/val.jsonl --output-dir models/ablations/synthetic_800 --lora-r 16 --learning-rate 0.0001 --epochs 3 --seed 42 --run-name doctune-synthetic_800` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/ablations/synthetic_800 --datasets data/test.jsonl data/golden.jsonl --output-json results/ablations/synthetic_800_eval.json --output-md results/ablations/synthetic_800_eval.md` |
| `synthetic_800_plus_50_golden_like` | 800 | 50 | 850 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --train-file data/ablations/synthetic_800_plus_50_golden_like/train.jsonl --val-file data/val.jsonl --output-dir models/ablations/synthetic_800_plus_50_golden_like --lora-r 16 --learning-rate 0.0001 --epochs 3 --seed 42 --run-name doctune-synthetic_800_plus_50_golden_like` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/ablations/synthetic_800_plus_50_golden_like --datasets data/test.jsonl data/golden.jsonl --output-json results/ablations/synthetic_800_plus_50_golden_like_eval.json --output-md results/ablations/synthetic_800_plus_50_golden_like_eval.md` |
| `synthetic_800_plus_100_golden_like` | 800 | 100 | 900 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --train-file data/ablations/synthetic_800_plus_100_golden_like/train.jsonl --val-file data/val.jsonl --output-dir models/ablations/synthetic_800_plus_100_golden_like --lora-r 16 --learning-rate 0.0001 --epochs 3 --seed 42 --run-name doctune-synthetic_800_plus_100_golden_like` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/ablations/synthetic_800_plus_100_golden_like --datasets data/test.jsonl data/golden.jsonl --output-json results/ablations/synthetic_800_plus_100_golden_like_eval.json --output-md results/ablations/synthetic_800_plus_100_golden_like_eval.md` |
| `synthetic_800_plus_200_golden_like` | 800 | 200 | 1000 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --train-file data/ablations/synthetic_800_plus_200_golden_like/train.jsonl --val-file data/val.jsonl --output-dir models/ablations/synthetic_800_plus_200_golden_like --lora-r 16 --learning-rate 0.0001 --epochs 3 --seed 42 --run-name doctune-synthetic_800_plus_200_golden_like` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/ablations/synthetic_800_plus_200_golden_like --datasets data/test.jsonl data/golden.jsonl --output-json results/ablations/synthetic_800_plus_200_golden_like_eval.json --output-md results/ablations/synthetic_800_plus_200_golden_like_eval.md` |

## Decision Rule

- Promote a new adapter only if golden/OOD accuracy improves materially without losing more than 1 percentage point on `data/test.jsonl`.
- If postprocess few-shot remains better on golden/OOD, collect more representative labels before another main training run.
- Keep the current adapter as rollback until a candidate clears both test and golden gates.

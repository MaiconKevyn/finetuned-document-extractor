# DocTune Fine-tuning Research Pack

Objective: Measure whether QLoRA fine-tuning improves structured payroll extraction beyond prompt-only baselines while controlling for adapter rank, learning rate, and seed variance.

## Datasets

| Split | Path |
|---|---|
| train | `data/train.jsonl` |
| validation | `data/val.jsonl` |
| test | `data/test.jsonl` |
| golden | `data/golden.jsonl` |

## Ablation Grid

| Run | LoRA r | LR | Dropout | Epochs | Seed | Train command | Eval command |
|---|---:|---:|---:|---:|---:|---|---|
| `r4_lr5e-5` | 4 | 5e-05 | 0.05 | 3 | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r4_lr5e-5 --lora-r 4 --learning-rate 5e-05 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r4_lr5e-5` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r4_lr5e-5 --output-json results/research/r4_lr5e-5_eval.json --output-md results/research/r4_lr5e-5_eval.md` |
| `r4_lr1e-4` | 4 | 0.0001 | 0.05 | 3 | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r4_lr1e-4 --lora-r 4 --learning-rate 0.0001 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r4_lr1e-4` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r4_lr1e-4 --output-json results/research/r4_lr1e-4_eval.json --output-md results/research/r4_lr1e-4_eval.md` |
| `r8_lr5e-5` | 8 | 5e-05 | 0.05 | 3 | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r8_lr5e-5 --lora-r 8 --learning-rate 5e-05 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r8_lr5e-5` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r8_lr5e-5 --output-json results/research/r8_lr5e-5_eval.json --output-md results/research/r8_lr5e-5_eval.md` |
| `r8_lr1e-4` | 8 | 0.0001 | 0.05 | 3 | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r8_lr1e-4 --lora-r 8 --learning-rate 0.0001 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r8_lr1e-4` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r8_lr1e-4 --output-json results/research/r8_lr1e-4_eval.json --output-md results/research/r8_lr1e-4_eval.md` |
| `r16_lr5e-5` | 16 | 5e-05 | 0.05 | 3 | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r16_lr5e-5 --lora-r 16 --learning-rate 5e-05 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r16_lr5e-5` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r16_lr5e-5 --output-json results/research/r16_lr5e-5_eval.json --output-md results/research/r16_lr5e-5_eval.md` |
| `r16_lr1e-4` | 16 | 0.0001 | 0.05 | 3 | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r16_lr1e-4 --lora-r 16 --learning-rate 0.0001 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r16_lr1e-4` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r16_lr1e-4 --output-json results/research/r16_lr1e-4_eval.json --output-md results/research/r16_lr1e-4_eval.md` |

## Winner Seed Sweep

| Run | Seed | Train command | Eval command |
|---|---:|---|---|
| `r16_lr1e-4_seed42` | 42 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r16_lr1e-4_seed42 --lora-r 16 --learning-rate 0.0001 --lora-dropout 0.05 --epochs 3 --seed 42 --run-name doctune-r16_lr1e-4_seed42` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r16_lr1e-4_seed42 --output-json results/research/r16_lr1e-4_seed42_eval.json --output-md results/research/r16_lr1e-4_seed42_eval.md` |
| `r16_lr1e-4_seed123` | 123 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r16_lr1e-4_seed123 --lora-r 16 --learning-rate 0.0001 --lora-dropout 0.05 --epochs 3 --seed 123 --run-name doctune-r16_lr1e-4_seed123` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r16_lr1e-4_seed123 --output-json results/research/r16_lr1e-4_seed123_eval.json --output-md results/research/r16_lr1e-4_seed123_eval.md` |
| `r16_lr1e-4_seed7` | 7 | `python scripts/finetune.py --model-id models/Qwen2.5-1.5B-Instruct --output-dir models/research/r16_lr1e-4_seed7 --lora-r 16 --learning-rate 0.0001 --lora-dropout 0.05 --epochs 3 --seed 7 --run-name doctune-r16_lr1e-4_seed7` | `python scripts/eval_suite.py --model-id models/Qwen2.5-1.5B-Instruct --adapter-path models/research/r16_lr1e-4_seed7 --output-json results/research/r16_lr1e-4_seed7_eval.json --output-md results/research/r16_lr1e-4_seed7_eval.md` |

## Required Metrics

`avg_field_accuracy`, `valid_json_rate`, `business_rule_compliance`, `hallucination_rate`, `latency_ms_p50`, `latency_ms_p95`, `latency_ms_p99`, `adapter_size_mb`, `training_time_minutes`

## Decision Rule

Pick the smallest adapter whose test-set accuracy is within 1 percentage point of the best run, unless latency or hallucination metrics regress materially.

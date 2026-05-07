# DocTune Evaluation Report

Status: pending full regeneration after OOD golden-set expansion.

`data/golden.jsonl` was expanded from 8 smoke records to 200 categorized OOD records. The previous full report is no longer current for golden/OOD claims.

Current available artifacts:

- `results/artifact_results.json`: last full synthetic `data/test.jsonl` benchmark.
- `results/golden_smoke_eval.md`: limited 3-sample smoke run validating the expanded golden format and the few-shot + postprocess evaluation path.
- `results/data_ablation_plan.md`: commands for the next controlled data-mixture training runs.

Regenerate the full report with:

```bash
python scripts/eval_suite.py \
  --model-id models/Qwen2.5-1.5B-Instruct \
  --adapter-path models/doctune-qwen-1.5b-lora \
  --datasets data/test.jsonl data/golden.jsonl
```

The full benchmark should compare raw prompt-only baselines, few-shot + postprocess baselines, and the fine-tuned adapter before promoting a new main training run.

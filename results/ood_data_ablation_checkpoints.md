# OOD Data Ablation Checkpoints

## Decision Context

The current adapter wins on `data/test.jsonl`, but few-shot baselines win on the handcrafted OOD golden set. The next technical step is not another blind training run. The next step is a controlled experiment that separates data coverage from training recipe.

## Checkpoint 1: Expand Golden Evaluation Set

Goal: replace the 8-record smoke-style golden set with a 100-200 record categorized OOD benchmark.

Implementation:

- Generate `data/golden.jsonl` from `scripts/build_golden_set.py`.
- Keep 10 OOD categories with deterministic `golden_category`, `template_id`, `case_id`, and `noise_level`.
- Categories: PT-BR currency, EU number format, missing invoice, missing deductions, heavy OCR, unseen template, low gross, high gross, noisy reference, compact table.
- Keep this file as evaluation-only data.

Acceptance:

- `data/golden.jsonl` has 200 records by default.
- Every record has the full extraction schema in `output`.
- `scripts/check_data_quality.py` passes.

## Checkpoint 2: Add Few-shot + Postprocess Baseline

Goal: measure the production-realistic baseline, not raw LLM output only.

Implementation:

- Add deterministic output normalization in `src/postprocess.py`.
- Normalize currency strings such as `$5,000.00`, `R$ 5.000,00`, and `5.000,00 EUR`.
- Normalize invoice references such as `Ref# 39109`.
- Infer missing `deductions`, `tax`, or `net_pay` only when the payroll arithmetic uniquely supports it.
- Add `baseline_3shot_postprocess`, `baseline_5shot_postprocess`, and `baseline_10shot_postprocess` to `scripts/eval_suite.py`.

Acceptance:

- Unit tests cover currency parsing, invoice normalization, and arithmetic inference.
- Evaluation report exposes category breakdown for OOD records.

## Checkpoint 3: Build Data Ablation Pack

Goal: create reproducible training datasets for the OOD data-mixture study.

Implementation:

- Keep `data/golden.jsonl` as the fixed OOD evaluation set.
- Generate a separate `data/ablations/golden_like_pool.jsonl` with a different seed to avoid train/eval leakage.
- Build variants:
  - `synthetic_800`
  - `synthetic_800_plus_50_golden_like`
  - `synthetic_800_plus_100_golden_like`
  - `synthetic_800_plus_200_golden_like`
- Generate train/eval commands in `results/data_ablation_plan.md`.

Acceptance:

- `data/ablations/manifest.json` lists every variant with sample counts and paths.
- No ablation train file contains `data/golden.jsonl` records directly.

## Checkpoint 4: Train And Evaluate Candidates

Goal: measure whether adding OOD-like labels recovers golden performance without losing synthetic test performance.

Implementation:

- Train each variant with `scripts/finetune.py`.
- Evaluate each adapter with `scripts/eval_suite.py` on both `data/test.jsonl` and `data/golden.jsonl`.
- Summarize runs with `scripts/data_ablation_pack.py summarize`.

Acceptance:

- Each candidate has an eval artifact in `results/ablations/`.
- Summary includes test field accuracy, test business rule, golden field accuracy, golden business rule, and golden hallucination.

## Checkpoint 5: Decide Main Training Run

Decision rule:

- Promote a new main adapter only if golden/OOD accuracy improves materially without losing more than 1 percentage point on `data/test.jsonl`.
- If few-shot + postprocess still beats every adapter on golden/OOD, collect more representative labels before another main training run.
- Keep the current adapter as rollback until a candidate clears both test and golden gates.

Portfolio interpretation:

- Fine-tuning is still justified because it beats strong prompt-only baselines in-domain.
- The research gap is data representativeness, not merely hyperparameter choice.
- The production baseline must include schema validation and deterministic postprocessing.

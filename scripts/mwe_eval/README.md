# MWE Evaluation Suite

This directory contains an evaluation harness for REMERGE against span-annotated MWE datasets.

## Supported datasets

- `parseme12` (PARSEME Shared Task 1.2)
- `streusle` (STREUSLE v5.0)
- `coam` (Hugging Face dataset `yusuke196/CoAM`)

## Download data

```bash
uv run python -m scripts.mwe_eval.download --dataset all
```

Arguments:

- `--dataset parseme12|streusle|coam|all`
- `--data-root /path/to/data/root`
- `--force`

Default data root is `/Users/peter/projects/remerge-mwe/data/mwe_eval`.

## Run evaluation

```bash
uv run python -m scripts.mwe_eval.run_eval \
  --dataset streusle \
  --split dev \
  --iterations 50 \
  --method log_likelihood
```

Core algorithm flags (forwarded to `remerge.run`):

- `--method frequency|log_likelihood|npmi|logdice|t_score|delta_p`
- `--min-count K`
- `--min-score FLOAT`
- `--rescore-interval N`
- `--min-range N`
- `--range-alpha FLOAT`
- `--min-p-ab FLOAT`
- `--min-p-ba FLOAT`
- `--min-merge-count N`
- `--winner-min-score-output FLOAT`
- `--winner-min-range-output N`
- `--search-strategy greedy|beam`
- `--beam-width N`
- `--beam-top-m N`
- `--consensus-method METHOD` (repeatable convenience)
- `--consensus-min-run-support N`
- `--consensus-min-method-support N`

Decode/scoring flags:

- `--decode-policy all|longest_per_start|maximal_non_overlapping`
- `--suppress-subspan-types` / `--no-suppress-subspan-types`
- `--prediction-stopword-policy none|block_stopword_stopword|block_any_stopword`
- `--prediction-block-punct-only` / `--no-prediction-block-punct-only`

Dataset and I/O flags:

- `--dataset parseme12|streusle|coam|all`
- `--split train|dev|test`
- `--parseme-lang CODE` (repeatable)
- `--stopword-policy none|block_stopword_stopword|block_any_stopword`
- `--stopword TOKEN` (repeatable)
- `--block-punct-only` / `--no-block-punct-only`
- `--download-missing`
- `--config scripts/mwe_eval/configs/default.json`
- `--output /path/to/results.json`

## Consensus config examples

Method-list convenience:

```json
{
  "dataset": "streusle",
  "split": "dev",
  "iterations": 50,
  "consensus_methods": ["frequency", "log_likelihood", "npmi"],
  "consensus_min_run_support": 2,
  "consensus_min_method_support": 2
}
```

Explicit per-run overrides:

```json
{
  "dataset": "streusle",
  "split": "dev",
  "iterations": 50,
  "consensus_runs": [
    {"method": "frequency", "min_count": 1},
    {"method": "log_likelihood", "min_p_ab": 0.4},
    {"method": "npmi", "search_strategy": "beam", "beam_width": 3}
  ]
}
```

## Output metrics

The runner emits JSON with:

- mention-level micro precision/recall/F1 on contiguous gold spans
- type-level precision/recall/F1 on contiguous MWE types
- counts including `gold_total`, `gold_contiguous`, `gold_discontinuous`, `predicted_total`, `matched`
- run timing and config snapshot

For `--dataset all`, it returns per-dataset results and a micro-aggregated summary.

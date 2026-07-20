---
kata: hdph
created: 2026-07-19
---

# POS training-data sufficiency and diversity study

`experiments/pos-tagger/evaluate_data_sufficiency.py` is a development-only,
gold-only study of the retained c2 recipe. It does not register a candidate,
change a quality floor, read the protected split, or authorize pseudo-labeling.

## Inputs and invariants

The command obtains train and development data only through the frozen
`tests/pos/evaluation/manifest.json` loader. It samples complete documents,
never retokenizes or relabels them, and records the resulting sample hash,
document-ID hash, domain token counts, source hashes, code revision, artifact
hashes, and both sample/model seeds. The POS evaluation is over all 24,822
retained development tokens. MWE utility uses only the aligned 5,366-token,
23-span STREUSLE development subset through the common bakeoff utility
function.

The default recipe matches retained c2 semantics: eight epochs, 262,144
feature buckets, the same direct-lexicon calibration, quantization, and
candidate pruning. A repeated condition changes only the documented training
document selection and trainer shuffle seed.

## Run

Run the economical five-replicate study with locally acquired pinned data:

```sh
uv run --no-sync python experiments/pos-tagger/evaluate_data_sufficiency.py \
  --acquisition-root data/pos_evaluation \
  --work-dir /tmp/remerge-pos-sufficiency \
  --output /tmp/remerge-pos-sufficiency/report.json \
  --replicates 5 --workers 2
```

It runs six learning-curve fractions (5%, 10%, 25%, 50%, 75%, and 100%) and
five leave-one-domain-out comparisons. For each held-out domain, the mixed
control starts from the leave-out sample's actual token count, then selects the
closest deterministic whole-document, proportionally stratified mixed sample.
The report records the requested count, actual delta, tolerance, and matching
status. A pair outside the documented 1% default tolerance is marked
`unmatched` and excluded from the diversity diagnosis rather than treated as
evidence. `--workers` makes condition runs independent and parallelizable.
`--dry-run` writes the complete condition plan and provenance without training
an artifact.

The MWE calculation is intentionally limited to representative learning-curve
sizes (5%, 50%, and 100%), because it uses the common harness's 10,000
document-bootstrap resamples. Its small 23-span development denominator makes
it diagnostic evidence, not an independent data-barrier conclusion.

## Calibration evidence

Framework verification on 2026-07-19 passed the focused study and bakeoff
controls, repository `prek --all-files`, and a development-only dry-run
calibration against `data/pos_evaluation`. The dry run accepted the pinned
train hash `e6a378…798e944` and development hash `ef962…a8f5df`, planned one
condition, and recorded `protected_final_evaluated: false`. It trained no
artifact and made no quality claim, so its diagnosis is deliberately
`unavailable`; the real report is the sole source of a data-barrier diagnosis.

One actual, explicitly non-authoritative calibration then trained the 1% sample
(13,514 retained train tokens). It completed c2 training in 11.45 seconds,
full-development inference in 0.88 seconds, and aligned development MWE
utility in 59.70 seconds; its development accuracy was 87.2573%. Its report
was labelled `calibration`, recorded `protected_final_evaluated: false`, and
reported `inconclusive` because one condition cannot establish a curve or a
matched-diversity contrast. Its report is local, untracked evidence at
`/tmp/remerge-pos-sufficiency-calibration.Kmof0d/report.json`.

After committing the evaluator at `cd7ab16`, a second calibration trained all
199,199 retained training tokens. It completed training and artifact creation
in 100.65 seconds, full-development inference in 0.81 seconds, and MWE utility
in 59.30 seconds. Development accuracy was 93.8119%; the run remained labelled
`calibration` and recorded `protected_final_evaluated: false`.

Before a full run, an explicit calibration command may use fewer
fractions/repetitions and a dirty checkout:

```sh
uv run --no-sync python experiments/pos-tagger/evaluate_data_sufficiency.py \
  --acquisition-root data/pos_evaluation \
  --work-dir /tmp/remerge-pos-sufficiency-calibration \
  --output /tmp/remerge-pos-sufficiency-calibration/report.json \
  --fractions 0.05 --mwe-fractions 0.05 --replicates 1 \
  --no-diversity --calibration
```

The resulting report is explicitly labelled `calibration`; it cannot supply a
data-barrier diagnosis. A normal study refuses to run unless the implementation
is committed and the worktree is clean, has at least five ascending fractions
through 100%, five repetitions, and matched diversity controls. This prevents
the recorded code revision from naming an unrelated commit.

The normal five-repetition design has about 53 full-training-set equivalents
across learning and diversity conditions. Interpolating between the measured
1% and 100% calibrations gives roughly 95 CPU-minutes for 80 condition fits and
evaluations, plus about 15 CPU-minutes for the 15 representative MWE
calculations. Expect roughly 55--70 wall-clock minutes with two workers or
30--40 minutes with four, subject to CPU contention. This is an estimate, not a
performance claim; the committed full report records every actual condition
runtime.

## Report and diagnosis

The stable report contains per-condition quality metrics from the common
harness (overall, macro F1, OOV, ambiguity, per-domain, and supported-tag F1),
condition-local token-frequency and tag-support curves, artifact/training
details, runtimes, MWE utility where selected, and repeat-level two-sided 95%
Student-t intervals. This is intentionally not called a bootstrap or percentile
interval at five repetitions. The MWE fields retain the common 10,000-resample
document bootstrap. Aggregates cover per-domain accuracy, supported-tag F1,
token-frequency and tag-support curves, and every numeric MWE utility field.

The diagnosis compares paired full-vs-next-smaller learning-curve accuracy and
paired matched-mixed-vs-leave-out accuracy for every domain:

- `volume_limited` requires a positive full-data marginal interval, every
  matched diversity contrast demonstrably flat, and excess OOV or low-support
  error;
- `diversity_limited` requires a positive matched-domain contrast, a
  demonstrably flat full-data marginal interval, and the same error
  concentration;
- `architecture_limited` requires both axes demonstrably flat and no such
  concentration;
- any negative, conflicting, unmatched, or wide result is `inconclusive`.

For this interpretation, *flat* means its t interval lies entirely from zero
to +0.25 accuracy points. A negative interval is never called flat. The report
includes OOV, unseen/one-count token-frequency, and zero/low tag-support error
excess intervals so the diagnosis cannot silently treat concentrated data
errors as architectural evidence.

This is a research interpretation rule, not a release gate. A future,
separate augmentation provenance decision would require reproducible positive
data-barrier evidence, concentration in OOV/low-support slices, and
representative development MWE utility. This study cannot authorize a corpus,
teacher, LLM, or pseudo-label generation.

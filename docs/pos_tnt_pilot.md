---
kata: bkh6
created: 2026-07-20
status: ready-for-development-evidence
---

# TnT-style UPOS pilot

This is P1's first, development-only, gold-only pilot under decision `brx7`.
It does not authorize production integration, a new dependency, a corpus,
teacher output, pseudo-labels, network access, or protected-final evaluation.
`1wda` remains blocked until this pilot has an attributable report and a human
reassesses the result.

## Model and artifact

`experiments/pos-tnt/tnt.py` implements a deterministic first-party trigram
HMM over the frozen 17 UPOS tags.  It learns normalized deleted-interpolation
weights from leave-one-out n-gram counts, applies smoothed lexical emissions,
and treats EOS as a scored internal transition.  Unknown words use suffix
evidence from training words whose global frequency is at most the configured
bound, backing off from the longest suffix to shorter suffixes and finally to
the smoothed tag prior.  Seen words always use only the lexical route.

For a seen word, an unattested UPOS tag has exactly zero lexical emission and
is never enumerated by Viterbi. The artifact records and validates a numeric
word-to-candidate mask, so this is an exact model property rather than a beam.
Backpointers retain the exact best path without copying every prefix.

The compact JSON artifact is `remerge-pos-tnt-v2`.  It contains numeric counts,
configuration, hashes, and learned weights—not raw forms, sentences, or gold
labels.  Hash collisions during training are rejected.  The shared harness
enforces the unchanged 20 MiB artifact ceiling.

`tnt_adapter:create_candidate` is the common-harness adapter.  It requires
`REMERGE_POS_TNT_ARTIFACT` and reports canonical-boundary predictions only.

## Predeclared evidence protocol

The grid is fixed before execution and has exactly two configurations:

| Grid | Transition / lexical / suffix smoothing | Suffix length | Rare-word maximum |
| --- | --- | --- | --- |
| 0 | 0.10 / 0.10 / 0.25 | 3 | 1 |
| 1 | 0.25 / 0.25 / 0.50 | 4 | 2 |

The five fixed seeds are `20260720` through `20260724`.  TnT itself has no
random branch: these are explicitly recorded as deterministic repetitions,
not independent samples or a basis for confidence intervals.  The evaluator
trains and serializes all five, requires exact byte equality and matching
digests, then runs the expensive common harness once per byte-identical grid
artifact.  The other four records explicitly reuse that timing/resource
evidence; they never imply five independent timing samples.

Run only after obtaining the already-retained, checksum-verified development
data and the pinned c2 artifact (SHA-256
`3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d`,
identity `remerge-pos-linear-v1` / `unicode-whitespace-v1`). The evaluator
requires a clean committed repository HEAD; put the output directory outside
the checkout:

```sh
uv run --no-sync python experiments/pos-tnt/evaluate.py \
  --acquisition-root /absolute/path/to/acquired-data \
  --c2-artifact /absolute/path/to/c2.artifact \
  --output-dir /absolute/path/to/tnt-evidence \
  --output /absolute/path/to/tnt-evidence/report.json
```

For each c2/TnT artifact, the evaluator writes an exact-argv registration with
the Git revision, train/dev hashes, artifact digest and identity, environment,
and seed.  It then invokes `experiments/pos-tagger/bakeoff.py` rather than
reimplementing its calculations.  Thus each record has the shared quality
slices (overall, macro F1, OOV, ambiguity, supported tags, domains), MWE
utility, artifact size, isolated load/RSS, warm latency, throughput, and gates.
No option can select the final split, and all reports must set
`protected_final_evaluated: false`.

Before training, the evaluator copies only the manifest-pinned train, dev, and
development MWE bytes into a temporary read-only acquisition mirror. Training
and every common-harness subprocess use that mirror, so source paths cannot be
swapped after checksum verification. It never opens protected-final paths.

## Retain or reject rule

The report selects exactly one artifact: first retain over reject, then higher
overall accuracy, macro F1, OOV accuracy, ambiguous accuracy, and finally the
lower grid index. Its top-level decision is that selected artifact's result,
not an any-run best case. A selection retains only when it passes every
unchanged common-harness development gate and improves at least one of OOV
accuracy, ambiguous accuracy, generated-MWE precision, or generated-MWE
recall versus c2. Otherwise it rejects while preserving the specific reason.
A completed, attributable rejection is the intended bounded research outcome;
promotion or any final evaluation requires a separate human decision.

---
kata: 1wda
created: 2026-07-20
status: implementation-ready
---

# Averaged structured-perceptron UPOS pilot

This is the second and final P1 pilot selected by `brx7`. It is a
development-only, gold-only research implementation. It does not authorize a
production model, final evaluation, new dependency, CRF/third pilot, corpus,
teacher, pseudo-label, threshold change, or generated-tag API.

## Current evidence status

Implementation and focused controls are ready for primary review. Per the
current handoff instruction, the full two-grid, five-seed common-harness run
has **not** been executed, so this document makes no retain/reject decision.
The complete run must occur from a clean committed revision with output
outside the checkout. A later reviewed report must replace this status with
one explicit retain or reject result; attributable rejection is a complete
bounded research outcome.

## Model and feature contract

`experiments/pos-structured-perceptron/structured_perceptron.py` implements a
first-party averaged structured perceptron. Training shuffles complete
sentences from a fixed seed, predicts with the current raw weights, forms one
structured delta `Counter`, touches every changed parameter at most once for
that sentence, and advances the averaging clock exactly once for every
sentence, including correct predictions. Only final averaged weights are
serialized.

Decoding is exact Viterbi over all 17 UPOS tags at every token—there is no
candidate pruning or beam. Scores include BOS-to-first, every adjacent tag
bigram, and last-to-EOS. Observation scores are cached once per position, so
decoding is `O(tokens * 17^2)`. Equal-score paths use the lexicographically
smallest complete numeric tag path, including tag index zero.

There is one predeclared lexical/morphological family:

- bias, case-preserved word, lowercase word, collapsed character shape;
- lowercase previous and next form with explicit boundary markers;
- lowercase prefixes and suffixes of lengths one through four; and
- digit, hyphen, and initial-uppercase indicators.

Feature names are FNV-1a hashed into 131,072 buckets. The cutoff is applied
only to training counts **after hashing**. Distinct pre-hash templates that
collide preserve multiplicity in scoring and updates. A separate full-width
lexical identity hash rejects collisions between distinct training forms.
Neither feature names nor forms are stored in the artifact.

## Artifact and adapter

The `RMSP0001` artifact is a zlib-compressed canonical JSON payload behind a
fixed binary header containing exact lengths and a SHA-256 of the uncompressed
payload. It contains the frozen tag inventory, configuration, true training
seed, sentence-step count, a training-input digest, sparse averaged feature
weights, and sparse BOS/adjacent/EOS transition weights. It contains no raw
form, sentence, corpus, or gold sequence. Loading rejects over-size,
truncated, trailing, duplicate-key, checksum, compression, schema, tag,
index, non-finite, zero-weight, and aggregate-shape corruption. Both writer
and common harness enforce the unchanged 20 MiB ceiling.

`structured_perceptron_adapter:create_candidate` requires
`REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT`, reports model identity
`remerge-pos-structured-perceptron-v1` and tokenizer identity
`canonical-boundaries-v1`, and emits tags over the exact supplied nesting.

## Predeclared grid and true training seeds

The grid is fixed before the full evidence run:

| Grid | Epochs | Post-hash feature cutoff | Buckets |
| --- | ---: | ---: | ---: |
| 0 | 4 | 2 | 131,072 |
| 1 | 6 | 1 | 131,072 |

The five true training seeds are `20260720` through `20260724`. Each seed
drives per-epoch sentence shuffling; every artifact is trained and run through
the common harness independently. The evaluator rejects a grid unless all
five artifacts have distinct digests. These are repeat runs for stability,
not a confidence interval and not the byte-reuse shortcut appropriate to the
deterministic TnT model.

## Frozen inputs and comparators

Before any model or harness subprocess runs, `evaluate.py` copies and
checksum-verifies only the manifest-pinned EWT train bytes, EWT development
bytes, and development STREUSLE/MWE bytes into a temporary read-only mirror.
Training and all candidate/comparator harness calls use that mirror. The
evaluator has no final-split option.

The comparator inputs are exact rather than transcribed headline values:

- retained c2 artifact SHA-256
  `3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d`;
- rejected TnT selected artifact SHA-256
  `76e850be9510aa64d69299592884da5a39e17e3a5b9a22844179e876231a9e94`;
- completed TnT report SHA-256
  `fdb003f9f862837ced77bc198ed672942501b907d2b6d195bb0dab740ebb567d`.

The TnT report must parse as a development-only rejection. c2, the actual TnT
artifact, and every perceptron artifact then run through the same
`experiments/pos-tagger/bakeoff.py` snapshot for overall/macro F1, OOV,
ambiguity, supported tags, every domain, MWE utility, artifact/load/RSS,
throughput, and latency.

Registrations record shell-safe exact argv, clean 40-character Git revision,
exact train/dev and source-code hashes, artifact digest/identity, environment,
and seed. The top-level report additionally records the snapshotted MWE hash,
pinned TnT report/artifact hashes, Python/platform, and `protected_final_evaluated:
false`.

## Selection and decision semantics

For each grid, compute the five-seed mean in this declared lexicographic
order: overall accuracy, macro F1, OOV accuracy, ambiguous accuracy,
generated-MWE precision, generated-MWE recall. Select the higher aggregate
tuple, breaking an exact tie toward the lower grid index. The selected
artifact is then the fixed lowest seed (`20260720`) within that grid—never the
best seed.

That selected artifact is retained only if it passes every unchanged common
development gate and improves at least one of OOV accuracy, ambiguous
accuracy, generated-MWE precision, or generated-MWE recall versus actual c2.
Otherwise it is rejected with the exact common-harness and comparator reasons.
TnT remains a recorded rejected comparator; beating TnT alone cannot retain
the perceptron.

## Full evidence command (not yet run)

Run only after these files are reviewed and committed, using operator-held
pinned artifacts/reports and an output directory outside this repository:

```sh
uv run --no-sync python experiments/pos-structured-perceptron/evaluate.py \
  --acquisition-root /absolute/path/to/pos-evaluation \
  --c2-artifact /absolute/path/to/c2.artifact \
  --tnt-artifact /absolute/path/to/tnt-selected.json \
  --tnt-report /absolute/path/to/tnt-report.json \
  --output-dir /absolute/path/to/perceptron-evidence \
  --output /absolute/path/to/perceptron-evidence/report.json
```

A dirty or uncommitted HEAD, in-checkout evidence directory, altered input,
wrong comparator digest/identity, malformed TnT status, non-distinct seeds,
candidate alignment failure, artifact corruption, final-data signal, or
common-harness contradiction produces a machine-readable rejection and exits
nonzero.

## Bounded implementation benchmark

This is an engineering smoke measurement, not full candidate evidence or a
performance claim. On the development machine, using checksum-verified EWT
train, grid-0 cutoff/buckets, one epoch, and seed `20260720`:

| Shape | Result |
| --- | ---: |
| Training input | 11,733 sentences / 199,199 tokens |
| One training epoch | 23.460 s |
| Serialized artifact after one epoch | 839,467 bytes |
| First 100 dev sentences | 2,327 tokens |
| Inference on that dev prefix | 0.228 s |

Command shape: import the pilot and project loader, call `load_tagged_split`
on `data/pos_evaluation`, train `Config(epochs=1, feature_cutoff=2,
feature_buckets=131072)`, then time `tag_sentence` over the first 100 dev
sentences. The full harness remains authoritative for resources.

## Focused controls and residual work

`tests/pos/test_structured_perceptron_pilot.py` covers the feature family and
post-hash multiplicity, lexical collision rejection, exact exhaustive Viterbi
with EOS/full-path ties, cached local scores, one-touch structured delta,
explicit snapshot averaging, correct-example steps, true seeded artifacts,
all-tag/index-zero serialization, strict corruption and size rejection,
candidate alignment, aggregate/fixed-seed selection, shell-safe provenance,
and immutable train/dev/MWE snapshots.

Residual work is intentionally limited to primary review, commit, and the full
external evidence run. No final evaluation, production work, third pilot, or
augmentation work follows without the next human decision.

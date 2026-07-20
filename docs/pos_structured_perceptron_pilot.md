---
kata: 1wda
created: 2026-07-20
status: rejected
---

# Averaged structured-perceptron UPOS pilot

This is the second and final P1 pilot selected by `brx7`. It is a
development-only, gold-only research implementation. It does not authorize a
production model, final evaluation, new dependency, CRF/third pilot, corpus,
teacher, pseudo-label, threshold change, or generated-tag API.

## Evidence status

The full two-grid, five-seed common-harness run completed from a clean
committed revision. Both recipes and all ten artifacts were rejected. The
selected structured-perceptron artifact is not registered for protected
qualification or production use.

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

## Full evidence command

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

## Completed development evidence

The attributable run completed from clean revision `548a22d`. Its 116 KiB
report has SHA-256
`ac1012332573355ce207ff99d6630532cee7caebb60a0e5bb1db5d65735d2c04`.
It contains all ten distinct candidate artifacts, every fixed seed in both
grids, same-window c2 and TnT comparator reports, exact source/input hashes,
and `protected_final_evaluated: false` throughout. Every candidate decision
is `reject`.

| Grid | Overall mean | Macro-F1 mean | OOV mean | Ambiguous mean | MWE precision / recall mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0: 4 epochs, cutoff 2 | 94.073% | 90.835% | 78.258% | 93.862% | 8.82% / 26.09% |
| 1: 6 epochs, cutoff 1 | 94.184% | 91.048% | 78.624% | 93.941% | 8.82% / 26.09% |

The predeclared rule selected grid 1 and its fixed lowest-seed artifact,
SHA-256
`a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d`.
That artifact reached 94.134% overall accuracy, 90.933% macro F1, 78.427% OOV
accuracy, and 93.943% ambiguous-token accuracy. Against c2, those are gains
of 0.342, 2.532, 0.891, and 0.186 points respectively. It nevertheless missed
the unchanged 95% overall and 82% OOV floors.

Generated-MWE precision was 8.82%, 0.267 points below c2, and recall remained
26.09%. The 1.45 MiB artifact loaded in a 174 ms median with 65.6 MiB maximum
incremental RSS, but inference reached only 8,111 tokens/s against the 50,000
tokens/s floor. Its exact rejection reasons were the overall, OOV, MWE
precision-improvement, MWE recall, and throughput gates.

The structured model materially outperformed the rejected TnT candidate and
modestly improved c2's UPOS metrics, but it did not improve downstream MWE
utility or satisfy the frozen contract. This completes and rejects the second
and final P1 research packet. The portfolio now has no candidate eligible for
protected-final registration or production integration.

## Focused controls

`tests/pos/test_structured_perceptron_pilot.py` covers the feature family and
post-hash multiplicity, lexical collision rejection, exact exhaustive Viterbi
with EOS/full-path ties, cached local scores, one-touch structured delta,
explicit snapshot averaging, correct-example steps, true seeded artifacts,
all-tag/index-zero serialization, strict corruption and size rejection,
candidate alignment, aggregate/fixed-seed selection, shell-safe provenance,
and immutable train/dev/MWE snapshots.

No final evaluation, production work, third pilot, or augmentation work is
authorized by this completed rejection.

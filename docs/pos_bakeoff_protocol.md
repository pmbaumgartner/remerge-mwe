---
kata: 9vqe
created: 2026-07-19
---

# Candidate-neutral POS development bakeoff

This is a development-only research harness for comparing a classical,
distilled, or small-neural UPOS candidate without reintroducing a production
tagger surface.  It consumes the frozen train/dev inputs through the existing
checksum-verifying loader and has no `final` argument, final loader call, or
automatic download path.  The protected final split remains unavailable to the
ordinary research command.

## Candidate contract and registration

A candidate is a zero-argument `MODULE:FACTORY`.  The returned object must
have immutable `model_id`, `tokenizer_id`, `artifact_path`, `artifact_bytes`,
and `artifact_sha256` fields and a `tag(boundaries)` method.  Its output is a
document/sentence/token tensor of UPOS strings with exactly the supplied
boundaries; changed counts or unsupported tags reject the run.

Before a run, create a JSON registration containing `candidate_id`, source
revision, exact command, seed, train/dev data hashes, artifact digest,
environment, model identity, and tokenizer identity.  The command rejects a
missing, malformed, identity-mismatched, or data-hash-mismatched registration.
The old linear trainer's `trainer_command`/`dev_report` fields are retained in
its historical report only; the bakeoff translates the durable provenance into
the candidate-neutral names above.

## Command

Run from the repository root after acquiring the pinned data offline:

```sh
uv run --no-sync python experiments/pos-tagger/bakeoff.py \
  --candidate my_candidate:create_candidate \
  --acquisition-root /absolute/path/to/pos_evaluation \
  --registration /absolute/path/to/candidate-registration.json \
  --output /absolute/path/to/dev-bakeoff.json
```

The command always writes machine-readable evidence.  It exits nonzero for a
missing or invalid candidate, alignment failure, malformed manifest, checksum
mismatch, or invalid registration.  Results report overall accuracy, macro
F1, OOV accuracy, ambiguous-token accuracy, per-domain accuracy, and each
observed UPOS F1.  They also run the frozen downstream MWE configuration using
unfiltered, gold-filtered, and generated-filtered exact occurrences.

UPOS quality is measured on all 24,822 retained development tokens and all
five named EWT domains. Downstream utility uses the separately aligned
STREUSLE development subset (192 documents, 546 sentences, 5,366 tokens, and
23 spans); predictions are selected from the same full-development candidate
run by document and sentence identity.

The protected final quality thresholds in
[`pos_quality_contract.md`](pos_quality_contract.md) are reported as context,
and are applied as labelled *development analog* gates.  A failed analog gate
writes its report and exits nonzero; passing it is still not a final-pass
claim.  The report labels these hard development gates separately from the
protected-final diagnostic.  MWE intervals use 10,000 document-bootstrap
resamples with the project seed.

## Controls, repetitions, and c2

`tests/pos/test_bakeoff.py` includes known-good, rotated-output, malformed
alignment, invalid-tag, and registration-mismatch controls.  The normal run
uses three warmups and 15 warm inference samples; report consumers should
retry an environmentally noisy candidate rather than compare a single timing.
The common harness measures candidate-factory cold load and incremental RSS in
15 fresh subprocesses.  Raw-text end-to-end latency is not part of the
candidate-neutral contract because its tokenizer would impose a product API;
the fixed-boundary inference metric remains comparable across candidates.

The retained rejected c2 experiment can be reproduced without touching final
data:

```sh
uv run --no-sync python experiments/pos-tagger/bakeoff.py --reproduce-c2 \
  --train /absolute/path/to/en_ewt-ud-train.conllu \
  --dev /absolute/path/to/en_ewt-ud-dev.conllu \
  --artifact /tmp/c2.rmpos --output /tmp/c2.json
```

It invokes the isolated `experiments/pos-linear/train.py` with c2's retained
seed, epoch count, and bucket count, then rejects any result outside the
documented 93.7691% deterministic tolerance.  This is reproducibility
evidence for a rejected experiment, not a production adapter or a final-split
evaluation. Before invoking the trainer, the command requires the exact pinned
train and development SHA-256 values above; an arbitrary path, including the
protected split, is rejected before any model or report is produced.

The pinned acquisition reproduced c2 on 2026-07-19 at
`0.9376913594973956` development accuracy.  Its train/dev SHA-256 values were
`e6a3784727e7726d4f1c2e10ff22dcd0ddeb8869ad6f85b73bb1ff5ef798e944` and
`ef962ac05d844eaff46eeded125937129bfc0876d43963d66810cb73ffa8f5df`; its
4,496,292-byte artifact digest was
`3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d`.

After `4nkw` corrected the manifest's contradictory development-retention
aggregate, the candidate-neutral command evaluated the same artifact over all
24,822 retained development tokens. It correctly exited nonzero with a
machine-readable `rejected` report and did not load final data. The common
result was 93.7918% overall accuracy, 88.4006% macro F1, 77.5359% OOV
accuracy, and 93.7563% ambiguous-token accuracy; all five domain point
accuracies were reported. Generated filtering improved development precision
by 8.67 points and reached 26.09% recall, so it did not meet the development
utility screen. The Python research adapter's measured median load was 23.4
ms, maximum incremental peak RSS was 45.9 MB, median throughput was 31,578
tokens/s, and full-development warm p95 was 858 ms on this run. These resource
numbers characterize the adapter, not the isolated Rust loader, and are not a
frozen performance baseline. The report rejected c2 for overall, OOV, SYM F1,
MWE precision/recall, and throughput screens.

`experiments/pos-tagger/c2_adapter.py` is a first-party development adapter
for that local `RMPOS001` artifact.  Set `REMERGE_POS_C2_ARTIFACT` and use
`PYTHONPATH=experiments/pos-tagger` plus `c2_adapter:create_candidate` with
the common command.  The harness verifies
the artifact bytes and digest itself; candidate-declared values are never
trusted.  It records 15 isolated factory-load/RSS samples plus 15 warm samples
(median, p95, and IQR/median).  IQR/median above 10% invalidates the timing:
stabilize and retry once; a second high-variance result is rejected.

---
kata: e6wr
created: 2026-07-19
---

# POS quality and downstream evaluation contract

**Decision:** Accept the absolute, slice-aware, downstream-utility, and
resource gates in this document for an optional built-in English UPOS model.
They are protected acceptance material.  They do not apply to the accepted
pretagged-only fallback, which ships no model.

**Why now / higher-level goal:** The model is useful only if its tags improve
the occurrence-constrained MWE candidates without compromising the existing
fast unfiltered path.  A comparison with an unspecified teacher, or a filter
that removes no candidates, would not demonstrate that outcome.

**Goal check:** This contract preserves the MIT-only, supplied-tag path and
the separate CC BY-SA model posture in
[`pos_model_provenance.md`](pos_model_provenance.md).  It evaluates canonical
`TaggedDocument` word occurrences from
[`pos_annotation_contract.md`](pos_annotation_contract.md), and evaluates the
literal occurrence-constrained behavior settled in
[`pos_filter_semantics.md`](pos_filter_semantics.md).

**Consequential:** Yes — these gates select a model, determine whether the
project may distribute one, and must be frozen before tuning or final-test
inspection.  Peter accepted the recommended option below through his 2026-07-19
blanket acceptance of the project recommendations.

## Evidence and current baseline

- There is no POS tagger, POS gold set, MWE gold set, model benchmark, or
  final-evaluation harness in the repository yet.  The ordinary test suite
  checks deterministic existing discovery behavior, not POS quality.
- The only current performance guard runs five frequency-discovery iterations
  against 359 transformed transcript files and permits 25 seconds.  It neither
  measures model cost nor has sufficient provenance to be a correctness or
  release-quality input.  It remains a legacy core smoke guard only until it
  is replaced or separately proven permissible.
- No teacher, teacher outputs, or TnT baseline has been approved.  A
  comparative-to-teacher release gate would therefore be unexecutable and
  would create a path to changing success after observing results.

`wtp0` supplies the frozen gold manifest and `0bhv` implements this contract.
Neither may change a number, denominator, split, or pass condition below.

## Viable threshold families

### A. Absolute, slice-aware, utility-first gates **(accepted)**

Use immutable gold labels, absolute token-quality floors, fixed error slices,
and a downstream comparison against the same unfiltered discovery run.  Require
both a useful precision/selectivity change and bounded loss of in-scope gold
MWEs.  Use same-machine performance comparisons in addition to portable
absolute resource ceilings.

This directly tests the promised product value and remains executable without
an external model.  It costs a small, licensed MWE evaluation set and a
deterministic benchmark fixture.

### B. Teacher or TnT parity

Accept a candidate when it is within a chosen gap of a strong offline tagger,
and use token accuracy only as a diagnostic.

This is a useful future diagnostic once a specific teacher has a provenance
decision, but it is not an acceptance gate in v1: no such model is approved,
and parity can preserve a poor result or hide MWE utility.

### C. Functional and throughput-only acceptance

Check alignment, API compatibility, model size, and speed, with no frozen gold
quality or downstream threshold.

This is insufficient: a uniformly wrong or no-op filter could pass.

## Protected evaluation design

### Canonical units and fixed manifest

`wtp0` must create a versioned manifest whose hashes and split assignment are
frozen before model/rule/threshold tuning.  Its canonical unit is the
validated `TaggedDocument` word occurrence `(document, sentence, token)`, not
raw whitespace, an interned token type, or a CoNLL-U range/empty-node line.

- Each source sentence/document belongs to exactly one of **train**, **dev**,
  or **final**.  No document, duplicate normalized token sequence, or derived
  feature record may cross splits.
- `final` contains at least 5,000 canonical gold tokens, at least 500 OOV
  tokens, at least 500 ambiguous tokens, and at least 1,000 tokens in each of
  three named source/domain slices.  A token whose surface form has been seen
  in training but is absent after the frozen normalizer is not OOV.
- **OOV** means an NFC, case-preserved surface form absent from the frozen
  training vocabulary.  **Ambiguous** means a form seen with two or more UPOS
  values in training; ambiguity is determined from training only.
- `final` also contains at least 200 in-scope, adjudicated gold MWE occurrence
  spans and at least 300 emitted-candidate occurrences under the fixed
  unfiltered baseline.  If the manifest cannot meet these adequacy floors,
  there is no built-in-model release claim; reshape to pretagged-only or create
  a separately approved data packet.
- A domain slice is a manifest-defined disjoint group with its own source or
  genre field.  EWT web genres are acceptable when their pinned upstream
  metadata supports the assignment.  Do not call the current unproven
  transcript corpus a spoken-domain slice.

The manifest records source/revision/license, adapter version, exact retained
word IDs, split and fixture hashes, MWE annotation guidance/adjudication, MWE
pattern set, discovery configuration, and the expected number of each unit.
Gold labels, final-text inputs where restriction requires it, and result files
are not included in a production artifact.

### Anti-leakage and candidate registry

Only train/dev data may select features, rules, model family, model size,
confidence threshold, templates, score/min-count settings, or any numeric
threshold.  A candidate must be registered by its source revision, trainer
command, seed, artifact digest, and dev report **before** it can be run on
`final`.  The final evaluator is read-only and emits a result keyed to that
registration.

One registered candidate receives one final evaluation.  A changed model,
rule, tokenizer, template, or threshold is a new candidate and may not use a
previous final result to choose its settings.  A final-result failure is a
release failure, not an invitation to tune on that result.

The harness must reject a split/hash mismatch, overlap, missing unit count,
wrong canonical token sequence, unknown model identity, non-single-threaded
measurement, and a result lacking the candidate registration.  Its self-tests
must show that a deterministic UPOS rotation fails the POS gate and that an
all-wildcard/no-op filter fails the utility gate.

## Hard acceptance gates

All accuracy, F1, and utility quantities use final-set point estimates and a
two-sided 95% interval clustered by document.  The harness uses 10,000 seeded
document-bootstrap resamples; the seed and interval method are reported.  A
hard lower-bound gate passes only when its lower bound meets the stated floor.

### 1. UPOS quality

The candidate must satisfy every applicable row:

| Metric on canonical final tokens | Required lower 95% bound |
| --- | ---: |
| Overall micro UPOS accuracy | 95.0% |
| Macro F1 over all observed UPOS classes | 80.0% |
| OOV-token accuracy | 82.0% |
| Ambiguous-token accuracy | 88.0% |
| Each named domain slice | 90.0% |
| Each UPOS class with at least 50 final occurrences | 50.0% F1 |

For every named domain slice, accuracy must also be within 5.0 percentage
points of overall accuracy.  A model fails rather than averaging away one bad
domain.  Classes with fewer than 50 final occurrences appear in the report but
are diagnostic only; the manifest adequacy rules must make that exception
uncommon.

There is no public confidence API in v1.  Thus calibration is not a release
gate.  An internal confidence used to select a fast path must be chosen only on
dev and reported on final as a diagnostic coverage/accuracy table; it may not
relax any row above or be surfaced to callers without reopening this decision.

No teacher comparison is an acceptance row in v1.  If a teacher or TnT is later
approved under a source-specific provenance decision, `0bhv` may report its
same-manifest result as diagnostic evidence only.  Turning it into a pass/fail
comparison requires a new `e6wr` decision before candidate final runs.

### 2. Downstream POS-filtered MWE utility

The final manifest fixes a nonempty OR-list of POS patterns, score method,
`min_count`, `min_score`, exhaustion policy, and requested-winner budget `K`.
The same frozen raw token input and discovery configuration run three ways:

1. **unfiltered:** existing discovery without a POS filter;
2. **gold-filtered:** occurrence-constrained discovery using final gold UPOS;
3. **generated-filtered:** the identical filter using candidate-generated
   UPOS.

An emitted candidate occurrence is `(winner type, document, sentence,
start-token, end-token)` after deterministic overlap cleanup.  It is a true
MWE only when its boundaries exactly equal an adjudicated gold span.  The
in-scope gold denominator is every adjudicated span of length at least two
whose gold UPOS sequence matches a fixed complete pattern and is representable
under the settled sentence/tokenization rules.  This definition intentionally
does not credit an intentional out-of-pattern exclusion as a recall failure.

For a run `r`, let `TP_r` be exact matching candidate occurrences, `C_r` all
emitted candidate occurrences, and `G` the in-scope gold denominator:

`precision_r = TP_r / C_r`; `recall_r = TP_r / G`.

The candidate passes only if all of the following are true:

- `precision_generated - precision_unfiltered >= 10.0` percentage points;
- at least 10.0% of unfiltered candidate occurrences are absent from the
  generated-filtered result (`1 - |C_generated ∩ C_unfiltered| / |C_unfiltered|`);
- `recall_generated >= 70.0%` and is no more than 5.0 percentage points below
  `recall_unfiltered`;
- `recall_generated` is no more than 5.0 percentage points below
  `recall_gold-filtered`; and
- the lower 95% bootstrap bound for the precision improvement is greater than
  zero, while the upper bound for each stated recall loss is at most its limit.

The first two rows prohibit a no-op filter.  The last two distinguish tagger
error from the intended selectivity of the approved POS templates.  Empty
candidate sets, a zero gold denominator, or an inadequate manifest fail rather
than yielding a vacuous percentage.  Report boundary-near misses, type-level
versus occurrence-level aggregation, and per-pattern results as diagnostics;
they do not replace exact-occurrence primary metrics.

### 3. Compatibility, packaging, and resources

- With no POS feature selected, the existing raw `run()`/`annotate()` API and
  results remain byte-for-byte equivalent on the frozen compatibility corpus.
  The POS runtime is neither loaded nor initialized.
- The separately installable model artifact is at most **20 MiB** compressed,
  has a recorded digest and CC BY-SA/provenance notice, and contains no raw
  corpus, gold labels, teacher output, Python training runtime, or neural
  runtime.
- On the declared same-machine reference profile, cold artifact load has a
  median of at most **250 ms** and incremental peak RSS of at most **128 MiB**.
  Cold load starts from a new process; RSS is measured against the same process
  without the artifact.
- On the project-authored deterministic 100,000-token batch, single-threaded
  inference-only median throughput is at least **50,000 tokens/s**.  Raw text
  to generated tags to occurrence-filtered MWE results is at least **35,000
  tokens/s**, and at least **35%** of same-input unfiltered-core throughput.
- On the deterministic 256-token small-call shape, warm end-to-end p95 is at
  most **15 ms**.  The long-segment shape reports peak RSS and completes
  without a tokenization or sentence-boundary divergence; it has no separate
  speed floor.
- The unfiltered core's median throughput on the same project-authored fixture
  may regress by no more than **5%** from the frozen pre-POS baseline.

The 100k fixture must be project-authored, deterministic, seed-recorded, and
hash-checked.  It replaces the current 359-file corpus for new POS and release
evidence.  The legacy corpus can remain an opt-in historical core smoke guard
only; its output cannot establish any gate above.

For every timed metric, `0bhv` runs a release build with one thread, three
warmups, then 15 measured repetitions, and reports median, p95 where required,
IQR/median, CPU/OS/compiler, power mode, manifest digest, command, and model
digest.  If IQR/median exceeds 10%, rerun after environmental stabilization; a
second failure is an invalid performance result, not a pass.  Candidate and
baseline must run consecutively on the same machine and configuration.

## Required evidence and change authority

`0bhv` must provide one committed command with nonzero rejection semantics,
machine-readable output, and the known-good/broken-control results.  `t0ev`
records final evidence against this contract; its close is release readiness,
not package-exposure authority.

Peter is the sole acceptance/change authority.  An implementation or harness
agent may correct a demonstrable parser or measurement defect, but must stop
and request a new recorded decision before changing protected material.  A
proposal to revise a floor requires all of the following:

1. a precise manifest/harness defect or at least two registered dev-only
   candidate results showing that the floor is unsuitable;
2. the proposed replacement and its effect on every frozen dev report;
3. evidence that no final result informed the proposal; and
4. Peter's explicit acceptance recorded in Kata.

If final results have already been seen, a relaxed gate requires a newly frozen
independent final split before another release claim.  Missing a hard gate
means use the pretagged-only fallback or defer the model; it never authorizes
weakening this contract by implementation fiat.

## Decision record

**Finding:** The repository's present tests and legacy performance guard cannot
measure a POS model or the product-level value of POS filtering.  The settled
token, provenance, and filtering contracts make an absolute oracle feasible
without an unapproved teacher.

**Decision:** Peter accepted Option A on 2026-07-19 through his blanket
approval of all recommendations.  The hard gates, denominators, fixed-split
discipline, resource ceilings, and revision process above are binding.

**Recommendation:** `wtp0` creates the protected licensed manifest and `0bhv`
implements the rejection harness before model work scales.  `a4jp`, `wra2`,
and `t0ev` must replace their provisional numeric language with this contract.

**Confidence:** Medium.  The gates are internally coherent and intentionally
conservative, but no project-specific gold baseline exists yet.  A licensed
manifest and dev-only candidate results can test the thresholds without
contaminating final evaluation.

**Rejected alternatives:** Teacher/TnT parity is unexecutable and not enough
to show utility; functional/throughput-only acceptance permits a wrong or
no-op product.

**What would change this decision:** A new approved teacher can add a
diagnostic comparison but not alter acceptance.  A material threshold or
denominator change follows the protected revision process above.

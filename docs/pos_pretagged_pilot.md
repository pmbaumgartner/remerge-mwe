---
kata: npak
created: 2026-07-19
---

# Pretagged POS filtering pilot scorecard

## Outcome

**Recommendation: accept and scale.** The representative APD Lite pilot adds a
useful pretagged-only vertical slice without a model or runtime dependency. It
implements the accepted occurrence-level input, bounded pattern language,
filtered scoring population, internal support merges, exact emitted occurrence
coordinates, and annotation behavior while leaving the existing unfiltered
engine as the default path.

## Scope delivered

- `TaggedToken` and sentence-nested `TaggedDocument` with coordinate-specific,
  fail-fast UPOS/form/metadata validation.
- Strict `from_conllu()` conversion retaining integer word rows and explicit
  document/sentence boundaries while skipping range and empty-node rows.
- Separate `run_tagged()` and `annotate_tagged()` APIs; the legacy raw APIs do
  not initialize tagged state.
- Exact tags, `*`, and finite exact alternatives in OR-ed patterns of length at
  least two.
- Occurrence-constrained candidate counts, marginals, scores, merges, results,
  and annotations. Surface-identical rejected occurrences remain untouched.
- Viable proper-subspan support merges for longer MWEs. They are not emitted,
  annotated, or counted as requested iterations.
- `TaggedWinnerInfo.occurrences` with canonical document, sentence, start, and
  exclusive-end token coordinates for protected downstream evaluation.

## Pilot evidence

| Signal | Result |
| --- | --- |
| First targeted implementation pass | Passed all 13 initial POS-filter tests after the first Rust build |
| Final functional suite | 68 passed, 1 opt-in performance test deselected |
| Rust suite | 12 passed |
| Repository quality gate | Ruff format/check, ty, and Cargo Clippy passed |
| Tagged filtering smoke | 100,000 canonical tokens, two emitted winners, seven measured runs: 0.1167 s median / about 857k tokens/s |
| Existing performance guard | Passed at 0.745 s against its 25 s ceiling |
| Runtime dependency delta | None |
| Existing Python result compatibility | Legacy `WinnerInfo`, `run()`, and `annotate()` tests unchanged and passing |

The tagged smoke is a pilot measurement, not release evidence: it used a
project-authored repeated ten-token sentence on the current machine after two
warmups, and it measures validation plus filtering/discovery without a tagger.
The common harness owns repeatable release-build baselines, RSS, machine
metadata, and variance rules.

## Rework and discarded approaches

- The implementation uses a separate Rust `PosEngine`; mutating the optimized
  global `Engine` was rejected because it would put occurrence checks on the
  unfiltered fast path.
- Output-only filtering and type-level any/all/ratio gating were discarded by
  the accepted semantics because rejected occurrences would still affect
  discovery.
- The first public diagnostic draft reused the legacy winner converter. A
  mechanical return-path mistake made three legacy equality tests fail; one
  correction restored `WinnerInfo` on the raw path and kept
  `TaggedWinnerInfo` exclusive to tagged APIs. No accepted semantic design was
  retried or weakened.
- The downstream harness exposed that rendered strings are not an acceptable
  occurrence oracle. The pilot added structured coordinates instead of asking
  evaluation code to infer spans.

## Human and integration cost

Peter's 2026-07-19 blanket acceptance selected all three prerequisite
recommendations, so the pilot required zero additional decision rounds and no
waiting time for authority. The only integration discovery was the structured
occurrence diagnostic requested by the quality harness; it was additive and
did not change legacy results.

## Rollback

The tagged path is isolated behind new exports and a dedicated Rust module.
Reverting the pilot commit removes `rust/src/pos/`, the `PosEngine` binding,
tagged Python types/functions, tests, and documentation. The existing engine,
raw constructors, scoring implementation, and package dependency set remain
independently usable. Until later integration explicitly opts into a built-in
tagger, package users select this feature only by calling a tagged API.

## Scale gate

Proceed to the protected evaluation dataset, common harness, and candidate
tagger packets. Do not claim model release readiness from this pilot: model
quality, data adequacy, model licensing, end-to-end cost, and final utility
remain owned by their downstream Kata issues.

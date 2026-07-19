---
kata: bdxy
created: 2026-07-19
---

# POS-aware discovery thermo-nuclear code-quality review

## Decision

**Do not approve the current branch for stable release until the P1 findings
below are corrected and the P2 findings are corrected or explicitly waived by
the technical owner.** The branch is functionally green, but working behavior
does not compensate for a frozen implementation guardrail violation, dead
public optionality, inverted evaluation dependencies, or the continuing
maintenance cost of a rejected architecture.

This is a read-only maintainability review of
`main...b4970ec0680ff80e73075baabea6ba452be1b201` on branch
`codex/bdxy-pos-aware-discovery`. It applies an unusually strict standard:
prefer restructurings that delete concepts and branches over local cleanup that
merely moves complexity.

## Review scale and verification

The branch adds approximately 7,793 lines across 43 files. No file crosses
1,000 lines, but the largest pressure points are:

| File | Current lines | Structural concern |
| --- | ---: | --- |
| `tests/pos/conftest.py` | 822 | mixed accepted and rejected harness concerns |
| `src/remerge/core.py` | 808 | grew from 385 lines through duplicated API orchestration |
| `rust/src/pos/linear.rs` | 675 | rejected experimental loader compiled by normal Rust tests |
| `bin/train-pos-linear.py` | 656 | rejected experimental trainer remains active repository tooling |
| `tests/performance/test_pos_tagger.py` | 587 | large release harness for the rejected automatic-tagger route |
| `tests/pos/evaluation/loader.py` | 574 | manifest, parsing, split policy, and gold loading combined |

Verification at the reviewed revision remained green:

- `uv run --no-sync pytest -q -m "not performance"`: 85 passed, 3 deselected;
- `cargo test`: 17 passed; and
- `uv run --no-sync prek run --all-files`: Ruff, ty, and Cargo Clippy passed.

These checks establish the current behavioral baseline. They do not resolve the
structural findings.

## P1 findings — release blockers

### 1. Replace full-corpus POS candidate reconstruction with incremental state

`rust/src/pos/filter.rs:108-186` reconstructs every eligible occurrence,
sorts and cleans every candidate location set, rebuilds both marginal tables,
and rescales every candidate each time `select_candidate()` runs. The run loop
calls it after every internal support merge and every emitted merge.

This directly contradicts the accepted guardrail in
`docs/pos_filter_semantics.md:164-165`:

> Maintain filtered counts incrementally around changed locations; do not
> rescan the corpus after every merge.

The committed throughput gate requests one winner from a repetitive
100,000-token workload. It therefore does not exercise the multiplicative cost
of many support/full merges over a large, varied corpus.

#### Recommended fix

Keep the dedicated `PosEngine`; preserving isolation from the unfiltered fast
path was a good decision. Replace its global reconstruction with a local
`PosCandidateIndex` that owns:

- the eligible candidate key for each active sentence adjacency;
- candidate-to-non-overlapping-occurrence collections;
- left and right marginal counts plus the total eligible population;
- reverse references needed to remove an adjacency contribution; and
- dirty candidate keys whose score or overlap-cleaned population changed.

Perform one full initialization scan. For each merge, remove contributions for
the left neighbor, merged adjacency, and right neighbor; mutate the active
spans; then insert only the newly formed neighboring adjacencies. Candidate
selection may initially scan the maintained candidate map—the important
simplification is eliminating repeated corpus traversal and reconstruction.
Add a generation-backed heap only if profiling shows candidate-map selection is
the next constraint.

#### Acceptance evidence

- Differential tests compare every emitted winner and exact occurrence against
  a deliberately slow reference implementation for mixed tags, overlaps,
  multiple patterns, support merges, exhaustion, and score methods.
- Test-only instrumentation proves that after initialization a merge examines
  only the changed sentence neighborhood, not every sentence.
- Add a varied 100,000-token, multi-pattern benchmark requesting enough winners
  to exercise repeated support and full merges; report time versus winner count
  and reject superlinear corpus rescanning behavior.
- Existing supplied-tag results, annotation, unfiltered compatibility, and
  performance gates remain unchanged.

### 2. Remove the nonexistent built-in-tagger mode from the public contract

`src/remerge/core.py:67-71` exposes `TaggedDocument.source` with
`"supplied" | "builtin"` and an optional `model_id`. Validation adds further
branches at lines 369-384. The accepted package ships no built-in tagger or
model loader, and native discovery consumes neither value. Only rejected-model
benchmark/control code constructs `source="builtin"`.

This is speculative public API surface for an architecture that was explicitly
retired. If released, it creates compatibility obligations and suggests a
capability the package does not have.

#### Recommended fix

- Make `TaggedDocument` represent only the shipped contract: sentence-nested
  caller-supplied tokens, with English language validation if the language
  field remains useful.
- Delete `source`, `model_id`, their validation branches, tests, stub/docs
  references, and experimental adapters that require them.
- Keep experimental model and tokenizer identities in candidate-registration
  and benchmark evidence objects, not in the MWE discovery input model.
- Require a new outcome/contract decision before a future generated-tag path
  extends the public type.

#### Acceptance evidence

- Public signatures, stubs, README examples, and contract docs expose no
  built-in/generated-tag mode.
- Source, wheel, and sdist smoke tests continue to exercise supplied tags.
- A package search finds no public `source="builtin"` or public `model_id`
  contract.

### 3. Isolate or remove the rejected automatic-tagger implementation

`rust/src/pos/mod.rs:3-4` compiles the 675-line linear loader during normal
`cargo test`. The root repository also keeps the 656-line trainer, dedicated
model tests, a dynamic tagger protocol and loader, a 587-line generated-tag
release harness, and special artifact-audit logic that explicitly rejects
`linear.rs` if packaging configuration leaks it.

The code is not passive historical evidence: it participates in normal build,
test, lint, and package-safety maintenance. The accepted result was to reject
this architecture before final evaluation or public exposure.

#### Recommended fix

The strongest simplification is to delete the executable experiment from the
product branch and retain the evaluation report, hashes, commits, and Git
history. If live research reproducibility is still required, move the trainer,
loader, fixtures, and tests into an explicitly isolated
`experiments/pos-linear/` workspace that:

- is not a member of the production Cargo workspace;
- is not imported by package or ordinary test code;
- has its own dependency lock and explicit research commands;
- is excluded from wheels, sdists, release CI, normal `cargo test`, and product
  type checking; and
- cannot add public exports without a new Kata outcome and protected-oracle
  decision.

Remove the dynamic generated-tagger hooks and release options from the accepted
pretagged harness. Once the experiment is isolated, the artifact auditor should
need only a positive production allowlist, not model-specific exceptions.

#### Acceptance evidence

- Root `cargo test`, Python test discovery, release CI, and artifact audits do
  not compile, import, or special-case the rejected linear architecture.
- The retained research report still records trainer revision, dataset hashes,
  seed, candidate results, and artifact digests.
- If an experiment workspace is retained, its reproducibility command runs
  independently without changing the production dependency graph.

### 4. Give evaluation data and alignment logic one canonical owner

`bin/evaluate-pos-selected-user.py:18-27` mutates `sys.path` and imports the
private test helper `_read_streusle`. Lines 82-168 then duplicate corpus
alignment and span-selection behavior adjacent to the final-gold loader in
`tests/pos/evaluation/loader.py`. It also hard-codes the development STREUSLE
hash and expected shape outside the manifest.

This is an inverted dependency: a release-evidence command depends on private
test layout and a second source of frozen corpus truth.

#### Recommended fix

- Add the approved development MWE source, checksum, alignment policy, and
  expected aggregate shape to the existing evaluation manifest.
- Expose one typed `load_gold_split(..., split="dev" | "final")` operation
  that owns CoNLL-U/CONLLULEX parsing, alignment, span policy, and checksum
  validation.
- Put the selected-user evaluator in `tests/pos/evaluation/selected_user.py`
  and invoke it with `python -m tests.pos.evaluation.selected_user`, or place a
  clean reusable module under non-packaged project tooling. Delete the
  `sys.path` mutation and private import.
- Keep the CLI as a thin argument/evidence renderer over the canonical loader.

#### Acceptance evidence

- Development and final gold use the same alignment and span-policy code.
- Corpus paths, hashes, revisions, and expected shapes appear once in the
  manifest.
- Moving or renaming the test package cannot silently invalidate a release
  command.
- The selected-user aggregate result remains exactly reproducible.

## P2 findings — required cleanup or explicit waiver

### 5. Collapse duplicated Python execution and annotation flows

`src/remerge/core.py` grew from 385 to 808 lines. `run()`,
`run_with_occurrences()`, and `run_tagged()` repeat iteration validation,
progress execution, status handling, and result conversion. `annotate()` and
`annotate_tagged()` separately repeat the progress/non-progress split and the
zero-iteration rendering call. The copies already use different local names and
place status handling in different parts of the flow.

`TaggedWinnerInfo` also carries the wrong concept name:
`run_with_occurrences()` returns it for untagged discovery. The actual
distinction is ordinary winners versus winners with exact occurrences.

#### Recommended fix

- Extract a small `_execute(engine, request)` helper that owns progress,
  status/exhaustion handling, and `StepResult` collection.
- Extract `_execute_and_annotate(engine, request, rendering)` for the one-pass
  non-progress path and the zero-iteration render-after-progress path.
- Keep raw and tagged public functions direct: validate their specific inputs,
  create the appropriate engine, call the shared execution helper, and apply a
  result projector.
- Rename `TaggedWinnerInfo` to `WinnerWithOccurrences` or
  `OccurrenceWinnerInfo`; reserve “tagged” for input/filtering semantics.
- Move CoNLL-U/tagged types and normalization into a focused tagged-input module
  if `core.py` remains materially above its pre-change size after deduplication.

Avoid a highly generic runner framework. Two small explicit helpers are enough
to delete the duplicated lifecycle without introducing magic.

#### Acceptance evidence

- Every public call preserves its signature and result behavior except the
  pre-stable removal/rename explicitly accepted above.
- Progress, exhaustion, minimum score, annotations, and occurrence coordinates
  are covered once through shared lifecycle tests plus thin API-specific tests.
- `core.py` has a clear ownership boundary and materially fewer duplicated
  branches.

### 6. Centralize deterministic candidate ranking

`rust/src/pos/filter.rs:74-83` independently recreates the core
score/frequency/lexical tie-break and adds `MatchKind`, left, and right ordering.
The unfiltered policy lives separately in `rust/src/engine.rs:60-67`. The two
implementations can drift, and the precedence of support versus full matches is
currently an incidental consequence of enum ordering.

#### Recommended fix

- Define one small shared ranking function/key for score, frequency, merged
  lexical sequence, and a final stable identity.
- Have each engine provide its explicit stable identity. For the POS engine,
  encode any intentional full-versus-support preference as a named documented
  field rather than relying on derived enum order.
- Add cross-engine tie fixtures proving the shared score/frequency/lexical
  policy and POS-specific final tie behavior.

Do not introduce a generic candidate framework merely to share a comparator.
A small explicit rank value is sufficient.

#### Acceptance evidence

- One canonical implementation owns score/frequency/lexical ordering.
- Existing deterministic results remain unchanged or any deliberate tie-policy
  correction is separately recorded before release.
- Tests fail if either engine bypasses or redefines the shared policy.

## Recommended remediation sequence

1. **Freeze the reviewed behavior.** Add multi-iteration differential and
   scaling sensors before restructuring `PosEngine`.
2. **Remove rejected concepts first.** Delete the built-in public mode and
   isolate/remove the linear experiment so later refactors do not preserve dead
   abstractions.
3. **Make candidate state incremental.** Replace global reconstruction while
   using the slow implementation only as a test oracle until equivalence is
   proven.
4. **Clean the evaluation boundary.** Move dev-gold identity into the manifest
   and eliminate the private-test import.
5. **Deduplicate orchestration and ranking.** Collapse Python lifecycle copies
   and centralize deterministic Rust ranking after the accepted surface is
   smaller.
6. **Rerun release qualification.** Rebuild all artifacts and repeat source,
   Rust, compatibility, performance, artifact, sdist-rebuild, and isolated
   installed-wheel checks from the exact corrected revision.

This order is intentional: deleting the retired model path and ghost API first
prevents the cleanup from creating abstractions around behavior that should not
exist.

## Stable-release gate after remediation

Stable exposure may be reconsidered only when:

- all four P1 findings are closed with reviewed code and executable evidence;
- both P2 findings are corrected, or Peter records a specific technical waiver
  with rationale and a bounded follow-up packet;
- multi-iteration filtered scaling demonstrates that the implementation no
  longer rescans the corpus after every merge;
- no built-in/generated model surface or executable production-model baggage
  remains in the stable package path;
- the source, 17 Rust tests or their corrected successors, non-performance
  Python suite, `prek`, performance gates, wheel/sdist audits, and isolated
  artifact smokes pass from the exact release revision; and
- the `rrde` human stable-exposure decision is revisited using the corrected
  implementation evidence.

No current functional result is rejected by this review. The rejection is of
the implementation structure and release readiness, not the supplied-tag
product outcome.

---
kata: 6ey0
created: 2026-07-19
---

# Pretagged-only POS technical release evaluation

## Recommendation

**Authorize a prerelease to selected users, but do not authorize stable PyPI
exposure yet.** The supplied-tag implementation passes its technical release
gate at source revision `b4b8a11c2e88aad9305c1f4e6af2f986945d616e`.
Selected-user evidence remains necessary to demonstrate real candidate
reduction or precision benefit before stable exposure.

No built-in tagger qualifies. The protected model final split was not read,
and neither the wheel nor sdist contains or exposes a model, model loader,
trainer, corpus, neural runtime, download path, or runtime dependency.

## Functional and utility evidence

The normal source suite passed 85 tests with three opt-in performance tests
deselected; the Rust suite passed 17 tests. The source, direct wheel, and wheel
rebuilt from the sdist all passed the same `bin/pypi-smoke.py` checks for:

- unchanged unfiltered discovery;
- strict CoNLL-U conversion;
- supplied `TaggedDocument` and `TaggedToken` inputs;
- `run_tagged()`, `annotate_tagged()`, and exact occurrence coordinates; and
- absence of `_core.LinearPosModel`.

The fixed project-authored development control has 20 documents. Each document
contains both matching `ADJ NOUN` spans and nonmatching distractor bigrams, so
every document bootstrap resample remains nonempty. This is a deterministic
wiring/selectivity control, not a real-world quality claim.

| Development utility metric | Result | Gate |
| --- | ---: | ---: |
| Unfiltered precision | 50% | diagnostic |
| Supplied-tag precision | 100% | at least +10 points |
| Candidate occurrences | 800 → 400 | at least 10% reduction |
| Unfiltered and supplied-tag recall | 100% | at least 70% |
| Recall loss | 0 points | at most 5 points |
| Precision-improvement interval | +50 to +50 points | lower bound above 0 |

## Performance evidence

The frozen pre-POS baseline came from revision
`a26662fa5d7ede6f04cadc370ac55d5a9084e695` on the same Apple arm64 machine.
Both runs used the pinned 100,000-token fixture digest
`63d91f8a7316d36c2e8006102bafc36a19049d54b5bd3ecf432d7aeb2cd61427`,
one thread, three warmups, and 15 measured repetitions.

| Measurement | Result | Gate |
| --- | ---: | ---: |
| Supplied-tag validation + filtering | 978,860 tokens/s | at least 100,000 |
| Filtering median | 102.16 ms | diagnostic |
| Filtering IQR / median | 9.71% | at most 10% |
| Current unfiltered core | 3,506,737 tokens/s | no more than 5% regression |
| Frozen pre-POS core | 3,480,470 tokens/s | baseline |
| Unfiltered change | +0.75% | pass |
| Unfiltered IQR / median | 3.33% | at most 10% |

The first timing attempt was rejected because filtering IQR/median exceeded
10%. The one stabilized rerun passed; the final committed-harness verification
above also passed. No measurements were averaged across attempts and no
threshold changed.

Machine-readable timing evidence is in
`docs/evidence/pos-pretagged-release-b4b8a11.json`; its frozen baseline is in
`docs/evidence/pos-pretagged-core-baseline-a26662f.json`.

## Artifact evidence

`uv build --wheel --sdist --no-sources --clear` produced fresh artifacts. The
artifact audit applies a positive namespace/source allowlist, rejects corpus
and model formats, checks MIT metadata and zero `Requires-Dist` entries, and
checks that the native extension stub does not expose the rejected loader.

| Artifact | SHA-256 | Members | Smoke |
| --- | --- | ---: | --- |
| Direct wheel | `04f6488d1a009c5682cf451c14dad5b8350f799a40c27607ca6fc4cc239bf973` | 9 | passed |
| Sdist | `e37a8dbaedec52c6615a66590b3f4b43cc667487c34f4ec9b30f325ebfb0a9a2` | 21 | rebuilt successfully |
| Wheel rebuilt from sdist | `cb6bdc20f5747666c8707502a3d9a1133118e0c8bc109de87d3527b4565d1b8f` | 9 | passed |

The two wheel digests differ because they were separate builds; their audited
member policies and installed behavior agree. Full artifact evidence is in
`docs/evidence/pos-pretagged-artifacts-b4b8a11.json`.

The Cargo package now has an explicit build-input allowlist. In particular it
excludes all test corpora, evaluation data/manifests, docs, scripts, the POS
trainer, and `rust/src/pos/linear.rs`. The experimental Rust loader compiles
only for repository tests, is not registered with PyO3, and is absent from the
release stub and artifacts. The trainer and research evaluation remain in Git
but are not release inputs.

## Residual risk and next gate

- The utility control proves exact occurrence wiring and selectivity, not value
  on a user's corpus. Selected users must supply their own aligned UPOS and
  report candidate reduction/precision benefit plus unacceptable exclusions.
- Filtering variance was close to the 10% ceiling on the final run. Recheck it
  on the release runner; a result above the ceiling is invalid, not a waiver.
- No public package tag or upload was created. Artifact hashes identify local
  verification builds only.
- The retained research trainer/loader increases repository maintenance cost.
  The retention review must remove it or explicitly accept that cost.

The next human gate is whether to create a **prerelease for selected users**.
Approval does not authorize stable PyPI exposure. After selected-user evidence,
Peter separately decides stable exposure and residual risk.

## Rollback

The supplied-tag surface remains additive. Reverting the POS pilot and release
gate commits removes `run_tagged()`, `annotate_tagged()`, exact tagged
occurrences, and POS package docs while leaving the existing unfiltered engine
and APIs usable. Before any upload, rollback is only a source/package rebuild;
there is no release to yank.

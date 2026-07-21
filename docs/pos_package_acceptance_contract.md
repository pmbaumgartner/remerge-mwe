---
kata: hzrh
created: 2026-07-20
status: awaiting-human-decision
---

# Standalone POS package acceptance contract

**Decision to make:** Choose what the first standalone `remerge-pos` workspace
package must prove before its implementation can be accepted, and what evidence
is reserved for the later publication/exposure decision.

**Why now / higher-level goal:** Peter selected B2 and the averaged structured
perceptron because generic POS tagging has value independently of REMERGE's MWE
filter. Package extraction (`4gt3`) and qualification (`88xq`) are blocked until
this oracle is frozen.

**Goal check:** The package should give the existing perceptron a safe,
independently useful home where it can improve. It should neither claim broad
English representativeness from one web treebank nor remain blocked by an MWE
benchmark that is irrelevant to standalone POS use.

**Consequential:** Yes — this contract controls package architecture,
qualification, protected-final use, and the evidence available to the later
Human exposure gate.

## Observed evidence

- The selected fixed-seed perceptron reached 94.134% overall accuracy, 90.933%
  macro F1, 78.427% OOV accuracy, and 93.943% ambiguous-token accuracy on the
  pinned EWT development set. It improved all four metrics over c2.
- The artifact is 1.45 MiB, loads in a 174 ms median with 65.6 MiB maximum
  incremental RSS, and tags 8,111 tokens/s in the research harness.
- Its generated MWE precision was 8.82% and recall was 26.09%. Those results
  concern one small, 23-span development evaluation and do not establish the
  value of a standalone POS package.
- EWT development contains 24,822 retained tokens across five web genres. The
  protected EWT final split contains 24,575 retained tokens, including 2,207
  OOV and 12,776 ambiguous-form tokens. It is evidence for the pinned English
  web-domain source, not for spoken or unrestricted English.
- Existing provenance authority permits pinned EWT-only training/evaluation.
  It does not permit GUM, the repository's unproven transcript fixture, a
  teacher, pseudo-labels, or another corpus without a new source-specific
  decision.
- The old built-in contract couples 95% overall, 82% OOV, downstream MWE, and
  50,000 token/s gates to REMERGE product value. B2 explicitly preserves that
  historical contract but does not apply it to this standalone package.

## Options

### Q1 — Stage-separated experimental package qualification (recommended)

Accept the workspace package when extraction preserves the selected
perceptron's behavior and the package meets deterministic API, artifact,
installability, safety, provenance, and non-regression gates. Measure EWT POS
quality, including one protected-final run, but carry that result to the later
Human exposure gate instead of inventing a new absolute score floor after
seeing development performance. MWE metrics are adapter diagnostics only.

The package API is generic, but the initial model claim remains narrow:
"experimental English UPOS model trained and evaluated on UD English EWT."
This option lets the package exist and improve without presenting 94.134% as a
universal standard or silently treating it as production quality.

Trade-off: qualification proves faithful, safe packaging rather than adequate
quality for public distribution. `cp7t` must interpret the frozen final report
and explicitly decide whether a prerelease is responsible.

### Q2 — Comparative protected-final qualification

Add hard final gates requiring the perceptron to beat frozen c2 on overall and
macro F1 and to avoid material OOV or ambiguity regression, while retaining all
Q1 engineering gates.

Trade-off: this is less arbitrary than a round-number absolute threshold and
can reject a model that fails to generalize. But c2 is itself a rejected
research baseline, small point deltas can be noisy, and beating it does not
prove a broadly useful package.

### Q3 — Retain the old absolute contract

Require the standalone model to satisfy the existing 95% overall, 82% OOV,
MWE, and 50,000 token/s gates.

Trade-off: this has the strongest continuity with the stopped built-in route,
but the selected perceptron is already known to fail it and MWE performance is
not the standalone product outcome. Choosing Q3 effectively reverses B2.

## Recommendation

Choose **Q1**. The most honest first acceptance boundary is: “did we preserve
the chosen model exactly and package it safely?” Quality remains measured and
protected, but publication remains a separate Human judgment. This avoids
both failure modes Peter identified: a post-hoc 94.x cutoff and a small MWE
dataset controlling an independently useful POS capability.

Evidence that would change this recommendation is an independently approved,
representative English evaluation set with a product-derived minimum quality
requirement, or a specific downstream standalone use case whose error budget
implies a defensible threshold. Neither exists today.

## Candidate and acceptance boundary under Q1

- **Candidate:** the separately installable workspace code package built from
  `4gt3`, plus an explicitly supplied model artifact derived from the selected
  grid-1 perceptron recipe. Package, model, source revision, configuration,
  training-input digest, and artifact digest identify one candidate.
- **Accept/reject decision:** whether the package implementation may merge as
  an experimental workspace capability and advance to the Human exposure gate.
  Passing does not authorize publication, stable support, REMERGE integration,
  or a broad English quality claim.
- **Acceptance owner:** Peter.
- **Technical custodian:** the `88xq` qualification implementation may encode
  this contract but cannot change it.

## Package and API contract under Q1

- Distribution name: `remerge-pos`; import package: `remerge_pos`. The exact
  workspace directory is an implementation choice, but it must build and test
  independently of the root maturin distribution.
- The stable inference boundary accepts caller-tokenized, sentence-nested NFC
  forms and returns exactly one of the 17 canonical UPOS values per input
  occurrence with identical nesting. Empty sentences, whitespace-bearing or
  non-NFC forms, unsupported languages, changed counts, and invalid artifacts
  fail explicitly.
- Model loading is explicit from a caller-supplied local path. Importing or
  constructing the package performs no download, network access, discovery,
  or implicit model initialization.
- The first prerelease has no public confidence value and makes no raw-text
  tokenization promise. Adapters may be added later without changing the
  occurrence-aligned core.
- Training/export must be reproducible from a recorded command, seed,
  dependency lock, and frozen input digest. It may begin as a documented tool;
  it is not part of the stable inference compatibility promise.
- Code remains MIT. EWT-derived weights remain a separate explicit CC BY-SA
  artifact with attribution and model-card material. No raw corpus, gold
  labels, teacher output, Python pickle, or executable payload appears in the
  code or model distribution.

## Failure-mode-to-sensor matrix under Q1

| Material failure mode | Consequence | Observable sensor | Role | Known gap |
| --- | --- | --- | --- | --- |
| Extraction changes predictions | Silent model regression | Exact tag-tensor comparison with the frozen pilot artifact on all retained EWT dev tokens | Hard gate | Does not prove broader quality |
| Input/output misalignment | Tags attach to wrong occurrences | Shape, NFC, whitespace, sentence-boundary, invalid-tag, and randomized round-trip tests | Hard gate | Raw-text tokenization is out of scope |
| Unsafe or unbounded artifact load | Resource exhaustion or code execution | Bounded binary loader tests for truncation, trailing bytes, checksum, compression, duplicate keys, schema, dimensions, and size | Hard gate | Deliberate local resource starvation |
| Non-reproducible model | Unattributable releases | Rebuild registration, true seed/config/input digests, and deterministic inference comparison | Hard gate | Training bytes can vary across platforms if not controlled |
| Packaging leaks data or license scope | Legal/provenance failure | Wheel/sdist file-list and metadata inspection; model-card/license manifest checks | Hard gate | Trained-model legal classification remains a release-owner risk |
| Package breaks root project | Integration regression | Clean package install plus root non-performance suite and all-file checks | Hard gate | Future consumers not yet known |
| Material performance regression | Package becomes impractical | Same-machine comparison against the frozen extracted reference; artifact/load/RSS ceilings | Hard/comparative | Current Python harness is not a portable speed baseline |
| Poor model quality | Incorrect tags | Frozen EWT dev metrics and one registered protected-final report | Diagnostic for `cp7t` | EWT is web-domain evidence only |
| Poor MWE behavior | Weak future adapter | Frozen MWE report when adapter work begins | Diagnostic | Present dev MWE set has 23 spans |

## Inputs, baselines, and provenance under Q1

| Input or baseline | Frozen identity | Authorized role |
| --- | --- | --- |
| UD English EWT train/dev/final | Existing manifest, pinned EWT r2.10 commit and file hashes | Train; extraction parity/dev diagnostics; one-shot protected final |
| Selected perceptron recipe | Grid 1: 6 epochs, cutoff 1, 131,072 buckets, fixed seed 20260720 | Initial model recipe |
| Selected pilot artifact | SHA-256 `a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d` | Extraction parity reference; not directly publishable from `/private/tmp` |
| Pilot evidence report | SHA-256 `ac1012332573355ce207ff99d6630532cee7caebb60a0e5bb1db5d65735d2c04` | Frozen metric/resource context |
| Project-authored deterministic tokens | Generator, seed, count, and digest fixed by `88xq` before timing | Portable package and performance sensor |
| STREUSLE MWE annotations | Existing manifest only | Diagnostic for a later REMERGE adapter, never standalone acceptance |

No new dataset, source revision, teacher, pseudo-label, or augmentation is
authorized. A broader empirical claim requires a new provenance decision and
new Blue acceptance revision before the data is inspected for threshold
selection.

## Leakage and protected-final protocol under Q1

- Train and dev may inform implementation and model improvement. Protected
  final remains unavailable until the package code, model recipe, artifact,
  dev report, thresholds, and harness revision are registered.
- One registered candidate receives one final run. The report is retained even
  if it is unfavorable. No code, feature, model, threshold, or claim may be
  tuned from it.
- Under Q1, final quality is a required diagnostic for `cp7t`, not a package
  merge gate. A later proposal to create a numeric quality floor must use new
  independent evidence or freeze a replacement final set before rerunning.
- Final inputs, labels, predictions, and reports remain outside the checkout;
  only approved aggregate evidence may be committed.

## Gates and metrics under Q1

### Hard package gates

1. Exact dev prediction parity with the frozen pilot artifact over 24,822
   retained canonical tokens; metric recomputation must reproduce the recorded
   selected development values within `1e-12`.
2. Deterministic inference: 15 repetitions produce identical tag tensors and
   stable candidate/artifact identities.
3. All API alignment and invalid-input controls reject without partial output
   or fallback behavior.
4. Artifact loading rejects every declared corruption control, never uses
   pickle/eval/import hooks, and enforces the existing 20 MiB compressed limit.
5. Wheel and sdist build, install in a clean environment, import without a
   model, and pass the documented inference smoke test with an explicit model.
6. Code/model contents and metadata satisfy the MIT/CC BY-SA separation and
   contain no corpus or gold data.
7. Same-machine extraction comparison shows no more than 10% throughput
   regression from the frozen package reference; cold-load median remains at
   most 250 ms and incremental peak RSS at most 128 MiB. High timing variance
   invalidates rather than passes the result.
8. Root non-performance tests and all-file checks pass without installing a
   model or making `remerge-pos` a mandatory root dependency.

### Required diagnostics

- EWT dev and protected-final overall accuracy, macro F1, OOV accuracy,
  ambiguous-token accuracy, five web-genre accuracies, and per-UPOS F1.
- Artifact bytes, cold load, incremental RSS, batch throughput, warm latency,
  and training time.
- Paired differences from c2 on EWT final, clearly labeled diagnostic.
- MWE metrics only in a later adapter report; absence of MWE evaluation cannot
  reject the standalone package.

Every hard gate rejects independently. Diagnostics never silently become
gates.

## Exact execution contract under Q1

`88xq` must provide this repository-root command shape:

```sh
uv run --no-sync python tools/pos_package_qualify.py \
  --acquisition-root /absolute/path/to/pos-evaluation \
  --registration /absolute/path/to/registered-candidate.json \
  --pilot-artifact /absolute/path/to/structured-perceptron-g1-s20260720.rmsp \
  --output /absolute/path/to/pos-package-evidence/report.json
```

The command must run from a clean committed revision, snapshot and verify all
inputs before candidate execution, always emit machine-readable evidence, and
return zero only when every hard package gate passes. A failed diagnostic does
not change the exit code under Q1, but its complete result is mandatory.

## Negative controls under Q1

- Known-good: the package loader and inference API using a freshly rebuilt
  selected-recipe artifact must pass parity and engineering gates.
- Rotated UPOS output must fail prediction parity.
- Dropped, inserted, or reordered tokens/sentences must fail alignment.
- Truncated, trailing, checksum-mismatched, zip-bomb/oversize, duplicate-key,
  wrong-schema, invalid-tag/index, non-finite, and aggregate-shape artifacts
  must fail loading.
- A wheel containing corpus text or omitting required license/model-card files
  must fail distribution inspection.
- A package that downloads or initializes a model during import must fail an
  offline/network-denied smoke test.
- A deliberately slowed candidate must fail the same-machine non-regression
  gate.

Each control must produce the expected nonzero result and retained report.

## Repeatability and evidence under Q1

- Correctness and artifact checks are deterministic; retries are not allowed
  to turn a failure into a pass.
- Performance uses three warmups and 15 measured repetitions on one declared
  machine profile. If IQR/median exceeds 10%, stabilize and rerun once; a
  second noisy result is invalid.
- The report records oracle version, candidate/package/model identities,
  source revision, exact argv, Python/platform/dependency versions, frozen
  input hashes, per-gate results, diagnostics, timings, timestamp, and actor.
- PR-tier checks use synthetic fixtures and corruption controls. The slower
  offline release-tier run owns EWT parity, one-shot final, clean artifact
  builds, and performance evidence.

## Protected material and change authority

After Peter selects an option, this document, the evaluation manifest, pilot
artifact/report hashes, gate definitions, negative controls, and protected
final protocol are protected acceptance material. Peter alone may authorize a
change through a new Blue decision. Any change after candidate registration
invalidates that registration and requires rerunning all affected evidence; a
change informed by final results requires a newly frozen independent final
set for another quality claim.

## Decision needed

- **Q1:** qualify faithful, safe experimental packaging; reserve quality
  adequacy and publication for `cp7t`.
- **Q2:** add comparative final-quality rejection against c2.
- **Q3:** retain the old absolute built-in/MWE contract.
- **Recommendation:** Q1, because it implements B2 without replacing 95% with
  another post-hoc number or letting the small MWE set govern standalone POS.

Peter's choice is required before this contract is frozen and `4gt3` starts.

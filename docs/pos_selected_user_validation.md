---
kata: pz1k
created: 2026-07-19
---

# Pretagged POS selected-user validation

## Recommendation

**Proceed to the separate stable-exposure decision.** The bounded local
prerelease produced a useful 92.9% reduction in exact candidate occurrences on
the representative corpus, improved exact precision by 8.53 percentage points,
and did not exclude any gold occurrence found by the unfiltered workflow.

This recommendation does not authorize publication. Absolute supplied-tag
precision was 8.96% and 17 of 23 in-scope gold spans were still missed, so the
feature should remain an explicit expert filter rather than be presented as a
high-recall or high-precision MWE detector.

## Authorized exposure

Peter authorized a private/local, checksum-audited wheel with himself as the
initial selected user, one representative corpus, and aggregate/redacted
evidence only. No artifact was uploaded, no third party was contacted, and no
stable exposure was authorized.

The wheel was built from clean source revision
`74934fd200e2fdab9d8d343c8a05c40e089b7d9b`:

| Artifact | SHA-256 | Audit and smoke result |
| --- | --- | --- |
| `remerge_mwe-0.4.1-cp312-abi3-macosx_11_0_arm64.whl` | `d195e97f2dcb4c3a1faebdc7a79094be8b4a84e034f257e3f2bbddaa1bd58dc8` | accepted; isolated install passed |
| `remerge_mwe-0.4.1.tar.gz` | `e37a8dbaedec52c6615a66590b3f4b43cc667487c34f4ec9b30f325ebfb0a9a2` | accepted |

The audit found nine allowlisted wheel members and 21 allowlisted sdist
members, no runtime dependencies, and no corpus, model, trainer, or public
model loader. The installed wheel passed unfiltered discovery, supplied-tag
discovery and annotation, strict CoNLL-U conversion, exact occurrence
coordinates, and rejected-loader checks.

## Representative workflow

The selected-user corpus was the reviews portion shared by UD English EWT
r2.10 development data and STREUSLE v4.5 development annotations. Both sources
are CC BY-SA 4.0. This is public, non-final data already approved for development
use; the protected final split was not read.

| Corpus fact | Value |
| --- | ---: |
| Documents | 192 |
| Sentences | 546 |
| Tokens | 5,366 |
| In-scope exact gold spans | 23 |

The evaluator verifies the pinned revisions, the UD EWT development checksum
`ef962ac05d844eaff46eeded125937129bfc0876d43963d66810cb73ffa8f5df`,
the STREUSLE development checksum
`37d940dbeb0f4d63ed8d0929f91980e491f79a907b7d28c48ec6e242f2db94d6`,
canonical token/tag alignment, and the frozen aggregate corpus shape before
discovery. The fixed configuration used 500 requested winners, log likelihood,
minimum count 2, minimum score 0, stop-on-exhaustion, and the four accepted
patterns: `ADJ NOUN`, `NOUN NOUN`, `VERB NOUN`, and `VERB PART`.

The reproducible command shape is:

```console
python bin/evaluate-pos-selected-user.py ACQUISITION_ROOT \
  --artifact-sha256 d195e97f2dcb4c3a1faebdc7a79094be8b4a84e034f257e3f2bbddaa1bd58dc8 \
  --source-revision 74934fd200e2fdab9d8d343c8a05c40e089b7d9b \
  --selected-user Peter --output EVIDENCE.json
```

The command was run with the isolated wheel interpreter rather than the source
environment.

## Aggregate results

| Metric | Unfiltered | Supplied tags |
| --- | ---: | ---: |
| Selected winners | 336 | 27 |
| Exact candidate occurrences | 944 | 67 |
| Gold occurrences found | 4 | 6 |
| Exact precision | 0.42% | 8.96% |
| Exact recall | 17.39% | 26.09% |

Candidate occurrences fell by 92.90%, and precision increased by 8.53
percentage points (about 21 times the unfiltered precision). Fifty-three
filtered candidates overlapped the unfiltered result; 14 were new because POS
constraints changed the iterative discovery path. Two of those new candidates
were gold spans.

The filtered path excluded none of the four gold occurrences found by the
unfiltered workflow. It nevertheless missed 17 of 23 in-scope gold spans. That
remaining miss rate is recorded as a material limitation, not attributed to a
built-in tagger, because this release surface accepts the corpus's gold UPOS
tags directly.

## Selected-user feedback

- **Selectivity:** useful. Reducing 944 review items to 67 materially lowers
  manual review burden.
- **Unacceptable exclusions:** none observed relative to the unfiltered result;
  the low absolute recall remains an explicit product limitation.
- **Compatibility:** the macOS arm64 CPython 3.12 isolated-wheel workflow and
  existing package smoke checks passed without defects.
- **Runtime:** 20 ms for supplied-tag discovery versus 15 ms unfiltered on this
  small corpus. The absolute overhead was not operationally surprising and is
  consistent with the prior throughput gate.
- **Documentation and support:** the existing supplied-tag contract plus the
  checksum-enforcing evaluator were sufficient. The user must still provide
  aligned UPOS tags; no automatic tagger or download is implied. No code or
  documentation intervention was required after installation.
- **Privacy:** no corpus text, token examples, credentials, or candidate output
  is retained here or in Kata. Only public-source identifiers, hashes, and
  aggregate counts are recorded.

## Rollback and next gate

The artifact existed only in a task-owned temporary directory. That directory
was withdrawn from `/tmp` into the operating-system trash after evidence
capture, so the action remains locally recoverable. There is no registry
upload, release tag, third-party copy, or external user to notify. Rebuilding
or replacing the local wheel is the complete rollback mechanism; the stable
package remains unchanged.

The next gate is the Human-mode decision in `rrde`: publish the supplied-tag
surface as stable, extend validation, reshape it, or stop. The evidence supports
stable exposure as an additive expert feature if the release notes preserve
the low-absolute-recall limitation and explicitly state that users must supply
aligned UPOS tags.

---
kata: cp7t
created: 2026-07-21
---

# POS package exposure decision

## Decision

Peter approved **E1: a versioned standalone prerelease only** on 2026-07-21.

This authorizes public prerelease work for the independently installable
`remerge-pos` package and its qualified EWT model artifact under the boundaries
below. It does not authorize a REMERGE adapter, stable release, automatic model
download, an independent repository, or any additional training corpus.

## Evidence supporting exposure

The Q1 qualification on source revision
`47ab7cb3060d0319d71e02ed4be0afcca6f2d95a` passed all eight frozen hard gates:
extraction parity, determinism, API/alignment, artifact safety, reproducible
export, distribution/install, performance, and root compatibility.

The qualified candidate is `remerge-pos==0.1.0` with model ID
`remerge-pos-structured-perceptron-v1` and artifact SHA-256
`a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d`.
Its diagnostic-only EWT final result was 94.2625% overall accuracy and 0.9236
macro-F1. The result is evidence for English web text, not a broad-English or
MWE claim.

The complete aggregate record is in `docs/pos_package_qualification.md` and
`docs/evidence/pos-package-qualification-q1.json`.

## Publication authority

E1 authorizes `dr0g` to:

- publish a PEP 440 prerelease of the standalone MIT code distribution;
- publish the exact qualified EWT model artifact as a distinctly versioned,
  independently licensed model distribution under the conservative CC BY-SA
  4.0 posture already accepted in `eg08`;
- include a `LICENSES/` notice with the CC BY-SA 4.0 terms, EWT attribution,
  upstream URL and revision, modification/training statement, artifact
  checksum, and manifest reference, plus the model card, limitations,
  installation instructions, and explicit local-model smoke path;
- choose a normal public prerelease channel with a documented yank/removal
  procedure; and
- make release-only metadata changes required for a prerelease version,
  provenance, and project links.

The MIT wheel and sdist must not contain model weights, corpus text, gold
labels, or an automatic downloader. Installing or importing the code package
must remain offline-safe and useful without REMERGE.

The separately published model must preserve the exact qualified bytes and
SHA-256. Its own archive and metadata must declare CC BY-SA 4.0, and release
verification must inspect the archive member list, metadata, manifest, and
license/attribution notice. A generic unlicensed binary attachment is not an
authorized model distribution. Existing recipients' CC BY-SA rights cannot be
revoked; rollback means stopping new distribution, yanking the affected code
release where supported, and publishing a corrected or superseding prerelease
with an accurate notice.

## Qualification delta rule

The frozen candidate used package version `0.1.0`, while a standards-compliant
prerelease needs a prerelease version identifier. `dr0g` may apply a strictly
release-only version/metadata delta and must retain a reviewed diff attestation
from the qualified revision.

After that delta it must rerun every affected non-protected engineering,
packaging, install, safety, compatibility, and performance check. It must not
rerun or reinterpret the consumed protected final. Any change to executable
tagging behavior, artifact bytes, training recipe/data, acceptance thresholds,
or product claim exceeds E1 and requires a new qualification decision and an
independent future final set where applicable.

## Integration authority

E1 does **not** authorize `pnd7` or any other REMERGE adapter. Existing
unfiltered and caller-supplied-tag paths remain unchanged, and `remerge-pos`
must not become a root dependency.

The current small, single-dataset MWE evidence remains separate from the
standalone tagger decision. Reopening adapter work requires a later explicit
Human decision with its own compatibility, MWE-evaluation, and rollback scope.

## Stable-release and repository authority

No stable release is authorized. Promotion from prerelease requires a later
Human gate informed by installation feedback, artifact availability,
provenance/license review, defect history, and claim calibration.

No independent repository migration is authorized. The delivery profile stays
Core: one repository, one optional workspace package, and one accountable
maintainer.

Additional POS corpora remain governed by `hrxv`; E1 does not authorize their
download, label inspection, training use, or evaluation use.

## Alternatives not selected

- **E0:** rejected because it would retain good qualification evidence without
  delivering the independently useful tagger Peter selected at B2.
- **E2:** rejected because REMERGE integration is not the current objective and
  would entangle the separate MWE question without making the standalone tagger
  available.
- **E3:** deferred because it couples release and integration work despite the
  explicit decision to keep standalone POS value separate from current MWE
  evidence.

## Next actions

1. Close `pnd7` without implementation because adapter authority was withheld.
2. Execute `dr0g` as a prerelease-only Red packet under this decision.
3. Preserve a yank/removal path and report the exact published artifact hashes.
4. Hold separate future Human gates for stable release, REMERGE integration,
   independent-repository migration, and the `hrxv` corpus portfolio.

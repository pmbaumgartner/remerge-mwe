---
kata: eg08
created: 2026-07-19
---

# POS model data and distribution decision

**Decision:** Use the permitted source and distribution policy below for
the optional English UPOS model, its evaluation inputs, its benchmark fixture,
and any teacher-derived data.

**Why now / higher-level goal:** `bdxy` permits an opt-in, high-throughput
English UPOS tagger only when its data and model are reproducible and lawful to
redistribute. The unfiltered API must remain MIT-licensed and unchanged.

**Goal check:** The decision must retain a credible route to a qualifying
built-in model, while preserving the approved pretagged-only fallback. It must
not silently turn a source-data restriction into an obligation for package
users.

**Consequential:** Yes. This choice controls the rights to train, evaluate,
benchmark, package, and publish a model, and it constrains downstream model and
release packets.

**Evaluation criteria:** A source must have identifiable upstream terms, a
version that can be pinned and hashed, a compatible commercial-distribution
path, an auditable train/dev/final-test split, and no undisclosed runtime or
network dependency. A published artifact must preserve attribution and license
notices without relicensing unrelated MIT code by accident.

This is engineering provenance analysis, not legal advice. The release owner
should obtain legal advice or written permission before relying on an uncertain
trained-model classification.

## Observed repository state

- The main distribution declares `MIT` in `pyproject.toml` and has no POS
  model, corpus manifest, teacher, training script, or model-package metadata
  today.
- `tests/performance/reference_corpus/` contains 359 lowercased and transformed
  dialogue transcripts that were present in the initial commit (`109e004`). Its
  README identifies a mixture of the Santa Barbara Corpus (SBC) and ICE-Canada,
  but records neither source URLs, source/snapshot checksums, transformation
  recipe, nor redistribution terms. It is therefore not a usable provenance
  source or correctness oracle. The current repository snapshot's aggregate
  SHA-256 is `f4f4083516ac629f080fec21b381ec102f671df8bc04f18038a9eeb5635740cf`;
  this is only a fingerprint of the unproven local copy, not a permission chain.
- No candidate teacher model is named in the repository or the current Kata
  chain. Consequently, no teacher, teacher logits, pseudo-labels, or teacher
  weights are approved by this record.

## Source evidence and allowed roles

| Resource | Evidence | Licensing consequence | Proposed status |
| --- | --- | --- | --- |
| UD English EWT | UD identifies EWT as a gold-standard English corpus built from LDC English Web Treebank source material and labels this treebank **CC BY-SA 4.0**. It has all 17 UPOS tags. [UD EWT](https://universaldependencies.org/treebanks/en_ewt/index.html), [EWT license](https://github.com/UniversalDependencies/UD_English-EWT/blob/master/LICENSE.txt) | Attribution is required. If the trained model is treated as Adapted Material, a shared model must be CC BY-SA 4.0 or a compatible license. The trained-model classification is a legal uncertainty; this record takes the conservative path rather than assuming it is exempt. | Viable conditional source for train, development, and held-out gold evaluation. Never include raw CoNLL-U or source text in a wheel/sdist. Pin an upstream release/commit and record hashes before use. |
| UD English GUM | UD labels GUM **CC BY-NC-SA 4.0**; its repository explains that the aggregate reflects more restrictive underlying texts. [UD GUM](https://universaldependencies.org/treebanks/en_gum/index.html), [GUM license](https://github.com/UniversalDependencies/UD_English-GUM/blob/master/LICENSE.txt) | The license permits use and sharing only for NonCommercial purposes and applies ShareAlike to Adapted Material. That is incompatible with a generally commercially distributable PyPI model under the conservative policy. | Rejected for train, development, final evaluation, teacher inputs, or any shipped model. Do not use merely because it has valuable spoken and genre coverage. |
| Existing 359-file reference corpus | Repository-local README and initial-commit history only. SBC now offers free transcript downloads, but the evidence does not identify the exact source files or establish terms for the processed snapshot; no authoritative ICE-Canada permission was found. [SBC access page](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english) | Citations and current free access do not prove that this transformed mixed snapshot may be redistributed, relicensed, or used as model/evaluation evidence. | Unapproved. Do not use for training, tuning, gold evaluation, model artifacts, or teacher inputs. Do not carry it into any new distribution. The human owner must either remove/replace it from the public repository or obtain and record source-specific permission, exact provenance, and transformation hashes. |
| Project-authored deterministic benchmark fixture | Can be generated from project-authored token sequences with no third-party text. | No external corpus terms; the fixture can follow the repository's MIT policy. It cannot claim spoken-domain realism. | Approved in principle as the replacement runtime fixture, after the performance packet defines the generator, seed, expected digest, and workload rationale. |
| External teacher model / logits / pseudo-labels | No candidate is presently identified. Creative Commons notes that training on CC material requires respecting conditions when copyright permission is needed; it does not decide a particular model's downstream status. [CC AI-training FAQ](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) | A model card alone is insufficient. The exact artifact, license, redistribution permission, training-data restrictions, API terms, and status of generated logits must be reviewed before use. | Rejected until a separate, source-specific approval is recorded. The v1 classifier must train only from approved gold labels; no distillation path is assumed. |

UD itself warns users to consult each individual treebank license, because its
treebanks have mixed terms and the underlying text can be limiting. [UD
licensing guidance](https://universaldependencies.org/contributing/licensing.html)

## Viable distribution options

### Option A — EWT-trained, separately distributed CC BY-SA model (recommended)

Train a classical model only on a pinned UD English EWT training split; tune on
the pinned development split; report the untouched EWT test split once under
the frozen contract. Ship no corpus text, labels, or teacher artifacts. Publish
the weights and their manifest as a separate explicit model distribution (for
example, `remerge-pos-en-ewt`) under CC BY-SA 4.0, with the required EWT
attribution, source URL, upstream revision, and model-training notice. The main
`remerge-mwe` code distribution remains MIT and must not download the model
automatically.

This treats the weight file conservatively as an adaptation. CC BY-SA requires
attribution and, when Adapted Material is shared, a same-elements compatible
license. [CC BY-SA legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
Separate distributions also make their legal scope explicit: Python packaging
metadata applies the declared `License-Expression` to the containing
distribution file, and supports carrying license files in archives. [PyPA core
metadata](https://packaging.python.org/en/latest/specifications/core-metadata/),
[PyPA project metadata](https://packaging.python.org/specifications/declaring-project-metadata/)

Trade-offs: this is the strongest currently evidenced built-in path, but it
adds an explicit model installation step and needs release-owner confirmation
that the conservative CC BY-SA treatment is acceptable. It may not satisfy the
domain-slice gates; failure means reshape, not weaker acceptance.

### Option B — Pretagged-only release; defer a shipped model

Ship the settled supplied-tag filtering path, retain MIT-only package terms,
and require callers to supply UPOS tags. Publish no model or model-training
artifact until written rights or legal review supports a distribution strategy.

Trade-offs: this exactly matches the authorized `bdxy` fallback and is the
lowest legal-distribution risk, but it does not deliver the preferred built-in
tagger outcome.

### Option C — User-trained/local model only

Publish an optional training recipe that requires a user to acquire their own
approved corpus and creates a local model. The project distributes neither
weights nor source data.

Trade-offs: this avoids distributing a possibly adapted model, but does not
provide a reproducibly shipped built-in model and shifts corpus compliance to
users. It is not recommended as the primary v1 plan.

## Recommendation

Choose **Option A**, subject to the release owner accepting a CC BY-SA model
artifact that is separate from the MIT code distribution. Keep **Option B** as
the mandatory fallback if legal review, source pinning, or frozen quality gates
do not support Option A. Do not use GUM, the current reference corpus, or any
teacher-derived material in v1.

This recommendation is conservative rather than a claim that every trained
classifier is necessarily Adapted Material. Creative Commons identifies the
question as fact- and jurisdiction-sensitive; applying CC BY-SA to the separate
model artifact avoids relying on the more permissive interpretation.

## Required provenance and packaging checklist

`wtp0`, `0bhv`, `a4jp`, and `t0ev` must apply the following before a model is
accepted or exposed:

1. Record the exact upstream EWT release tag and immutable commit, canonical
   URL, license URL/text, acquisition date and command, SHA-256 for each source
   file, and the required citation/attribution.
2. Commit a manifest that assigns every source sentence/document to exactly one
   of train, development, or final-test; hash each split; record the
   whitespace-tokenization conversion, filtering, and excluded records. The
   final-test files and hashes are frozen before tuning.
3. Record trainer/source revision, command, dependency lock, seed, feature
   schema, model format/version, output artifact SHA-256, and deterministic
   rebuild result. Keep raw corpus, gold labels, dev/test text, and teacher
   outputs out of wheels/sdists and test fixtures.
4. For the separate model distribution, include a `LICENSES/` notice containing
   CC BY-SA 4.0, EWT attribution, upstream URL and revision, modification and
   training statement, model checksum, and this manifest reference. Inspect the
   built wheel/sdist file list and `METADATA` to confirm that the model package
   declares CC BY-SA 4.0 and contains its notice. The code package continues to
   declare only its MIT code terms.
5. Replace the current performance corpus with a project-authored deterministic
   fixture, or obtain a documented permission chain and exact reproducible
   source for every retained file. Until then, do not use its outputs as
   acceptance evidence and do not include it in future release artifacts.
6. Before any teacher experiment, create a source-specific decision record with
   the exact model artifact/digest, license and terms URLs captured on the
   acquisition date, rights to use the API/weights and redistribute logits,
   training-data restrictions, and a policy for excluding teacher outputs from
   shipped artifacts. No approval is implied by this document.

## Rejected alternatives

- **UD GUM for a broadly distributable model:** rejected because the official
  treebank license is CC BY-NC-SA 4.0, and a commercial-use restriction is
  incompatible with the intended package-distribution posture.
- **Treating the current reference corpus as implicitly permissible:** rejected
  because citation, a historic fixture, and current access are not a permission
  chain for the specific processed files.
- **Using an unspecified teacher “only offline”:** rejected because it would
  create untraceable training influence and makes it impossible to audit the
  permissions for logits or pseudo-labels.
- **Bundling EWT weights under the root MIT declaration without separation:**
  rejected because the project cannot demonstrate that the containing
  distribution's license accurately covers both code and a conservatively
  CC-BY-SA model artifact.

## Open legal and evidence uncertainties

- Whether a particular trained classical model is Adapted Material under
  CC BY-SA is not settled here. This record deliberately adopts the stricter
  distribution treatment; counsel or written rights-holder permission could
  support a different treatment.
- EWT's underlying English Web Treebank provenance is identified by UD, but the
  project must pin and preserve the exact UD release/commit and its supplied
  notices before use. The project must not infer rights beyond that treebank's
  license file.
- The current mixed SBC/ICE snapshot has neither a file-level origin map nor
  evidence of the terms that governed its 2022 acquisition and transformation.
  Current SBC availability does not resolve the ICE portion or the snapshot.
- No model/teacher has yet been nominated. A future model card, hosted API, or
  repository can change this record only after its own terms and artifact are
  audited.

## What would change the recommendation

- A source with a current, authoritative **CC0 or CC BY 4.0** permission for
  the exact data and explicit permission compatible with publishing a trained
  commercial model could replace EWT and simplify the model license.
- Written permission from the EWT rights holders that explicitly covers the
  intended model artifact and distribution terms could allow a different
  packaging arrangement.
- A complete, verifiable permission chain for every existing benchmark file
  could allow its retention as a benchmark only; it would still need separate
  suitability and leakage review before it informed model quality gates.
- A named teacher whose artifact terms, training-data restrictions, and
  output-redistribution rights survive source-specific review could enable a
  separately authorized distillation experiment.

## Decision record

- **Option A:** accepted. Use EWT-only training/evaluation under the pinned,
  no-corpus-in-artifacts policy and distribute a separately installed CC BY-SA
  model package; retain Option B if it fails legal or quality gates.
- **Option B:** mandatory fallback. Release pretagged-only functionality and defer shipped
  model artifacts until a future provenance decision.
- **Option C:** rejected as the primary v1 path. User-trained local models, with no project-shipped
  weights.
- **Decision:** Peter accepted Option A with Option B as the mandated fallback
  on 2026-07-19 through his blanket approval of all project recommendations. It
  is the release owner's acceptance of the separate CC BY-SA model posture,
  but it does not waive the recorded legal-review, provenance, quality, or
  packaging gates. This path was recommended because
  it is the only currently evidenced route to the preferred built-in capability
  while containing the ShareAlike obligation to a distinct artifact.
- **What would change the recommendation:** a more permissive authoritative
  corpus permission, written EWT model-distribution permission, or legal advice
  that supports an equally auditable, less restrictive arrangement.

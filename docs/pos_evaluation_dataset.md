---
kata: wtp0
created: 2026-07-19
---

# Frozen POS and MWE evaluation dataset

`tests/pos/evaluation/manifest.json` is the project-owned manifest for the
release oracle in [`pos_quality_contract.md`](pos_quality_contract.md). It
does **not** include corpus text or gold labels. The data must be acquired into
an untracked directory and checksum-verified before a protected evaluation.

## Selected sources and provenance

The manifest intentionally pins **UD English EWT r2.10**
(`b33472d3ce50a62056d6057f1b8a723e9d211176`) rather than a newer EWT release.
It is the revision used by **STREUSLE v4.5**
(`6c7855e717239e79321074765cc0f96dbcf72d1a`), so the MWE source and POS input
have exact sentence and integer-word alignment. Both sources are CC BY-SA 4.0;
STREUSLE records that its MWE and supersense annotations are CC BY-SA 4.0 and
that its EWT Reviews sentences are redistributed with permission of Google and
LDC. The project distributes neither corpus nor annotations in its MIT code
package.

The manifest records the official repositories, tags, immutable revisions,
license-file and data-file SHA-256 values, and attribution URLs. Its checked
counts were obtained solely by the acquisition validator:

| Final-set property | Frozen count |
| --- | ---: |
| Canonical EWT final tokens after frozen deduplication | 24,575 |
| OOV tokens against retained EWT train | 2,207 |
| Ambiguous-form tokens against retained EWT train | 12,776 |
| Answers / email / newsgroup / reviews / weblog | 5,278 / 5,761 / 3,695 / 5,347 / 4,494 |
| Contiguous, strong, length-2+ STREUSLE final MWE spans | 229 |

The source split contains duplicate normalized sentences, which would violate
the quality contract's no-cross-split-leakage rule. The frozen adapter keeps
the first NFC token sequence by train, then dev, then final precedence; it
retains 199,199/24,822/24,575 train/dev/final tokens. This deterministic
selection is versioned by the source hashes and retained-counts in the
manifest. Twelve STREUSLE final sentences are consequently unretained. The
count excludes them, as well as weak and discontinuous MWEs, because v1
discovery emits contiguous spans only. The validator proves that every retained
STREUSLE test sentence has exactly the same retained `(FORM, UPOS)` sequence in
EWT final before using any span. This produces more than the contract's 200
in-scope span floor without pretending that a gappy MWE is an exact span.

## Offline acquisition and validation

Acquire the two repositories outside this checkout at the pinned commits. Put
them under one untracked directory as follows:

```text
<acquisition-root>/ud-ewt/en_ewt-ud-{train,dev,test}.conllu
<acquisition-root>/ud-ewt/LICENSE.txt
<acquisition-root>/streusle/test/streusle.ud_test.conllulex
<acquisition-root>/streusle/LICENSE.txt
```

Then the protected harness calls:

```python
from pathlib import Path
from tests.pos.evaluation.loader import load_final_gold, load_manifest, validate_acquired_dataset

evidence = validate_acquired_dataset(load_manifest(), Path("<acquisition-root>"))
gold = load_final_gold(load_manifest(), Path("<acquisition-root>"), allow_final=True)
```

The loader has no download path. It rejects a missing file, hash mismatch,
invalid canonical token, count mismatch, document or normalized-sentence split
overlap, insufficient OOV/ambiguous/domain coverage, STREUSLE/EWT token
mismatch, or an inadequate MWE denominator. Its return value contains only
aggregate adequacy counts; the benchmark harness owns final metric calculation
and result emission.

## Protected-final discipline

`final` is protected acceptance material, not a development fixture. Raw EWT
files, STREUSLE labels, generated candidate outputs, and final results stay in
the separately provisioned acquisition/evaluation environment and are ignored
by this repository. Model/rule/threshold work receives only the train and dev
paths. A candidate is registered from dev evidence before the harness is given
the acquisition root; one registered candidate receives one final run. A
different model, tokenizer, pattern, or threshold is a new candidate.

The metadata and test loader are intentionally public because they contain no
gold text or labels. They are not a security boundary by themselves: the
separate filesystem/CI credential boundary prevents developers and tuning jobs
from reading final data. Any change to a source revision, hash, split, MWE
policy, filter setting, or adequacy floor requires Peter's recorded acceptance
under the quality contract; observing a final result never authorizes a retry
with tuned settings.

## Scope and residual risk

This dataset is a release-quality POS and contiguous-MWE oracle for the
accepted English web-domain source—not evidence for spoken language or a
general MWE benchmark. The EWT/Reviews sources cover five web genres and the
manifest explicitly does not describe any of them as spoken. The release
harness must still verify the frozen unfiltered candidate-count floor and the
quality/utility gates; this packet supplies the trustworthy input, not a model
pass result.

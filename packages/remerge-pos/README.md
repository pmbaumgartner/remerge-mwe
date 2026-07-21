# remerge-pos

`remerge-pos` is an experimental package for English Universal POS (UPOS)
tagging. Its first model claim is deliberately narrow: an experimental English
UPOS model trained and evaluated on UD English EWT.

## Experimental prerelease

The first public exposure is GitHub prerelease `remerge-pos-v0.1.0a1`. The MIT
code wheel/sdist and CC BY-SA EWT model are distinct distributions:

```text
https://github.com/pmbaumgartner/remerge-mwe/releases/tag/remerge-pos-v0.1.0a1
```

Install the code wheel from that release, download and verify
`remerge-pos-en-ewt-0.1.0a1.tar.gz`, then load its `model.rmsp` from an explicit
local path. The release page publishes SHA-256 checksums and the model archive
contains its own license, attribution, manifest, and model card.

This is not a stable release or a broad-English claim. It has no raw-text
tokenizer, confidence API, automatic model download, or REMERGE adapter. The
qualified evidence covers caller-tokenized English web text from UD English
EWT; MWE behavior was not a package gate.

## Inference boundary

The stable inference boundary is tokenized sentences, not raw text. Callers
provide sentence-nested, NFC-normalized token forms and receive one canonical
UPOS tag per input occurrence with the same sentence nesting. There are 17
canonical UPOS values. Empty sentences, whitespace-bearing or non-NFC forms,
unsupported languages, changed counts, and invalid artifacts fail explicitly.

Models load only from an explicit, caller-supplied local path. Importing or
constructing the package never downloads a model, accesses the network,
discovers model files, or initializes a default model.

```python
from remerge_pos import Tagger

tagger = Tagger.load("/path/to/model.rmsp")
tags = tagger.tag(
    (
        ("The", "watch", "stopped", "."),
        ("I", "watch", "birds", "."),
    )
)
```

`tags` is a tuple of sentence tuples with exactly one UPOS string for each
input form. Pass `language="en"` explicitly when a caller carries language
metadata; other declarations are rejected. The package does not attempt
language identification.

## Scope and provenance

The initial package is experimental. It does not make a broad English-quality
claim, provide confidence values, promise raw-text tokenization, or integrate
with REMERGE by default.

Code is MIT-licensed. EWT-derived model weights, if distributed later, are
separate CC BY-SA artifacts with their own attribution and model-card material.
The code package does not bundle a model, raw corpus text, gold labels, teacher
output, or executable model payload.

Training and export tooling may be documented for reproducibility, but are not
part of the stable inference API. They require a recorded command, seed,
dependency lock, and frozen input digest.

The provisional tooling is available from `remerge_pos.training`:

```python
from pathlib import Path

from remerge_pos.training import (
    SELECTED_CONFIG,
    SELECTED_SEED,
    train,
    write_artifact,
)

gold_sentences = (
    (("The", "DET"), ("watch", "NOUN")),
    (("I", "PRON"), ("watch", "VERB")),
)
model = train(gold_sentences, SELECTED_CONFIG, SELECTED_SEED)
digest = write_artifact(model, Path("model.rmsp"))
```

The selected recipe is six epochs, post-hash feature cutoff one, 131,072
feature buckets, and seed `20260720`. A reproducible build record must also pin
the complete training input and its digest; this package supplies no corpus.

## Artifact and compatibility boundary

The initial model format is `RMSP0001`, with schema identity
`remerge-pos-structured-perceptron-v1`. It contains the tag inventory,
configuration, seed, training digest, step count, sparse averaged observation
weights, and sparse transition weights. It contains no forms, sentences, gold
labels, Python pickle, or executable payload.

Loading is bounded to a 20 MiB compressed artifact and a 64 MiB uncompressed
payload. It rejects truncation, trailing data, checksum or compression errors,
duplicate JSON keys or weight rows, incompatible schemas, invalid tags or
indices, non-finite/zero weights, and invalid aggregate shapes.

An EWT-derived model distribution must remain separate from this MIT code
package. Its model card and license material must identify UD English EWT,
the pinned upstream revision and file hashes, the training command and input
digest, configuration and seed, `remerge-pos` version, artifact schema, artifact
SHA-256, attribution, and CC BY-SA 4.0 terms.

## Extraction record

The private perceptron implementation preserves the completed pilot's feature
family, FNV-1a hashing, exact Viterbi decoding and tie-breaking, update order,
averaging, canonical serialization, and valid `RMSP0001` bytes. Intentional
production-boundary changes are limited to:

- the public `Tagger` wrapper validates complete sentence nesting before
  inference and exposes only explicit local loading;
- file loading reads at most the accepted artifact limit instead of reading an
  arbitrarily large path before validation;
- malformed schema containers are normalized to explicit `ValueError`; and
- the historical experiment module is a compatibility shim over this package,
  leaving one maintained implementation.

Full EWT prediction parity, protected-final diagnostics, and same-machine
performance evidence belong to the separate qualification packet. This package
does not claim that extraction alone qualifies the model for publication.

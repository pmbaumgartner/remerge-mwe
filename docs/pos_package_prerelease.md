# remerge-pos 0.1.0a1

This is the first experimental public prerelease of the standalone
`remerge-pos` English UPOS tagger. It accepts caller-tokenized, NFC-normalized
English sentences and preserves token alignment. It does not tokenize raw text,
return confidence values, download models, identify languages, or integrate
with REMERGE.

## Assets and checksums

The code and model are deliberately separate distributions.

| Asset | Bytes | SHA-256 | License |
| --- | ---: | --- | --- |
| `remerge_pos-0.1.0a1-py3-none-any.whl` | 11,889 | `1a78f798f2f3fad0beadff699aa01b6362f5a8cc7e0ba0b7355701a5154de248` | MIT |
| `remerge_pos-0.1.0a1.tar.gz` | 9,851 | `f60c6ca9c1c3a1a19c05106c498ca5dcd16661e8809017fe379847d484701fcc` | MIT |
| `remerge-pos-en-ewt-0.1.0a1.tar.gz` | 1,525,432 | `fa37f77004cf0c4218c2e8999240ddc0fc21b4a05f266b75285c1f71525c583b` | CC BY-SA 4.0 |

Verify each downloaded file against this table before installation or
extraction. The model archive contains `LICENSES/CC-BY-SA-4.0.txt`,
`ATTRIBUTION.md`, `MODEL_CARD.md`, and `MANIFEST.json` beside `model.rmsp`.
Its manifest pins the upstream EWT revision, acquisition record, source-file
hashes, training command and revision, dependency lock, configuration, seed,
and qualification evidence.

The model payload itself is the exact qualified artifact: 1,515,159 bytes with
SHA-256
`a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d`.
The archive contains no corpus text, gold labels, Python pickle, or executable
payload.

## Installation and use

Python 3.12 or newer is required. Download the wheel and model archive locally,
verify both hashes, then install and extract them explicitly:

```sh
python -m pip install ./remerge_pos-0.1.0a1-py3-none-any.whl
tar -xzf remerge-pos-en-ewt-0.1.0a1.tar.gz
```

```python
from remerge_pos import Tagger

tagger = Tagger.load("remerge-pos-en-ewt-0.1.0a1/model.rmsp")
tags = tagger.tag((("The", "watch", "stopped", "."),))
```

Importing or constructing `remerge-pos` never accesses the network. The model
must always be loaded from an explicit caller-supplied local path.

## Qualification and limitations

The candidate passed the Q1 API alignment, artifact safety, determinism,
distribution, extraction parity, performance, reproducible-export, and root
compatibility gates. The release-only build then produced two byte-identical
copies of every asset, verified the exact wheel and sdist module bytes against
Q1, installed both distributions in clean environments, exercised the
qualified model, and denied network access during import.

The reported protected-final EWT diagnostics were 94.2625% overall accuracy,
0.9236 macro-F1, 78.7947% OOV accuracy, and 94.6775% ambiguous-token accuracy.
These results cover caller-tokenized English web text from UD English EWT; they
are not a broad-English claim. MWE behavior was not a package gate.

- Qualified runtime source: `47ab7cb3060d0319d71e02ed4be0afcca6f2d95a`
- Release candidate source: `53ffffb78813a00d7a58a668a821334b67bd0f2a`
- [Qualification record](https://github.com/pmbaumgartner/remerge-mwe/blob/remerge-pos-v0.1.0a1/docs/pos_package_qualification.md)
- [Exposure decision](https://github.com/pmbaumgartner/remerge-mwe/blob/remerge-pos-v0.1.0a1/docs/pos_package_exposure_decision.md)

## Licensing and withdrawal

The wheel and source distribution contain MIT-licensed code only. The separate
EWT-derived model distribution takes the conservative CC BY-SA 4.0 posture and
contains its complete terms and attribution. This separation does not assert
that every trained model is necessarily adapted material; it avoids depending
on the more permissive legal interpretation.

This prerelease may be withdrawn if a security, licensing, provenance, or
artifact-integrity defect appears. Before withdrawal, its exact assets,
checksums, notices, and evidence will be archived. Withdrawal cannot revoke
rights already granted under CC BY-SA 4.0. No stable release, PyPI publication,
automatic downloader, extra corpus, or REMERGE adapter is authorized here.

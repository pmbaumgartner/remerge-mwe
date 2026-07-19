# Linear POS tagger development evaluation

Status: production route rejected; experimental implementation retained.

Date: 2026-07-19

## Decision

The quantized candidate-pruned linear tagger does not qualify for release. Its
best development-set point accuracy is 93.77%, below the frozen 95% lower-bound
quality gate in `docs/pos_quality_contract.md`. The protected final split was
not evaluated.

Per the pre-authorized outcome fallback, v1 reshapes to the pretagged-only POS
filtering path. The native model loader and deterministic trainer remain
experimental, are not exposed through the public API, and must not be packaged
with a model or described as production-ready.

## Evidence

Both candidates used only the pinned UD English EWT r2.10 train and development
splits from `tests/pos/evaluation-manifest.json`. The seed was `20260719`, the
feature table contained 262,144 buckets, and the model artifacts were not added
to the repository.

| Candidate | Epochs | Full precision | Quantized | Quantized + candidate pruning | Artifact SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| c1 | 5 | 93.6658% | 93.6498% | 93.6339% | `98235a1e21ff4c12dcebdb0d1d33136b4dc0cd2dbf9ce4cf8f5006158072c189` |
| c2 | 8 | 92.2144% | 92.1866% | 93.7691% | `2d9e809a998e108b1f1a163e808e262fb9c5feef1c17aff0e4e4b537bbe3a0e4` |

Candidate c2 trained and inferred within the same ambiguous-form candidate
masks. This was the best bounded corrective attempt. Its quantization loss was
0.0278 percentage points; candidate pruning improved its accuracy by 1.5826
percentage points. The calibrated direct lexicon accepted 20.21% of development
tokens at 99.016% accuracy with a minimum training support of eight.

The development file SHA-256 was
`ef962ac05d844eaff46eeded125937129bfc0876d43963d66810cb73ffa8f5df` for
both candidates. Their committed trainer commands are present in their report
schema and can be reconstructed from the pinned inputs, but the temporary
reports and artifacts are deliberately not release inputs.

## Consequences

- Do not spend or disclose the protected final split on this rejected route.
- Do not weaken the quality contract or substitute development point accuracy
  for its bootstrap lower bound.
- Close or reshape downstream model packaging, public tagger integration, and
  model-release packets; they are not prerequisites for the pretagged fallback.
- Retain the experimental implementation only as reproducible research support.
  A future model attempt requires a new bounded packet and must pass the same
  frozen oracle before any public or packaging work resumes.

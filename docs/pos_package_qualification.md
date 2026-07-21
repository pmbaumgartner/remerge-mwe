---
kata: 88xq
created: 2026-07-20
---

# Standalone POS package qualification

## Outcome

The frozen `remerge-pos==0.1.0` averaged structured perceptron candidate passed
all Q1 engineering and packaging gates. Its one authorized EWT protected-final
run was completed and retained as a diagnostic. Q1 intentionally defines no
post-hoc POS quality floor, so final quality did not alter the pass result.

This qualifies the package for the next Human decision. It does **not** publish
the model, expose it through REMERGE, broaden the claim beyond English web text,
or authorize another corpus. Those choices remain with `cp7t` and `hrxv`.

## Frozen candidate

- Source revision: `47ab7cb3060d0319d71e02ed4be0afcca6f2d95a`
- Package: `remerge-pos==0.1.0`
- Model: `remerge-pos-structured-perceptron-v1`
- Recipe: 6 epochs, feature cutoff 1, 131,072 feature buckets, seed 20260720
- Artifact SHA-256:
  `a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d`
- Artifact size: 1,515,159 bytes
- Training data: pinned UD English EWT r2.10 train split only
- Model posture: separate CC BY-SA 4.0 artifact; MIT code distributions contain
  neither model weights nor corpus data

The selected recipe was run by an isolated Python interpreter from the built
wheel, not from the workspace import path. It reproduced the frozen artifact
byte-for-byte and reproduced the complete 24,822-token development prediction
tensor and frozen development metrics.

## Hard-gate result

| Gate | Result | Evidence |
| --- | --- | --- |
| Extraction parity | Pass | Exact artifact bytes, prediction tensor, token count, and frozen metrics |
| Determinism | Pass | Three warmups and 15 identical measured inference repetitions |
| API and alignment | Pass | Invalid nesting/forms/language plus dropped, inserted, reordered, invalid-tag, and rotated-output controls rejected |
| Artifact safety | Pass | 13 malformed or hostile artifact controls rejected; unsafe source operations absent |
| Reproducible export | Pass | Installed-wheel rebuild matched the registered config, seed, training hash, and artifact digest |
| Distribution | Pass | Clean wheel and sdist installs, network-denied imports, explicit-model smoke tests, file allowlists, license presence, and corpus-leak controls passed |
| Performance | Pass | 9,428.34 dev tokens/s versus a 7,300.18 minimum; 176.7 ms median cold load; 69,402,624-byte maximum incremental peak RSS |
| Root compatibility | Pass | Package tests, 143-test non-performance suite, and all-file `prek` checks passed |

The retained code distributions are MIT-only and contain eight files each. The
wheel SHA-256 is
`d15afc0c0f231e5448cdc40f521fcc5ee2b665ed1e545010e63a8c95c8ba5284`;
the sdist SHA-256 is
`764dfde1c66b40a8d3a639565db539fd9bddc732be7e50d36e3c43e2ae426e23`.

## Quality diagnostics

| Split/model | Overall accuracy | Macro F1 | OOV accuracy | Ambiguous accuracy |
| --- | ---: | ---: | ---: | ---: |
| EWT development, candidate | 94.1342% | 90.9330% | 78.4265% | 93.9427% |
| EWT protected final, candidate | 94.2625% | 92.3572% | 78.7947% | 94.6775% |
| EWT protected final, c2 | 93.6358% | 89.3951% | 75.9855% | 94.2862% |
| Candidate minus c2 | +0.6267 pp | +2.9620 pp | +2.8092 pp | +0.3914 pp |

On the 24,575 protected-final tokens, both systems were correct on 22,652;
only the candidate was correct on 513; only c2 was correct on 359; and both
were wrong on 1,051. Per-domain and per-UPOS candidate slices are retained in
`docs/evidence/pos-package-qualification-q1.json`.

These measurements are EWT English-web-domain evidence, not an unrestricted
English claim. MWE behavior was not evaluated and is not a standalone-package
gate.

## Protected-final controls and retained evidence

Development registration passed before any protected read. The one-shot marker
was then created before final access, and the harness durably recorded an
in-progress report before invoking the final loader. The registration cannot be
consumed again.

- Development report SHA-256:
  `7c944ca4958b2ba9b0939c07d8fd7928ff8736a96fa964f80468d3f7e2539104`
- Registration SHA-256:
  `f7fb9d1f86af17c9d44909e487298cee2c6f767b0183dac6067ff5afac6d5368`
- Consumption-marker SHA-256:
  `aa2d60da20da85313c9acc1ba13a87fa6db29f6279240ca7df90c87c17342f2f`
- Full final report SHA-256:
  `e0365d308455ca86216dc0980ed30bab03c47d9822d28d0b74809361f52a5604`

Full reports, registration, marker, wheels, sdist, model card, and model manifest
remain outside the checkout. The committed evidence contains only approved
aggregates and hashes—no raw EWT text, gold labels, prediction tensors, or model
weights.

## Qualification history

The first development-only attempt on source revision `5488b9a` stopped before
registration or final access because a one-repetition parity smoke exposed a
singleton-IQR bug in the oracle. Revision `47ab7cb` added the explicit
one-sample rule and regression coverage without changing candidate behavior,
data, thresholds, or claims. A fresh development registration then passed and
was the only registration consumed.

## Verification

Commands run on the frozen candidate revision included:

```text
uv run --no-sync pytest -q tests/pos/test_package_qualification.py packages/remerge-pos/tests
uv run --no-sync pytest -q -m "not performance"
uv run --no-sync prek run --all-files
```

Results: 30 focused/package tests passed; 143 non-performance tests passed with
2 performance tests deselected; every `prek` hook passed.

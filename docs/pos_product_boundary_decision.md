---
kata: c1sh
created: 2026-07-20
status: awaiting-human-decision
---

# POS product-boundary decision

This is the final Human gate for outcome `v4a2`. It does not revisit P1+D1,
weaken a quality threshold, inspect protected final, or authorize a third
pilot. It decides what product/package boundary follows the completed pilot
portfolio.

## Portfolio evidence

The candidate-neutral harness, compact-tagger survey, and gold-data
sufficiency study completed first. Peter then selected P1+D1: sequential TnT
and averaged structured-perceptron pilots, with augmentation provenance
deferred.

Both authorized pilots completed on pinned train/development inputs:

| Candidate | Selected development result | Decision |
| --- | --- | --- |
| TnT | 91.217% overall, 61.999% OOV, 9.38% MWE precision, 26.09% recall, 7,644 tok/s | Reject |
| Structured perceptron | 94.134% overall, 78.427% OOV, 8.82% MWE precision, 26.09% recall, 8,111 tok/s | Reject |

The perceptron improved c2 by 0.342 overall points, 2.532 macro-F1 points,
0.891 OOV points, and 0.186 ambiguous-token points. It still missed the
unchanged 95% overall and 82% OOV floors, did not improve downstream MWE
utility, and missed the 50,000-token/s floor. All ten perceptron artifacts
rejected. Protected final remains unobserved, and no candidate is registered
for qualification.

## Boundary options

- **B0 — Stop at experimental evidence (recommended).** Retire both artifacts
  as attributable negative evidence. Create no qualification, integration,
  exposure, packaging, or release work.
- **B1 — Internal REMERGE module.** This cannot promote either rejected
  artifact. It would require a separately authorized architecture/data cycle
  before a candidate could return to this gate.
- **B2 — Separate workspace package.** There is no eligible capability to
  package, so this adds a boundary without a qualifying product.
- **B3 — Independent project plus REMERGE adapter.** There is likewise no
  eligible capability; this also triggers a Core-to-Extended delivery-profile
  review.

## Recommendation

Select **B0**. It follows the root stop conditions: two complementary
sequence-aware architectures produced no credible downstream MWE improvement
and neither satisfied the frozen qualification prerequisites. D1 also means
there is no authorized data/teacher branch to continue. Stopping closes the
outcome honestly without observing final or preserving a non-qualifying
integration surface.

## Human decision record

Awaiting Peter's selection. The decision must state whether any protected
qualification, integration, exposure, packaging, or release work is
authorized. Under B0, all are explicitly unauthorized.

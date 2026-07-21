---
kata: c1sh
created: 2026-07-20
status: decided-b2
---

# POS product-boundary decision

This Human gate reshapes outcome `v4a2` around a separate workspace package.
It does not revisit P1+D1, silently weaken the existing REMERGE quality
contract, inspect protected final, or authorize new training data. It decides
what product/package boundary follows the completed pilot portfolio.

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

- **B0 — Stop at experimental evidence.** Retire both artifacts
  as attributable negative evidence. Create no qualification, integration,
  exposure, packaging, or release work.
- **B1 — Internal REMERGE module.** This cannot promote either rejected
  artifact. It would require a separately authorized architecture/data cycle
  before a candidate could return to this gate.
- **B2 — Separate workspace package.** Treat generic POS tagging as an
  independently useful capability, with its own package boundary and
  qualification contract.
- **B3 — Independent project plus REMERGE adapter.** There is likewise no
  eligible capability; this also triggers a Core-to-Extended delivery-profile
  review.

## Human decision record

Peter selected **B2** on 2026-07-20 and selected the averaged structured
perceptron as the initial architecture. His rationale is:

- The perceptron's result is very close to the 95% benchmark, and 95% is too
  arbitrary to decide whether an independently useful generic POS tagger
  deserves a package.
- A generic POS tagger is useful independently of MWE discovery. Isolating it
  in a workspace package gives it a stable place for continued improvement.
- MWE performance is a separate concern, and the current MWE evaluation is
  too small and too dependent on one dataset to serve as the standalone POS
  package's acceptance oracle.

This decision authorizes a Core delivery phase that defines a new standalone
POS acceptance contract, extracts the averaged-perceptron implementation into
a separately installable workspace package, and qualifies that package against
the new contract. The old 95% overall, 82% OOV, downstream MWE, and 50,000
token/s gates remain historical and authoritative for the stopped built-in
REMERGE route until a Human explicitly changes that route; B2 does not waive
them.

The decision does **not** authorize observing protected final, adding a corpus,
teacher, pseudo-label, or augmentation branch, making the package a mandatory
REMERGE dependency, automatically downloading a model, publishing a stable or
prerelease artifact, or exposing generated tags through REMERGE. Those actions
remain behind explicit provenance, qualification, and Human exposure gates.

The delivery profile remains **Core**: one repository, one separately
installable workspace package, and one accountable maintainer. Moving the
package to an independent repository or adding broader organizational release
coordination requires a fresh profile review.

## Follow-up ledger

The B2 work is tracked by child issues of `v4a2`, in dependency order:

1. `hzrh` — freeze the standalone package contract and qualification gates.
2. `4gt3` — add the averaged-perceptron workspace package.
3. `88xq` — add and run the standalone qualification harness.
4. `cp7t` — hold a Human gate for prerelease and optional REMERGE exposure.
5. `pnd7` — add an optional REMERGE adapter only if that gate authorizes it.
6. `dr0g` — publish a prerelease only if that gate authorizes it.

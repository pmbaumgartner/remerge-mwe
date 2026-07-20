---
kata: brx7
created: 2026-07-19
status: decided
---

# POS pilot portfolio and data-strategy decision

## Final record

**Finding:** Gold-data volume, domain diversity, and low-support errors all
matter, but c2's downstream MWE utility plateaus. The best next information
comes from two complementary sequence-aware models, not more data machinery.

**Decision:** On 2026-07-20, Peter selected **P1 + D1**: run a TnT-style
trigram HMM and an averaged structured perceptron sequentially, and defer any
augmentation-provenance gate.

**Recommendation:** Execute the two gold-only pilots with work in progress
limited to one. Reassess architecture and product boundary after both reports;
do not create CRF, teacher, corpus, or pseudo-label work under this decision.

**Follow-up issues:** `bkh6` runs the TnT-style pilot first. `1wda` runs the
structured-perceptron pilot and is blocked by `bkh6`. No augmentation issue was
created under D1.

**Confidence:** Medium. The evidence strongly motivates sequence-aware pilots,
but the small MWE development denominator and incomplete matched diversity
repeats leave uncertainty about which mechanism will matter most.

**Rejected alternatives:** P2's CRF ceiling and D2's provenance analysis were
deferred as premature cost and coordination. P3/D3 would leave credible
sequence and data hypotheses untested or irreversibly foreclose later evidence.

Nothing in this decision approves a production model, an external dependency,
a corpus, a teacher, pseudo-label generation, or use of the protected final
split.

## Evidence consumed

- The retained c2 model reproduced at 93.7691% on raw development data. Under
  the common retained-development harness it remained below the protected
  quality, MWE utility, and throughput analogues.
- The completed 80-condition gold study reached 93.830% mean accuracy at full
  retained training data. The paired 100%-minus-75% gain was +0.452 points
  (95% t interval +0.344 to +0.560).
- Full-data OOV accuracy was 77.338%; OOV error exceeded overall error by
  16.492 points. Matched diversity controls showed positive held-out-domain
  gains for email, reviews, and weblog, while answers and newsgroup remained
  wide or incomplete.
- c2 downstream MWE precision and recall stopped improving materially after
  50% training. Full-data generated-filter precision was 9.04% versus 0.42%
  unfiltered, and recall was 26.09% on the small 23-span development subset.
- The formal data diagnosis is `inconclusive`: volume, diversity, and
  low-support signals coexist. The evidence rejects a simple architecture-only
  plateau but does not isolate one primary data limitation.
- The implementation survey identifies TnT-style trigram HMM, averaged
  structured perceptron, and linear-chain CRF as complementary gold-only,
  first-party classical pilots.

## Pilot portfolio options

### P1. Two-pilot sequence test — recommended

Select a TnT-style trigram HMM and an averaged structured perceptron. Run one
at a time. TnT is the smallest direct test of suffix emissions and tag
transitions; the perceptron is the smallest discriminative sequence successor
to c2. Defer CRF unless a later human decision finds that these results leave a
specific capacity question unanswered.

This has the best information-to-cost ratio and tests both observed hard
slices without immediately taking on CRF optimization and serialization.

### P2. Three-pilot classical ceiling

Select TnT, structured perceptron, and linear-chain CRF, still with work in
progress limited to one pilot. CRF runs last and only after both smaller
reports exist. This gives the strongest classical capacity comparison but has
the highest implementation and tuning cost.

### P3. Stop after evidence

Select no pilot. Retain the harness, survey, and data study as research
evidence and keep the product pretagged-only. This avoids further cost but
leaves the observed sequence, OOV, and domain-diversity hypotheses untested.

## Augmentation-provenance options

### D1. Defer provenance gate — recommended

Do not open an augmentation gate yet. Reconsider only if a gold-only sequence
pilot approaches the protected quality/utility analogues while its residual
errors remain demonstrably OOV/low-support limited. The present c2 downstream
plateau makes additional labels premature.

### D2. Authorize a provenance gate only

Open a separate consequential decision that must name and license-review one
exact unlabeled corpus, teacher set, output-retention policy, token-alignment
method, maximum pseudo-token budget, and redistribution posture. Conventional
ensemble agreement must be the first comparator. This authorizes analysis
only: it does not authorize label generation or a teacher/API call.

### D3. Reject augmentation for this outcome

Record that pseudo-labeling and teacher data will not be considered anywhere
under `v4a2`, even if a pilot later exposes a data barrier. Reopening would
require a new outcome-level human decision.

## Bounds for any selected pilot

- Maximum initial pilot count is the selected portfolio's size; work in
  progress is one.
- Gold train/development data only; no final data, teacher output, new corpus,
  network model call, or production integration.
- First-party prototype with no new runtime dependency or external spend.
- One predeclared bounded tuning grid and five fixed evidence seeds per pilot.
- The common harness, protected thresholds, canonical alignment, and resource
  measurements are unchanged. The 20 MiB artifact ceiling is a hard stop.
- Stop a pilot on alignment/provenance failure, resource-ceiling failure, no
  credible hard-slice or MWE improvement over c2, or any need to weaken a
  protected gate.
- Do not inspect the protected final split. Registration for a one-shot final
  evaluation requires another human selection after dev evidence.

## Recommendation and strongest alternative

Choose **P1 + D1**: two gold-only sequence pilots, one at a time, with
augmentation provenance deferred. The strongest rejected alternative is
**P2 + D2**, which buys a fuller capacity ceiling and earlier data-lineage
work at materially greater implementation, licensing, and coordination cost.

Evidence that would change the recommendation is: (1) TnT and perceptron leave
the same reproducible sequence-context errors with resource headroom, which
would justify CRF; or (2) a gold-only sequence pilot approaches downstream and
quality gates while a positive learning-curve marginal and concentrated OOV
errors persist, which would justify opening the provenance gate.

## Human record

- Pilot portfolio: **P1 — TnT-style trigram HMM and averaged structured perceptron**
- Augmentation provenance: **D1 — deferred**
- Decision owner: **Peter**
- Recorded: **2026-07-20, explicit task response `P1 + d1`**
- Rationale or amendments: **Accepted the recommendation without amendment**

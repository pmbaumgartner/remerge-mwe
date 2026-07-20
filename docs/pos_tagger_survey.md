---
kata: q3t1
created: 2026-07-19
---

# Compact POS tagger and teacher/student survey

## Purpose and non-decision

This implementation-oriented survey recommends a small bakeoff portfolio for
the optional English UPOS capability in `v4a2`. It does **not** select a
production model, approve a teacher or data, authorize pseudo-labels, change
the protected oracle, or revive the retired generated-tagger route.

The retained independent linear experiment reached 93.7691% development point
accuracy and was rejected before final evaluation. A new candidate must keep
the exact caller-provided form and sentence nesting, emit one of 17 UPOS tags,
and never retokenize or repair inputs. Those constraints, the quality gates,
and the one-shot final policy in [the annotation contract](pos_annotation_contract.md)
and [quality contract](pos_quality_contract.md) are authoritative.

## Reproducible search scope

Search date: **2026-07-19** (America/New_York). It covered
pre-CNN/pre-transformer supervised POS families (HMM/TnT, maximum entropy,
averaged/structured perceptron,
linear-chain CRF, transformation rules, cyclic context, sequential SVM,
memory-based tagging, dictionaries, and OOV suffix/morphology) plus offline
teacher/student methods (self-training, ensembles, hard sequences, soft
distributions, disagreement filters, rule extraction, and sequence
distillation). Later neural/transformer systems were considered only as
potential offline teachers or quality ceilings.

Starting sources were ACL Anthology, author/university publication pages, and
official implementation/license pages. Exact searches:

```text
"TnT - A Statistical Part-of-Speech Tagger" Brants
"A Maximum Entropy Part-Of-Speech Tagger" Ratnaparkhi
"Discriminative Training Methods for Hidden Markov Models" Collins
"Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"
"A Simple Rule-Based Part of Speech Tagger" Brill
"Feature-Rich Part-of-Speech Tagging with a Cyclic Dependency Network"
"SVMTool" Giménez Márquez POS
"MBT: A Memory-Based Part of Speech Tagger-Generator"
"Semi-supervised Training for the Averaged Perceptron POS Tagger"
"Simple semi-supervised training of part-of-speech taggers"
"Distilling the Knowledge in a Neural Network"
"Sequence-Level Knowledge Distillation"
```

Technical claims below cite the primary paper (or an author-hosted primary
paper); implementation/license claims cite the implementation owner. Search
stopped once each family had an authoritative description, a bounded
falsification experiment, and a disposition. It excludes unsourced
leaderboards, unapproved model implementations, and production work.

### Accuracy is not comparable across papers

Published headline accuracy cannot rank candidates: corpora, tokenization,
tagsets (often Penn tags rather than UPOS), splits, OOV definitions, training
data, and hardware differ. Thus TnT's Penn Treebank/NEGRA results, Brill's
Brown-Corpus result, and Søgaard's WSJ result are non-comparable to each other
and to the pinned EWT development evidence. Only REMERGE's common harness,
fixed token contract, pinned train/dev data, and protected-final procedure can
rank models.

## Common bakeoff contract

Every serious candidate uses only approved gold train/dev data; the same UPOS
mapping, normalization, seed set, and sentence-token inputs; and the common
dev analogue of the quality/resource report. Record overall, OOV,
ambiguous-form, per-UPOS, and domain-slice metrics; boundary mismatches
separately (and reject them); artifact bytes, cold load, RSS, inference and
end-to-end throughput, and 256-token p95; and downstream MWE utility without
reading final data. A final run remains a separately registered one-shot
evaluation. EWT, model distribution, and teacher-data terms remain controlled
by [the provenance decision](pos_model_provenance.md): no teacher output or
additional unlabeled source is currently approved.

## Supervised candidate scorecards

The following are serious because each tests a distinct compact mechanism.
Footprint predictions are mechanisms to measure, never REMERGE measurements.

### 1. TnT-style trigram HMM — pilot

**Mechanism/source.** TnT is a second-order HMM decoded over a sentence, with
interpolation smoothing and unknown-word suffix handling ([Brants
2000](https://aclanthology.org/A00-1031/)).

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Suffix emissions directly target unseen forms; trigram transitions resolve local ambiguity. It cannot use arbitrary lexical conjunctions. |
| Artifact / runtime | Counts, lexicon, suffix tables, and a small transition table; fixed-17-tag Viterbi is linear in sentence length. Expected compactness must be measured. |
| Training complexity | Deterministic counts, smoothing/suffix tables, and Viterbi; no external runtime needed. |
| Implementation availability / license | No implementation is selected. Build a first-party prototype; the cited paper is not a software license and must not be treated as one. |
| Licensing / provenance | The paper does not license an implementation. A first-party implementation avoids runtime dependency; EWT/model rules still apply. |
| Integration fit | Strong: consume canonical nested forms and return exactly aligned tags. Raw-text timing must use the frozen tokenizer profile. |
| Bounded falsifier | Gold train only; predeclared small interpolation/suffix grid on dev. Stop if it misses dev gates or gives no material OOV/ambiguity gain over the retained linear baseline while retaining resources. |

### 2. Averaged structured perceptron — pilot

**Mechanism/source.** Collins trains a complete-label-sequence scorer with
perceptron updates and sequence decoding ([Collins
2002](https://aclanthology.org/W02-1001/)). It is the smallest discriminative
sequence counterpart to the retained independent linear model.

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Reuses word/shape/suffix features for OOVs and adds adjacent-tag features for ambiguity. The retained token-local baseline makes that value testable. |
| Artifact / runtime | Sparse/hash weights plus 17x17 transitions; Viterbi adds predictable sentence-local work. Quantization is plausible but unproven. |
| Training complexity | Averaged multi-pass updates, deterministic feature order/hashing, and Viterbi/beam decode; more sensitive than HMM counts but lighter than CRF optimization. |
| Implementation availability / license | No implementation is selected. Build a first-party prototype; Collins's paper is algorithm evidence, not a software license. |
| Licensing / provenance | Algorithm paper, no required dependency; native gold-only implementation avoids teacher provenance. |
| Integration fit | Strong if all features derive from supplied forms. It must not expose confidence or affect the unfiltered path. |
| Bounded falsifier | One template family: retained lexical/shape/suffix features plus tag bigrams; fixed epochs/seeds, dev only. Stop if gains do not justify decode cost or any resource bound fails. |

### 3. Linear-chain CRF — pilot quality ceiling

**Mechanism/source.** A linear-chain CRF is a conditionally trained sequence
model that supports overlapping input features while normalizing over label
sequences ([Lafferty, McCallum & Pereira
2001](https://people.cs.umass.edu/~mccallum/publications-by-topic.html)).

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Can combine suffix/shape/lexical features with globally normalized adjacent-tag decisions; tests whether structured conditional capacity closes the remaining gap. |
| Artifact / runtime | Sparse weights/transitions and Viterbi; feature maps and optimizer choice can exceed perceptron size. Int8/pruning are prospects, not assumptions. |
| Training complexity | Regularized forward-backward optimization, feature cutoff, deterministic convergence, serialization; heaviest classical pilot. |
| Implementation availability / license | No implementation is selected. Start with an isolated first-party prototype or a separately license-reviewed evaluator; the paper is not a software license. |
| Licensing / provenance | Paper is not a library license. Use isolated license-reviewed or first-party tooling; do not add a production dependency for research. |
| Integration fit | Exact-boundary fit is good; resource risk is higher than TnT/perceptron. |
| Bounded falsifier | Fixed regularization/cutoff grid and maximum iterations on dev. Reject if it does not materially beat perceptron on hard slices or violates size/load/throughput. |

### 4. Transformation-based residual correction — defer

**Mechanism/source.** Brill learns ordered, context-sensitive substitutions
over a lexical initial tagger and describes capitalization/suffix treatment of
unseen words ([Brill 1992](https://aclanthology.org/A92-1021/)).

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Explicit local rules can correct recurring ambiguity and suffix/case rules can help OOVs; they do not optimize a sequence globally. |
| Artifact / runtime | Compact lexicon plus ordered rules, inexpensive inference. |
| Training complexity | Rule-template/candidate search, held-out induction, order, and stop criteria add tuning surface. |
| Licensing / provenance | Paper only; first-party learner still needs approved gold data. |
| Integration fit | Good mechanically, but never weakens exact-boundary/UPOS requirements. |
| Bounded falsifier | The pre-existing one-cycle residual-rule policy is the sole allowed experiment; defer on no material dev/downstream gain within resource limits. |

`wsjt` already deferred this route: a rejected base model does not justify
additional residual optimization. Reconsider only under a new outcome with a
qualified base candidate.

### 5. Classic maximum entropy and cyclic context — defer

**Mechanism/source.** Ratnaparkhi's classic maximum-entropy POS tagger is the
in-scope conditional log-linear baseline ([Ratnaparkhi
1996](https://aclanthology.org/W96-0213/)). Toutanova et al. extend that
feature-rich direction with preceding and following tag context through a
cyclic dependency network ([Toutanova et al.
2003](https://aclanthology.org/N03-1033/)). Published WSJ results are
explicitly non-comparable here.

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Fine-grained morphology/lexical features and bidirectional context target both slices. |
| Artifact / runtime | Feature tables can grow; iterative/cyclic prediction is less predictable than the selected decoders. |
| Training complexity | Regularized log-linear fitting, cyclic inference, and feature engineering. |
| Implementation availability / license | No MaxEnt or cyclic implementation is selected. Both papers are research evidence, not a software license; no external package is authorized. |
| Licensing / provenance | Research papers only; this creates no approved dependency or teacher data. |
| Integration fit | Possible, but convergence/determinism risk is separate. |
| Bounded falsifier | Reconsider only if selected pilots identify a reproducible right-context error class with resource headroom. |

CRF already tests feature-rich conditional sequence tagging with a clearer
inference contract, so cyclic context is not a first-pass pilot.

### 6. Sequential SVM / SVMTool — reject for this bakeoff

**Mechanism/source.** SVMTool is a configurable SVM-based POS generator; its
official project page records its method and LGPL license
([SVMTool](https://www.cs.upc.edu/~nlp/SVMTool/)).

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Credible morphological/context features, but overlaps strongly with perceptron/CRF questions. |
| Artifact / runtime | Support vectors and external tooling make footprint/deployment less predictable than compact native weights. |
| Training complexity | Kernel/features, sequential strategy, and external toolchain. |
| Licensing / provenance | LGPL requires a dependency/distribution review outside scope; no wrapper is authorized. |
| Integration fit | Adapter possible but poor optional-first-party fit. |
| Bounded falsifier | Do not run. Reopen only with a license-reviewed reproducible linear implementation showing a distinct same-harness advantage. |

### 7. Memory-based tagging and tag dictionaries — reject as primary model

**Mechanism/source.** MBT predicts from stored similar contextual cases; its
authors identify incremental learning and unknown-word behavior as benefits
([Daelemans et al. 1996](https://aclanthology.org/W96-0102/)). A tag dictionary
remains a useful shared lexical diagnostic.

| Criterion | Assessment |
| --- | --- |
| OOV / ambiguity benefit | Stored cases help seen ambiguity; OOV quality relies on morphology/generalization rather than storage. |
| Artifact / runtime | A full case base is data-size-sensitive and nearest-neighbor lookup is less predictable; dictionary alone is small but insufficient. |
| Training complexity | Storage is simple but distance/index/compression choices become the effective model. |
| Licensing / provenance | Paper only; case retention increases the need to exclude corpus text from distributable artifacts. |
| Integration fit | Boundary-safe, but weak against the 20 MiB/provenance constraints if broad cases ship. |
| Bounded falsifier | Do not build a primary pilot. Keep only compact train-derived dictionary/suffix diagnostics; revisit only if a compressed case model uniquely improves hard slices. |

## Teacher/student and semi-supervised scorecards

Teachers are offline training experiments only after a source-specific decision
names the exact teacher artifact, terms, input corpus, output retention rules,
and model/logit rights. This is a pause trigger: no current teacher output,
soft distribution, pseudo-label, or unlabeled corpus is approved.

### A. Gold-only baseline — required control

Every selected student is trained on pinned approved gold data. It has no
teacher risk and is the denominator for every augmentation claim. An augmented
result that does not beat this identical dev protocol is rejected.

### B. Ensemble self-training with disagreement filtering — strongest credible conventional alternative to LLM labeling

Train diverse gold-only models (for example HMM, perceptron, CRF); label a
separately approved corpus only where two teachers agree; train one compact
student on those **hard sequence labels**. Søgaard directly combines
tri-training and disagreement-based co-training for POS and notes the mixed
history of semi-supervised tagging ([Søgaard
2010](https://aclanthology.org/P10-2038/)). Spoustová et al. describe
semi-supervised averaged-perceptron training with an ensemble
([Spoustová et al. 2009](https://aclanthology.org/E09-1087/)).

| Criterion | Assessment |
| --- | --- |
| Unique mechanism / benefit | Agreement filters label noise without an LLM; disagreement is an abstention/audit signal. Benefit is corpus-dependent and unassumed. |
| Student artifact / runtime | One compact selected student ships; teachers and labels stay offline. |
| Training complexity | Diverse teachers, split isolation, dev-only agreement/volume selection, full pseudo-label lineage, and leakage prevention. |
| Licensing / provenance | Requires approval of the unlabeled source and model-output use; approved EWT does not approve either. |
| Integration fit | Strong after training if every teacher uses canonical boundaries. |
| Bounded falsifier | One approved corpus, fixed teachers/agreement rule/max pseudo-token budget, one student. Reject if gold dev OOV/ambiguity/MWE utility do not beat gold-only or lineage/resources fail. |

This is the required non-LLM comparator. Any LLM-labeling proposal must beat it
as well as gold-only under identical lineage and dev reporting.

### C. Offline teacher-to-student distillation — defer pending provenance

Hinton, Vinyals, and Dean describe compressing ensemble knowledge into a
smaller model via teacher distributions ([Hinton et al.
2015](https://arxiv.org/abs/1503.02531)); Kim and Rush describe sequence-level
targets ([Kim & Rush 2016](https://aclanthology.org/D16-1139/)).

| Criterion | Assessment |
| --- | --- |
| Unique mechanism / benefit | A stronger teacher can provide hard decoded sequences, soft token distributions, or sequence targets that retain ambiguity information. |
| Student artifact / runtime | Same compact student; teacher, logits, and pseudo-corpus never ship. |
| Training complexity | Teacher choice, temperature/loss mixing, sequence decoding, storage controls, and gold/teacher balance. |
| Licensing / provenance | Blocked: no exact teacher, weights/API terms, training-data restrictions, or logit retention right is approved. |
| Integration fit | Only if generated against canonical tokens; raw-text teacher tokenization may not be silently aligned. |
| Bounded falsifier | After approval, compare hard sequences and soft distributions on the same source/fixed student. Reject either if it loses to gold-only and conventional-ensemble controls. |

### D. Teacher rule extraction — reject for current portfolio

This combines unapproved teacher provenance with the deferred residual-rule
search and has no distinct evidence over gold rules plus a compact student.
Reconsider only if an approved teacher produces a stable held-out rule benefit
that every compact student misses.

## Recommendation: three complementary pilots

1. **TnT-style trigram HMM**: the smallest credible sequence/OOV-suffix
   baseline; tests whether conventional transitions move hard slices.
2. **Averaged structured perceptron**: isolates discriminative lexical and
   morphological features plus sequence decoding from the retained token-local
   experiment.
3. **Linear-chain CRF**: bounded higher-capacity classical ceiling; tests
   whether added training complexity buys a materially different result.

These span generative counts, discriminative margins, and conditional
probabilistic sequence modeling, share the native token contract, and require
no unapproved data. A later human gate must authorize any pilot. If only two
can run, use TnT and structured perceptron first; CRF is the
complexity-versus-quality discriminator.

### Evidence that changes this recommendation

- Replace TnT only if an estimate or valid dev run shows no plausible path to
  hard OOV/ambiguity floors; prefer a compact dictionary/suffix diagnostic,
  not a larger model by default.
- If perceptron matches CRF within predeclared practical tolerance on all dev
  slices and beats TnT, do not expand to SVM/cyclic models; prefer simplicity.
- If CRF materially improves hard slices and MWE utility within every resource
  bound, it is evidence for a later human production choice; otherwise no
  deferred classical family gets an automatic turn.
- After exact teacher/corpus approval, test conventional disagreement-filtered
  augmentation first. LLM labeling must beat that and gold-only controls.
- Revisit cyclic/SVM only with a reproducible unaddressed error class, reviewed
  license, and common-budget headroom.

## Deferred/rejected register

| Family | Disposition | New evidence required |
| --- | --- | --- |
| Brill/residual correction | Deferred | Qualified base candidate and a new bounded packet showing a downstream-relevant error pattern. |
| MaxEnt cyclic/bidirectional context | Deferred | Selected pilots miss a specific right-context class within resource headroom. |
| Sequential SVM/SVMTool | Rejected | License-reviewed reproducible implementation with distinct same-harness advantage. |
| Memory-based primary tagger | Rejected | Compressed cases beat dictionary/suffix features without corpus retention or size/runtime failure. |
| Deployed neural/transformer student | Deferred | Separate outcome and independent proof of all runtime/resource constraints. |
| Unspecified teacher, LLM labels, soft logits | Blocked | Exact artifact/terms/corpus/output lineage/use rights are approved before generation. |
| Teacher rule extraction | Rejected | Approved teacher yields a stable held-out benefit no compact student matches. |

## Source ledger

- [Brants, *TnT* (ANLP 2000)](https://aclanthology.org/A00-1031/)
- [Collins, structured perceptrons (EMNLP 2002)](https://aclanthology.org/W02-1001/)
- [Lafferty, McCallum, Pereira, CRFs (ICML 2001)](https://people.cs.umass.edu/~mccallum/publications-by-topic.html)
- [Brill, transformation-based tagging (ACL 1992)](https://aclanthology.org/A92-1021/)
- [Ratnaparkhi, maximum-entropy POS tagging (EMNLP 1996)](https://aclanthology.org/W96-0213/)
- [Toutanova et al., cyclic dependency network (HLT-NAACL 2003)](https://aclanthology.org/N03-1033/)
- [Official SVMTool project and LGPL notice](https://www.cs.upc.edu/~nlp/SVMTool/)
- [Daelemans et al., MBT (VLC 1996)](https://aclanthology.org/W96-0102/)
- [Spoustová et al., semi-supervised perceptron (EACL 2009)](https://aclanthology.org/E09-1087/)
- [Søgaard, semi-supervised POS tagging (ACL 2010)](https://aclanthology.org/P10-2038/)
- [Hinton, Vinyals, Dean, distillation (2015)](https://arxiv.org/abs/1503.02531)
- [Kim and Rush, sequence-level distillation (EMNLP 2016)](https://aclanthology.org/D16-1139/)

For REMERGE-specific data/model restrictions, the authoritative source remains
[`pos_model_provenance.md`](pos_model_provenance.md), including its linked UD
and Creative Commons licensing evidence.

---
kata: hrxv
status: decision-pending
owner: Peter
researched: 2026-07-20
---

# Additional POS corpus and model-license decision

## Decision to make

Choose whether `remerge-pos` should broaden its English corpus portfolio and,
if so, whether that work belongs in the generally usable model lane or in a
separate noncommercial research lane.

This packet does not change the released EWT-only `0.1.0a1` model. It is based
only on public metadata and license research. No new corpus text, labels,
training split, or protected evaluation split was downloaded or inspected.

## The short legal answer

“Creative Commons” is a family of licenses, not one permission. CC BY,
CC BY-SA, and CC BY-NC-SA have materially different conditions. A treebank may
also contain several separately governed layers: original text or speech,
annotations, a compiled database, conversion code, and access terms. A CC label
on an annotation repository does not prove that every underlying text, privacy
interest, contract, or database right is covered.

A trained model is not automatically an Adapted Material merely because its
training data used a CC license. The answer depends on applicable law, the acts
performed during acquisition and training, whether an exception applies,
whether the model stores or can reproduce protected expression or a substantial
database portion, and what is later shared. Creative Commons itself describes
that analysis as fact- and jurisdiction-dependent.

The project should nevertheless keep its existing conservative posture:

1. Treat a publicly shared model trained on ShareAlike data as if it were
   Adapted Material unless counsel or the rights holder approves a less
   restrictive posture.
2. Publish code separately under MIT and publish the model under the most
   restrictive compatible data license, with source-by-source attribution,
   change notices, immutable revisions, input hashes, and a model manifest.
3. Keep NonCommercial data out of the generally usable lane. If selected, it
   gets a separate model, package identity, evidence, and CC BY-NC-SA posture;
   it never silently replaces the non-NC model.
4. Test the classical artifact for memorization or recoverability even though
   the current `RMSP0001` format stores hashed numeric weights and no forms,
   sentences, or gold labels.
5. Treat unclear underlying-text rights, access contracts, and personal or
   child speech as blockers rather than inferring permission from the UD
   repository label.

This is a conservative operating policy, not a legal conclusion or legal
advice. The primary legal references are Creative Commons’
[AI-training primer](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/),
the [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en),
and UD’s own [treebank licensing guidance](https://universaldependencies.org/contributing/licensing.html).

## What the source audit changed

The supplied inventory correctly identified useful domains, but three examples
show why license labels alone are insufficient:

- **CHILDES:** UD English CHILDES says CC BY-SA 4.0 and includes text, but it is
  drawn from CHILDES transcripts. TalkBank’s ground rules say its data default
  to CC BY-NC-SA 3.0 and expressly say the data cannot be incorporated into
  commercial products, including models. CHILDES therefore cannot enter the
  general model lane without source-specific permission.
- **ESLSpok:** the UD treebank is CC BY-SA 4.0; the identified NICT JLE source
  independently permits use and redistribution under CC BY-SA 3.0. It has a
  substantially clearer chain, although participant/privacy, attribution,
  version-compatibility, and split-leakage checks still remain.
- **ATIS:** the UD repository says CC BY-SA 4.0, but its cited source passes
  through an unofficial GitHub copy, Kaggle, and Microsoft CNTK without an
  explicit source-data license in the cited README. ATIS is blocked pending a
  rights-holder or canonical-license record.

## Proposed source disposition

These are proposed defaults for the Human decision, not present authority to
acquire or inspect labels.

| Source | Proposed role | General lane | Rationale and condition |
| --- | --- | --- | --- |
| UD English EWT r2.10 | Existing train/dev/protected final | Approved, unchanged | Already authorized, qualified, and released under the conservative separate CC BY-SA model posture. |
| UD English ESLSpok r2.18 | Train plus dev; untouched test as source-specific protected final | **Conditional** | Best near-term non-NC complement: 2,320 L2 spoken sentences, UD UPOS, UD CC BY-SA 4.0, NICT JLE source CC BY-SA 3.0. Require exact source/license manifest, participant-safe handling, cross-source deduplication, and license-version review. |
| MASC 3.0 | Later broad-domain training candidate after mapping pilot | **Conditional** | About 500k words across 19 genres. Official landing page says CC BY 3.0 US, while another official page says “without restrictions”; use CC BY 3.0 conservatively. Requires exact archive/license pin, document-level rights audit, PTB-tag-to-UPOS mapping, and a native-UD consistency study. |
| UD English PUD r2.18 | Protected news/wiki diagnostic only | **Conditional** | Purpose-built 1,000-sentence test corpus under CC BY-SA 3.0. Never train or tune on it; audit source/translation attribution and cross-treebank overlap first. |
| UD English Pronouns r2.18 | Frozen targeted diagnostic only | **Conditional** | 285 constructed grammar examples under CC BY-SA 4.0. Useful for rare independent-genitive/pronoun behavior, not representative training data. |
| UD English CTeTex r2.18 | Frozen technical-domain diagnostic only | **Conditional** | 276 software-requirements sentences under UD CC BY-SA 4.0, drawn partly from PURE and public SRS documents. Require document-level underlying-text provenance before use. |
| UD English CHILDES r2.18 | Separate noncommercial lane only | Rejected | Large and valuable spoken/child-directed data, but the underlying TalkBank default is CC BY-NC-SA 3.0 with an express commercial-model restriction, plus sensitive participant/child-speech governance. |
| UD English GUM r2.18 | Separate noncommercial lane training | Rejected | Strongest broad multi-genre UD complement, but CC BY-NC-SA 4.0. A GUM-trained public model must not become the generally usable lane by accident. |
| UD English GENTLE r2.18 | Separate noncommercial protected diagnostic | Rejected | Designed as an out-of-domain test set and licensed CC BY-NC-SA 4.0. Do not train on it. |
| Tweebank v2 | Separate noncommercial social-media lane | Rejected | Useful UD-compatible social text, but CC BY-NC-SA 4.0 plus platform/deletion/privacy provenance. |
| UD English ParTUT / LinES r2.18 | Noncommercial research only | Rejected | Both carry NC terms; smaller benefit than GUM and additional parallel/source-text provenance. |
| UD English GUMReddit r2.18 | None | Rejected | Annotation repository is CC BY 4.0 but omits source text; reconstruction adds platform terms, deletion, attribution, and reproducibility risk. |
| UD English ATIS r2.18 | None until written/canonical source rights | Rejected | UD label is CC BY-SA 4.0, but the repository’s cited upstream chain does not establish a reliable original-text license. |
| UD English LittlePrince r2.18 | None pending translation rights | Rejected | Test-only literary sample under repository CC BY-SA 4.0, but the English translation/source-text permission chain is not established by the treebank page. |
| Treebank of Learner English / UD English ESL | None pending source access | Rejected | Annotations are available, but the underlying Cambridge Learner Corpus text is not distributed; do not reconstruct or acquire it without a separate agreement. |
| Penn Treebank, Revised News Treebank, OntoNotes | None under this decision | Rejected | LDC contract review, authorized access, non-redistribution controls, PTB-to-UPOS mapping, and model-distribution rights all require a separate consequential decision. |

“Rejected” here means rejected from the proposed general corpus packet, not a
claim that all possible research uses are unlawful.

## Frozen release pins for the UD inventory

If Peter chooses an expansion option, acquisition work must begin from the UD
v2.18 tags and verify these immutable commits before any label inspection:

| Treebank | UD v2.18 commit |
| --- | --- |
| CHILDES | `41596ed7793a47cb0cba96e40bf654404dbc171e` |
| ESLSpok | `f4d0ebe5eae0bc179a988934e15f018dcacfe73b` |
| ATIS | `dce409dd1b526745ab0980360ed2b9f803e93531` |
| PUD | `e173a1be1b442faf34e7d5a502189ad5d9d1e197` |
| LittlePrince | `a1936378cd57c9cda6cb57da938b164760a42529` |
| CTeTex | `3208ccc0be5c003d8357ad52ca2b359ae31eb6a4` |
| Pronouns | `9c330df067c606f91d80effb249550687c8ad8c3` |
| GENTLE | `93a5069df8ded256c3e038938b9ea17baf75c73c` |
| GUM | `b58e74bc22d17220c9198864c253a50a897bf27f` |
| GUMReddit | `cec04ae98e110305dc00cf147d6547c121d9d8dd` |
| ParTUT | `9cb91499ada4e284dfad3ea69b3b1dd10d2c516a` |
| LinES | `07a998d3fe0fa4e2bc6aaf651692615f89458a1a` |

These pins record the public release state; they do not by themselves clear the
underlying rights identified above.

## Options

### A — Non-NC staged expansion (recommended)

Keep one generally usable model lane. Authorize a source-specific audit and
manifest for ESLSpok, then use only its training split for a bounded mixed-data
experiment; use its dev split for development and keep its test split
protected. Add PUD and Pronouns as frozen diagnostics, and CTeTex only if its
document-level source audit clears. Treat MASC as a later, separate mapping
pilot rather than silently mixing PTB-style tags.

The replacement model, if it wins new Human-approved gates, remains a separate
CC BY-SA 4.0 model distribution with attribution for every source. EWT
performance must not regress beyond a predeclared tolerance, source-aware
sampling must prevent the larger corpus from dominating, and no evaluation set
may enter training.

**Benefit:** improves spoken/L2 coverage without accepting an NC model, while
adding clean held-out evidence outside EWT.

**Tradeoff:** limited near-term training volume; source and license-version
audit still precede acquisition; MASC breadth requires later mapping work.

### B — Separate noncommercial breadth lane

Keep the EWT model as the generally usable lane and create a clearly different
noncommercial research model using CHILDES and GUM training data, with GENTLE
and possibly Tweebank as protected diagnostics. License the model
CC BY-NC-SA, use a different package/model identity, and prevent automatic or
default substitution for the non-NC model.

**Benefit:** far more spoken and genre coverage quickly.

**Tradeoff:** commercial use is excluded; child/speech ethics and source-layer
governance are materially heavier; two model lanes increase maintenance and
user-confusion risk.

### C — Evaluation-first, no retraining

Do not authorize any new training corpus yet. After source clearance, freeze
PUD, Pronouns, CTeTex, and ESLSpok as diagnostic/protected evaluation sources,
measure the existing EWT model once, and use only aggregate error evidence to
decide whether a training expansion is justified.

**Benefit:** preserves the current model/license posture and produces evidence
before increasing corpus and release complexity.

**Tradeoff:** does not itself improve the tagger; observing a protected result
must not turn that set into development data.

Retaining EWT-only with no additional evaluation remains the explicit fallback
if none of these options is acceptable.

## Controls required under any expansion

- Create a corpus manifest before labels are exposed to development code. It
  records source URL, release/tag/commit, acquisition date and command, file and
  license hashes, annotation and original-text licensors, access terms,
  attribution/citation, permitted roles, and raw-data retention policy.
- Assign every document to exactly one of training, development, diagnostic, or
  protected-final roles; use source-provided splits when trustworthy.
- Deduplicate normalized documents/sentences across EWT and every new source
  before training, without using protected labels to tune the rule.
- Keep raw text, gold labels, personal metadata, and source-only identifiers out
  of the repository, wheels, sdists, model artifacts, logs, and public evidence.
- Evaluate source-specific metrics and aggregate metrics. Predeclare sampling,
  EWT non-regression, OOV/ambiguity, per-UPOS, model-size, latency, and
  memorization/recoverability gates before training.
- Make a new model candidate and qualification run. The published EWT-only
  result is historical evidence and cannot be silently overwritten.
- Preserve MIT code / CC model separation. Include all license texts,
  attribution, changes, training inputs and hashes, trainer revision, seed,
  dependency lock, model checksum, limitations, and source-specific metrics in
  the model distribution.

## Counsel or permission triggers

Stop and seek counsel or written permission before:

- putting any NC corpus or NC-derived model into a commercial or generally
  usable lane;
- relying on a UD repository license where the cited original text has no clear
  license or has stricter terms;
- accepting LDC, Cambridge Learner Corpus, platform-reconstructed social data,
  or another click-through/data-use agreement;
- combining ShareAlike versions when the planned outbound model license is not
  clearly compatible;
- distributing a model that can reproduce source passages, gold annotations,
  or a substantial database portion; or
- acquiring sensitive child, clinical, or identifiable speech outside the
  source’s access, ethics, and retention rules.

## Decision record

**Pending Peter’s selection.** No option, corpus, acquisition, or new model is
authorized until the Human decision is appended here and recorded on `hrxv`.

## Primary source notes

- [UD English CHILDES](https://universaldependencies.org/treebanks/en_childes/index.html)
  identifies the CC BY-SA 4.0 treebank and its CHILDES transcript source;
  [TalkBank’s ground rules](https://talkbank.org/0share/rules.html) supply the
  stricter underlying-data condition.
- [UD English ESLSpok](https://universaldependencies.org/treebanks/en_eslspok/index.html)
  identifies the treebank and NICT JLE source; the
  [NICT JLE page](https://alaginrc.nict.go.jp/nict_jle/index_E.html) supplies
  its CC BY-SA 3.0 source license.
- [UD English ATIS](https://universaldependencies.org/treebanks/en_atis/index.html)
  identifies its source chain; the cited
  [ATIS mirror README](https://github.com/howl-anderson/ATIS_dataset/blob/master/README.en-US.md)
  gives credits but no source-data license.
- [UD English PUD](https://universaldependencies.org/treebanks/en_pud/index.html),
  [Pronouns](https://universaldependencies.org/treebanks/en_pronouns/index.html),
  [CTeTex](https://universaldependencies.org/treebanks/en_ctetex/index.html),
  [GENTLE](https://universaldependencies.org/treebanks/en_gentle/index.html),
  and [GUM](https://universaldependencies.org/treebanks/en_gum/index.html)
  supply their intended roles, licenses, domains, and sizes.
- The [MASC landing page](https://anc.org/data/masc/) supplies the conservative
  CC BY 3.0 US posture; its
  [about page](https://anc.org/data/masc/about/) contains the conflicting
  “without restrictions” wording that must be resolved in an exact manifest.

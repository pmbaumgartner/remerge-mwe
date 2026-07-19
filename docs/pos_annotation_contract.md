---
kata: vdwc
created: 2026-07-19
---

# POS annotation and token-alignment contract

**Decision:** Adopt a public, sentence-nested annotated-token input
as the canonical POS representation, with a strict CoNLL-U adapter and a
separate raw-text built-in-tagger convenience path.

**Why now / higher-level goal:** `bdxy` must add occurrence-level English
UPOS filtering without changing existing unfiltered behavior or allowing a
tagger's tokenization to diverge from the discovery engine. This contract is
the shared boundary for the pretagged pilot (`npak`), quality oracle (`e6wr`),
and every candidate built-in tagger.

**Goal check:** The proposed contract preserves the current raw-string API as
is. It adds opt-in behavior only, keeps tags on token *occurrences* rather
than interned token IDs, and gives a built-in tagger one tokenization pass to
share with the engine.

**Consequential:** Yes — the selected shape becomes a public API, fixes the
training/inference boundary, and constrains all downstream implementations and
evaluation.

## Observed evidence

- The public API currently accepts `list[str]`; `_make_engine` passes that
  directly into the PyO3 `Engine` constructor
  ([`src/remerge/core.py`](../src/remerge/core.py)). The existing API has no
  way to represent an external token boundary or a tag occurrence.
- Rust ingestion first segments each document, then calls
  `split_whitespace()` on each segment
  ([`rust/src/interner.rs`](../rust/src/interner.rs)). It discards empty
  segments, retains document boundaries, and represents all remaining
  boundaries as engine lines.
- The existing `delimiter` splitter treats delimiter-separated segments as
  boundaries, while `sentencex` creates its own sentence boundaries. The
  latter currently ignores `line_delimiter`
  ([`tests/test_segmentation.py`](../tests/test_segmentation.py)).
- The engine's interner maps a surface form to one `TokenId`, so POS cannot
  live on an interned lexical ID: the same form can have different POS at
  different locations. Occurrence-aligned storage is required.
- The existing README deliberately documents whitespace normalization, so
  attempting to reconstruct supplied tags by re-tokenizing text would be
  ambiguous and would lose punctuation/contraction conventions
  ([`README.md`](../README.md)).
- Universal Dependencies CoNLL-U uses integer word lines for syntactic words;
  multiword-token range lines are not individually POS-tagged and empty-node
  decimal lines do not denote surface tokens. Its standard is UTF-8 NFC and
  blank lines delimit sentences. [UD CoNLL-U format](https://universaldependencies.org/format.html)
  The required English v1 tag universe is the 17 Universal POS tags.
  [UD UPOS inventory](https://universaldependencies.org/u/pos/)

## Evaluation criteria

1. A caller can state the exact token and sentence sequence to which supplied
   tags apply, including punctuation and contractions.
2. The representation permits distinct tags for equal token strings at distinct
   occurrences.
3. The representation is usable unchanged by pretagged input and generated
   tags, and can be converted to the engine without a second tokenizer pass.
4. A malformed, mismatched, unknown, or unreproducible annotation fails at the
   boundary rather than changing POS semantics silently.
5. Existing `run()` and `annotate()` calls retain their current raw-string and
   whitespace-normalization behavior when POS is not requested.

## Viable options

### Option A — raw documents plus parallel tag arrays

Keep `corpus: list[str]`, select an existing splitter, and add one tag array
per engine segment.

This minimizes a new public type and looks familiar to callers with already
split text. It fails the main contract: the caller cannot reliably predict
`split_whitespace()` or SentenceX output for punctuation, contractions,
Unicode, and CoNLL-U multiword tokens. It also makes the raw/tag alignment
dependent on an implementation detail and encourages tagger-specific
retokenization.

### Option B — canonical nested annotated tokens, with adapters **(recommended)**

Add a `TaggedDocument` representation of explicit, non-empty sentences of
`TaggedToken(form, upos)`. A pretagged discovery entry point accepts only this
representation. A separate CoNLL-U adapter converts corpus files to it. The
built-in tagger's raw-text convenience entry point first creates the same
representation, then gives its token vectors directly to the engine.

This adds a small public type but makes every boundary explicit, makes
alignment validation local and deterministic, and gives training/evaluation a
stable conversion rule.

### Option C — CoNLL-U text as the only tagged input

Accept only CoNLL-U and treat it as both the interchange and public API.

This gives a known file format but makes ordinary pretagged use unnecessarily
heavy, leaks parsing/training details into the discovery API, and cannot
represent tags produced in memory without serializing and parsing a document.

## Accepted contract

Choose **Option B** with the following normative v1 behavior.

### 1. Canonical representation and public surface

The public tagged-input entry point is separate from the existing raw-string
`run()` and `annotate()` entry points. Its exact Python names can be chosen
during `npak`, but it must accept only this logical shape:

```python
@dataclass(frozen=True, slots=True)
class TaggedToken:
    form: str
    upos: str

@dataclass(frozen=True, slots=True)
class TaggedDocument:
    sentences: tuple[tuple[TaggedToken, ...], ...]
    language: str = "en"
    source: Literal["supplied", "builtin"] = "supplied"
    model_id: str | None = None
```

`sentences` is the canonical occurrence order. A token occurrence is identified
by `(document_index, sentence_index, token_index)` before merges; `upos` is
stored at that occurrence, not on a `TokenId`, lexeme, or MWE type. A document
may be empty; a non-empty document contains only non-empty sentences, and a
sentence contains one or more tokens.

The data classes are illustrative API names, not a separate decision. A
compatible frozen record type or validated mapping is acceptable only if it
has exactly the same values, nesting, and validation behavior. Do not overload
the existing raw `corpus` argument with these values: that would weaken typing
and risk interpreting a sequence of strings as the wrong representation.

### 2. Forms, Unicode, case, and UPOS

- `form` must be a non-empty NFC Unicode string containing no Unicode
  whitespace. The API rejects non-NFC forms rather than normalizing them;
  this keeps alignment, corpus hashes, and model evidence reproducible.
- Preserve form case exactly. A model may derive case features internally, but
  it must emit the original canonical form and may not case-fold the engine
  token stream.
- `upos` must be exactly one of `ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`,
  `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`,
  `VERB`, or `X`. `_`, lowercase aliases, XPOS, morphology, and missing tags
  are invalid. `X` is an accepted observed category, not a missing-value
  sentinel.
- Punctuation is an ordinary token with `PUNCT`; it is never dropped just
  because the old whitespace tokenizer would have attached it to a word.

### 3. Sentence and document boundaries

- For pretagged input, `TaggedDocument.sentences` is authoritative. The
  existing `splitter`, `line_delimiter`, and `sentencex_language` parameters
  are invalid on that entry point rather than silently ignored. No candidate
  may cross a supplied sentence boundary.
- For raw input with a built-in tagger, sentence segmentation is part of that
  tagger's documented preprocessing profile. The profile must name its
  segmenter and version, language (`"en"` in v1), and tokenization rules; it
  yields `TaggedDocument` before engine construction. The tagger and engine
  share those tokens and boundaries exactly.
- Existing non-POS calls retain their existing splitter behavior unchanged.

### 4. CoNLL-U conversion

`from_conllu()` is an import adapter, not an alternate canonical input format.
It creates one `TaggedDocument` per CoNLL-U document (`# newdoc id` starts a
new document; a file without one is one document), and one canonical sentence
per blank-line-delimited CoNLL-U sentence.

- Require valid UTF-8 NFC text, ten tab-separated fields, and non-empty
  integer-ID word lines with a valid UPOS field.
- Retain only integer-ID word lines, in their file order. Skip multiword-token
  range IDs and decimal empty-node IDs. Preserve their syntactic-word `FORM`
  and UPOS on the retained records; do not substitute a multiword surface form
  or derive a tag from XPOS/FEATS.
- Preserve punctuation word lines. `# text` and `SpaceAfter=No` are provenance
  diagnostics only; v1 neither aligns a raw document against them nor uses
  them to detokenize.
- Reject an invalid row or sentence; do not repair, re-tokenize, or pad it.
  This is deliberate: any raw text intended for built-in tagging must first
  pass through that model's frozen tokenizer instead.

### 5. Metadata, confidence, and reproducibility

- Every tagged document records `language="en"` in v1. Other languages are
  rejected at the public boundary.
- `source="supplied"` may include an optional caller-provided `model_id` as
  provenance. `source="builtin"` requires a non-empty immutable identity
  containing the model artifact version/content digest and preprocessing
  profile version. The result/diagnostic metadata must expose this identity.
- Public per-token confidence is **not in v1**. Internally produced confidence
  may be retained for evaluation, but it must not become a public value until
  `e6wr` approves a calibration metric, dataset, and maximum error.

### 6. Validation and one-pass integration

- The tagged entry point validates the complete nested structure before engine
  construction. Errors are `ValueError` and identify the first failing
  document/sentence/token coordinate and field; there is no truncation,
  broadcasting, fallback tag, or implicit retokenization.
- The implementation creates the interner and occurrence annotations from the
  validated token vectors. The raw built-in pipeline must produce those vectors
  once and pass them to the engine. It must not tokenize raw text for tagging
  and then call the legacy raw-string constructor for discovery.
- The unfiltered raw path must not load a model, invoke the POS tokenizer, or
  acquire POS-specific memory unless POS behavior was explicitly selected.

## Acceptance examples for the future implementation

These examples make the selected semantics testable; they do not prescribe
filtering policy, which remains `ycg1`'s decision.

```python
TaggedDocument(
    sentences=((
        TaggedToken("I", "PRON"),
        TaggedToken("can't", "VERB"),
        TaggedToken("go", "VERB"),
        TaggedToken(".", "PUNCT"),
    ),),
)

# Same spelling, distinct occurrence-level tags.
TaggedDocument(
    sentences=((
        TaggedToken("watch", "VERB"),
        TaggedToken("the", "DET"),
        TaggedToken("watch", "NOUN"),
    ),),
)
```

The first sentence contains four canonical tokens even though legacy
`split_whitespace()` would see `"can't"` and `"."` only according to spaces
in a raw string. The second proves that one interned surface form must not
carry a global POS value. A CoNLL-U range row such as `1-2\tvámonos\t_ ...`
does not itself become a `TaggedToken`; its following integer word rows do.

## Consequences and follow-on work

- `npak` implements the tagged-input entry point, occurrence-aligned storage,
  validation, and no-model pretagged path against this contract.
- `e6wr` may define quality and calibration gates only after this token unit,
  sentence unit, and model-identity contract is settled.
- `wtp0` and `0bhv` must freeze and verify the adapter's CoNLL-U parsing and
  any model preprocessing-profile provenance.
- `a4jp`, `wsjt`, and `2hfz` must train/evaluate on the adapter's retained
  integer-ID word sequence and demonstrate that their raw inference tokenizer
  produces the same canonical token sequence for frozen evaluation examples.
- `wra2` integrates the selected model by producing the canonical token vectors
  once; no POS path may invoke the legacy raw tokenizer a second time.

## What would change the recommendation

Choose Option A only if a measured prototype demonstrates that every supported
tag source and built-in tokenizer produces the exact same token sequence as
the existing raw engine for punctuation, contractions, Unicode, and sentence
boundaries. Current code evidence contradicts that premise.

Choose Option C only if the product deliberately limits all pretagged users to
UD CoNLL-U files and accepts serialization as the in-memory API boundary.

Relax NFC rejection only if an approved provenance and hashing scheme proves
that normalizing input cannot obscure a supplied-token alignment error. Add
public confidence only after `e6wr` freezes a calibration contract.

## Decision record

- **Option A:** rejected. Raw documents plus parallel tag arrays have the
  smallest apparent API,
  but alignment remains coupled to hidden tokenization behavior.
- **Option B:** accepted. Explicit sentence-nested annotated tokens plus CoNLL-U adapter;
  a small new type, but stable occurrence alignment and one-pass integration.
- **Option C:** rejected. CoNLL-U only is standard interchange, but burdens ordinary
  callers and does not fit generated tags naturally.
- **Decision:** Peter accepted the recommendation on 2026-07-19 through his
  blanket approval of all project recommendations. Option B and the normative
  v1 rules above are binding downstream inputs.
- **What would change the recommendation:** a prototype proving exact shared
  tokenization for every supported source (Option A), or a deliberate decision
  to make CoNLL-U the sole user-facing format (Option C).

Peter Baumgartner is the Human-mode decision owner.

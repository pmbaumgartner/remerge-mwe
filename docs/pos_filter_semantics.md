# POS filtering semantics decision packet

**Kata:** `ycg1`  
**Status:** recommended option accepted by Peter's blanket approval on 2026-07-19  
**Decision owner:** Peter (technical and acceptance authority)

## Decision

Adopt **occurrence-constrained discovery** for v1. POS patterns determine which
individual token occurrences participate in candidate counts, scores, merges,
returned winners, and annotations. They never become attributes of a lexical
type, and a matching occurrence must not make differently tagged occurrences of
the same surface form eligible.

This is the only filtering mode in v1. Do not add output-only post-filtering,
type-level any/all/ratio gates, confidence thresholds, or regex-like pattern
syntax.

## Why

The current engine selects a `BigramId` from global type-level counts, then
merges every non-overlapping occurrence of that bigram. Applying POS only after
selection would therefore allow rejected occurrences to affect frequency,
score, merge count, later candidates, and annotations. A type-level gate has the
same flaw: one accepted tag sequence would authorize all surface-identical
occurrences.

Occurrence-constrained discovery makes the reported result literal: every count
and annotated span supporting a POS-filtered winner satisfied the requested POS
pattern.

## Normative v1 semantics

### Pattern language

- A filter is an OR-list of patterns.
- A pattern contains at least two positions.
- Each position is one exact Universal POS tag, a finite set of exact tags, or
  `*`.
- `*` matches any UPOS value, including `X`. An exact `X` matches only `X`.
- Pattern and alternative order are canonicalized; duplicates are removed.
- Empty patterns, one-position patterns, unknown tags, empty alternatives, and
  malformed input are errors.
- There is no negation, optional position, repetition, arbitrary regular
  expression, confidence predicate, lemma predicate, or feature predicate in
  v1.

The annotation contract supplies one resolved UPOS tag per token. Missing or
invalid tags fail validation before discovery; they are not silently coerced to
`X`.

### Occurrence eligibility

Evaluate patterns against the original token occurrences and sentence
boundaries. Never aggregate POS onto a `LexemeId` or surface string.

An active lexeme span is:

1. a **full match** when its original UPOS sequence matches a complete pattern;
2. **viable support** when it is a contiguous proper subspan of at least one
   pattern; or
3. **ineligible** otherwise.

Only adjacent occurrences whose combined original span is a full match or
viable support may become merge candidates. This support rule permits the
internal merges needed to discover MWEs of length three or more. Support merges
are internal search state: they are not returned and are not annotated unless
their resulting span is also a full match.

If a span is both a full match and viable support for a longer pattern, it is an
emitted winner when selected and may continue to participate in discovery of
the longer pattern.

### Counts and scores

Candidate statistics use the same occurrence population as eligibility:

- candidate frequency is the number of eligible, non-overlapping occurrence
  locations for that candidate at selection time;
- left and right marginals and the total adjacency count used by the scoring
  function are derived from the active eligible adjacency population, not the
  unfiltered corpus;
- `min_count` and `min_score` apply to this filtered population;
- existing deterministic score, frequency, and lexical tie-breaking remains in
  force;
- `merge_token_count` is the number of eligible occurrences merged for the
  emitted winner, never the number of surface-identical occurrences in the raw
  corpus.

There is no separate any/all/majority acceptance ratio. A candidate is eligible
when its matching occurrence count reaches `min_count`.

### Iteration, stopping, and output

- The requested iteration count counts emitted full-match winners.
- Internal viable-support merges do not consume that count and are not included
  in returned winner lists.
- Search stops when the requested number of full matches is emitted or no
  qualifying full/support candidate remains.
- The existing exhausted-search policy still applies: stop or raise according
  to the caller's explicit setting; do not pad results.
- `min_score` applies to support merges as well as full matches. If no remaining
  eligible candidate clears it, discovery is exhausted.
- Only occurrences of emitted full-match winners receive MWE annotations.
  Internal support merges and rejected surface-identical occurrences remain
  unannotated.
- Progress exposed as user-visible iterations counts emitted winners. Internal
  merge-step counts may be exposed only as diagnostics.

### Sentence and overlap behavior

Patterns cannot cross sentence boundaries. Existing left-to-right overlap
cleanup remains deterministic, but it operates only on eligible locations.
Eligibility is recomputed or incrementally maintained for the affected
locations after each merge.

## Examples

### Same words, different POS

For surface text `record deal`, suppose one occurrence is `VERB NOUN` and
another is `NOUN NOUN`. Under the pattern `VERB NOUN`, only the first occurrence
contributes to frequency and scoring, is merged, and may be annotated. The
second remains two tokens. With `min_count=2`, this candidate is rejected unless
there are at least two matching `VERB NOUN` occurrences.

### Three-token MWE

For pattern `ADJ NOUN NOUN` and tokens `new york office`, the engine may first
merge an eligible proper subspan as viable support. That internal merge is not a
result and consumes no requested iteration. When the full three-token span is
selected, it is returned and annotated as one MWE occurrence.

### Exhaustion

If the caller requests five winners but the eligible population yields two,
the engine returns two or raises according to `on_exhausted`. Internal support
merges never manufacture additional winners.

## Options considered

### A. Occurrence-constrained discovery — accepted

Filter the discovery population and every downstream statistic and mutation.
This has the strongest semantic integrity and directly meets the project
outcome, at the cost of occurrence-aware accounting and internal support state.

### B. Output-only post-filtering — rejected

Run the current engine unchanged and discard nonmatching outputs. This is easy
and fast, but rejected occurrences still determine scores, merges, later
candidates, and annotations. It does not provide POS-aware discovery.

### C. Type-level gate, then merge every occurrence — rejected

Accept a lexical type when any, all, or a ratio of its occurrences match. This
adds policy surface while still merging rejected occurrences and making counts
hard to interpret.

## Implementation guardrails

- Preserve original per-token UPOS sequences through merges so eligibility is
  occurrence-local and auditable.
- Maintain filtered counts incrementally around changed locations; do not
  rescan the corpus after every merge.
- Keep the current unfiltered fast path unchanged when no POS filter is
  supplied.
- Add differential tests against a small, obviously correct reference
  implementation and explicit mixed-POS/long-pattern cases.
- Benchmark both unfiltered overhead and filtered throughput before accepting
  the implementation gate.

## Revisit triggers

Reopen this decision only with evidence for one of these needs: confidence-aware
filtering, a richer pattern language, user demand for an explicitly named
output-only analysis mode, or unacceptable measured cost that cannot be fixed
within the occurrence-aware design.

# REMERGE Improvement Plan (Balanced Reliability + Algorithm Quality)

## Summary
Stabilize correctness first (empty-candidate handling and `min_count` math), then add deterministic extraction controls and targeted algorithm quality improvements, while tightening tests and maintainability. Keep core behavior recognizable but make outcomes reproducible and safer for real corpora.

## Public API / Interface Changes
1. Extend `run()` with:
   - `tie_breaker: Literal["legacy_first_seen", "deterministic"] = "deterministic"`
   - `on_exhausted: Literal["stop", "raise"] = "stop"`
   - `min_score: Optional[float] = None` (early-stop threshold on winning score; method-specific interpretation)
2. Add `NoCandidateBigramError(ValueError)` used when `on_exhausted="raise"` and no valid candidate remains.
3. Replace `SelectionMethod` alias with `class SelectionMethod(str, Enum)` while continuing to accept equivalent string values for backward compatibility.

## Implementation Plan
1. Correctness foundation
   - Introduce a unified candidate-building path that returns scored candidates plus global totals from unfiltered bigram stats.
   - Apply `min_count` only as a candidate inclusion filter.
   - Ensure LL/NPMI denominator uses full current bigram total, not filtered total.
   - Make selectors return "no candidate" cleanly instead of throwing from `argmax`/indexing.
2. Run-loop robustness
   - In `run()`, handle "no candidate" via `on_exhausted`:
     - `"stop"`: break loop and return winners collected so far.
     - `"raise"`: raise `NoCandidateBigramError` with method/min_count context.
   - Apply `min_score` after selecting the best candidate; if below threshold, stop (or raise if `on_exhausted="raise"`).
3. Deterministic tie-breaking (algorithm quality + reproducibility)
   - For equal score, rank by:
     - higher raw bigram frequency
     - lexicographic order of concatenated token tuple (stable final tie-break)
   - Keep `"legacy_first_seen"` for compatibility testing and migration.
4. Maintainability/performance refactor
   - Compute `clean_bigram_locations` once per winner and reuse it through merge/update steps.
   - Change `bigrams_to_locations` value type from `List[...]` to `Set[...]` for O(1)-style removal and easier invariant checks.
   - Prune empty location entries immediately after updates.
   - Remove unused imports/types and isolate scoring math into small pure functions.
5. Test expansion
   - Add deterministic unit tests for:
     - empty corpus and single-token corpus
     - `iterations` greater than available merges
     - `min_count` filtering behavior across all methods (including frequency)
     - LL/NPMI denominator correctness with `min_count > 0`
     - tie resolution under both tie-breaker modes
     - `on_exhausted` stop vs raise behavior
     - `min_score` early stopping
   - Add invariant tests after each merge:
     - counter totals match location maps
     - no negative counts
     - winner bigram removed after merge
   - Sort fixture corpus file loading for deterministic test input order.
6. Docs and migration notes
   - Update README API table/examples for new `run()` options and deterministic default behavior.
   - Document semantic change: returned winners may be fewer than requested `iterations` when exhausted or thresholded.
   - Add short "reproducibility" section explaining tie-break choices.

## Test Cases and Scenarios
1. `run([["a"]], 5, on_exhausted="stop")` returns `[]` without error.
2. Same input with `on_exhausted="raise"` raises `NoCandidateBigramError`.
3. Frequency method respects `min_count` and excludes low-frequency winners.
4. NPMI/LL with `min_count` produce finite values and stable rankings using full denominators.
5. Tie corpus with shuffled line/file order gives identical output under `tie_breaker="deterministic"`.
6. Existing regression corpus still yields expected first winners unless explicitly changed by deterministic tie policy.
7. Merge invariants pass across multi-iteration synthetic corpora with repeated/overlapping patterns.

## Assumptions and Defaults
1. Small API additions are acceptable.
2. Balanced sequencing is preferred: each milestone includes one reliability item and one algorithm-quality item.
3. Default behavior should favor reproducibility and safety:
   - `tie_breaker="deterministic"`
   - `on_exhausted="stop"`
   - `min_score=None` (disabled unless user sets it)
4. Discontinuous/gapped bigrams remain out of scope for this iteration.

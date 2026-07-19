---
kata: 0bhv
created: 2026-07-19
---

# Retired generated-tagger benchmark protocol

This records the frozen protocol used to reject the linear English UPOS model.
Its executable release harness was removed when v1 reshaped to supplied tags;
it is not a current product or release command. The protected gates and frozen
manifest remain historical evidence.

A future generated-tagger attempt requires a new Kata outcome, a fresh
candidate registration, and an explicitly authorized executable oracle before
any protected-final evaluation. The isolated rejected trainer and loader live
under `experiments/pos-linear/` and use development data only.

## Boundary and provenance

- The final gold split is checksum-verified, offline-only data. It is never
  committed, downloaded by the test, put in a package, or read by ordinary
  training/dev tests.
- `tests/pos/evaluation/loader.py` is the only accepted loader. Its
  `load_final_gold(..., allow_final=True)` path verifies source checksums,
  split isolation, adequacy floors, canonical CoNLL-U alignment, and the
  adjudicated MWE spans before the release harness sees a final token.
- The 100,000-token runtime fixture is generated in
  `tests/pos/conftest.py` from project-authored token/tag sequences. Its
  SHA-256 is recorded in every result. It is a workload control, not a gold
  quality corpus or a claim about spoken-domain performance.
- The prior 359-file transcript corpus is not used for any POS quality,
  resource, or release decision.

## Candidate adapter

`--pos-tagger MODULE:FACTORY` identifies a zero-argument factory. Its object
must expose the following narrow adapter; it is intentionally not a production
tagger API.

```python
class BenchmarkTagger:
    model_id: str             # immutable model + preprocessing identity
    tokenizer_id: str         # immutable tokenizer/profile identity
    artifact_bytes: int       # compressed model artifact size
    artifact_sha256: str      # lowercase digest of that artifact

    def tag(self, documents: tuple[tuple[tuple[str, ...], ...], ...]) -> list[TaggedDocument]: ...
    def tag_text(self, documents: tuple[str, ...]) -> list[TaggedDocument]: ...
```

`tag()` receives fixed document/sentence/token boundaries and must return the
same forms and nesting. Its exact model and tokenizer identities live in the
candidate-registration and benchmark evidence objects, not discovery input.
`tag_text()` is used only for raw-text end-to-end timing and must reproduce the
same frozen boundaries. Any changed document count, sentence boundary, token
boundary, form, or UPOS value rejects the run. The harness does not re-tokenize
or repair a result.

## Required inputs before final evaluation

The evaluator rejects rather than guesses when any input is absent.

1. `--pos-benchmark-manifest`: the committed frozen manifest.
2. `--pos-benchmark-acquisition-root`: an operator-provided directory holding
   the pinned offline EWT/STREUSLE clones expected by the manifest.
3. `--pos-candidate-registration`: a read-only JSON object created before any
   final run, with `candidate_id`, `source_revision`, `trainer_command`,
   `seed`, `model_id`, `artifact_sha256`, `dev_report`, and
   `final_evaluated: false`. Its model identity and digest must equal the
   adapter values. A registered candidate receives one final run only.
4. `--pos-core-baseline-evidence`: the frozen pre-POS core baseline JSON with
   `fixture_sha256` and positive `unfiltered_core_tokens_per_second` measured
   from the same project-authored 100k workload.
5. `--pos-benchmark-evidence`: a new JSON output path. A rejected run writes
   its diagnostic evidence there before returning nonzero.

The candidate-registration and baseline files are operator records, not
repository fixtures. Editing an acceptance threshold, split, digest, or
manifest after observing final results requires the separate change authority
recorded in `pos_quality_contract.md`.

## Executable status

There is intentionally no root test command for this protocol. Normal pytest,
Cargo, release CI, and artifact audits do not import, compile, or special-case
the rejected generated-tagger architecture. The retained experiment commands
are documented in `experiments/pos-linear/README.md`; they do not consume the
protected final split or certify a model.

## Metrics and exact MWE utility

The harness computes final-token micro UPOS accuracy, macro F1, OOV accuracy,
ambiguous-token accuracy, every named-domain accuracy, and F1 for each UPOS
class. OOV and ambiguity use the frozen training vocabulary only. It calculates
two-sided 95% intervals with 10,000 seeded document-clustered bootstrap
resamples.

For MWE utility, it runs unfiltered, gold-filtered, and generated-filtered
discovery with the manifest's patterns and discovery configuration. Candidate
and gold MWEs are compared as exact original
`(document, sentence, start-token, end-token)` occurrences. The harness
requires structured occurrence diagnostics from both `run_with_occurrences()`
and tagged discovery; it never reconstructs positions from rendered annotation
strings. Empty candidate or gold populations reject rather than producing a
vacuous percentage.

The protected acceptance floors are those in `pos_quality_contract.md`:

- overall lower accuracy bound 95.0%, macro F1 80.0%, OOV 82.0%, ambiguous
  88.0%, each domain 90.0%, and class F1 50.0% when its final count is at
  least 50;
- generated-filter precision improves by at least 10 points, removes at least
  10% of unfiltered candidates, has at least 70% recall, and stays within the
  stated 5-point unfiltered/gold-filtered recall losses; and
- the precision-improvement interval is positive and recall-loss interval
  upper bounds stay within their limits.

## Performance protocol and rejection rules

All timing uses a release build, the explicit one-thread environment above,
three warmups, then 15 measured repetitions. The evidence records individual
durations, median, p95 where required, and IQR/median. If IQR/median exceeds
10%, stabilize the environment and rerun; a second high-variance result is
invalid, never a pass.

The fixture has four deterministic shapes: 256-token small call, 4,096-token
medium batch, 16,384-token long segment, and the 100,000-token reference
batch. The harness separately reports inference-only, filtering-only,
unfiltered discovery-only, and raw-text-to-filtered-MWE costs. Cold loading is
measured in 15 new Python processes, separate from warm calls; RSS is
incremental process peak RSS, with macOS' byte unit and Linux's KiB unit
normalized to bytes. Artifact bytes remain distinct from memory.

Resource rejection limits are: 20 MiB compressed artifact, 250 ms median cold
load, 128 MiB incremental peak RSS, 50k inference-only tokens/s, 35k
end-to-end tokens/s and 35% of same-run unfiltered-core throughput, 15 ms warm
small-call p95, and no more than a 5% unfiltered-core regression from the
frozen pre-POS baseline. The long shape must complete with matching boundaries
and reports RSS; it has no extra speed floor.

## Evidence format

The JSON evidence is stable-key formatted and contains the invoked command,
timestamp, adapter/model identities and digest, protocol version, hardware and
power-mode metadata, candidate registration, manifest-derived quality and
interval results, exact utility results, workload token counts and hashes,
individual timing samples/statistics, cold RSS/load samples, frozen baseline,
and final `accepted` or `rejected` status. A rejected run includes the failure
class/message before pytest exits nonzero.

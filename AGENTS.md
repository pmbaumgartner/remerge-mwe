# AGENTS.md

- Before finishing a task, run `uv run --no-sync prek run --all-files`.
- Use `uv run --no-sync pytest -q -m "not performance"` while iterating; run the opt-in performance guard only for performance-sensitive changes.
- After changing `rust/`, run `cargo test`, rebuild with `uv run --no-sync maturin develop`, then run the affected Python tests; rebuild first if Python behavior appears stale.
- To inspect PyO3 interpreter selection, run `PYO3_PRINT_CONFIG=1 uv run --no-sync maturin develop`; this diagnostic intentionally exits nonzero after printing the configuration. Force the project interpreter with `PYO3_PYTHON=.venv/bin/python` if needed.
- If `cargo test` reports unresolved Python symbols, keep `python-extension` out of Cargo's default features; after correcting the feature configuration, run `cargo clean` once before retesting.

<!-- BEGIN KATA (managed by `kata init --with-agents`) -->
## kata issue tracker

This project uses [kata](https://github.com/kenn-io/kata) as its shared issue
ledger. Run `kata quickstart` at the start of each session for the full agent
contract. The short version:

- Search before creating: `kata search "<keywords>" --agent`.
- Prefer updating existing issues over duplicates (`kata comment`, `kata label add`, `kata edit`).
- Default to `--agent` for ordinary reads and mutations; use `--json` only when a script needs structured data.
- Close only verified work: `kata close <ref> --done --message "<scope + verification>" --commit <sha>`.
- If work is incomplete, label `needs-review` and comment what remains rather than closing.
- Never `kata delete` or `kata purge` without explicit user authorization.

## kata work.* conventions (agent orchestration)

When working a kata-tracked issue, keep its `work.*` metadata truthful:

- On claim/start: `kata meta set <ref> work.attention ok`; if the work has a
  dedicated branch, stamp it once with `kata meta set <ref> work.branch <branch>`.
- Signal live state: `kata meta set <ref> work.attention stuck|needs-human|ok`
  plus a one-line `work.attention_msg` saying why. Raise `stuck` when you cannot
  proceed, `needs-human` when you want review; clear back to `ok` when unblocked.
- Never stop with the signal stale: close the issue, or leave the attention
  pair reflecting the hand-off.
- Coordinators read `work.*` on issues they delegated; only the working agent
  writes them. `work.*` on closed issues is meaningless.
<!-- END KATA -->

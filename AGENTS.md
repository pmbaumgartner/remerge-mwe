# AGENTS.md

- Before finishing a task, run `uv run --no-sync prek run --all-files`.
- Use `uv run --no-sync pytest -q -m "not performance"` while iterating; run the opt-in performance guard only for performance-sensitive changes.
- After changing `rust/`, run `cargo test`, rebuild with `uv run --no-sync maturin develop`, then run the affected Python tests; rebuild first if Python behavior appears stale.
- To inspect PyO3 interpreter selection, run `PYO3_PRINT_CONFIG=1 uv run --no-sync maturin develop`; this diagnostic intentionally exits nonzero after printing the configuration. Force the project interpreter with `PYO3_PYTHON=.venv/bin/python` if needed.
- If `cargo test` reports unresolved Python symbols, keep `python-extension` out of Cargo's default features; after correcting the feature configuration, run `cargo clean` once before retesting.

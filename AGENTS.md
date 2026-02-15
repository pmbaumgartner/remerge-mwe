# AGENTS.md

## Project Tooling
- Package and environment management: `uv`
- Linting/formatting: `ruff`
- Type checking: `ty`

## Standard Workflow
1. Sync the environment after pulling or changing dependencies:
   - `uv sync --all-groups`
2. If dependency constraints changed in `pyproject.toml`, refresh lock + sync:
   - `uv lock`
   - `uv sync --all-groups --frozen`
3. After code changes, run quality checks (formatting first):
   - `uv run ruff format src tests`
   - `uv run ruff check src tests`
   - `uv run ty check src tests`
4. Run tests before finishing:
   - `uv run pytest -v`

## Rust/PyO3 Fast Loop
1. Recommended command order for quickest feedback:
   - `cargo check`
   - `cargo test`
   - `uv run --no-sync maturin develop`
   - `uv run --no-sync pytest -q tests/test_smoke.py`
   - `uv run --no-sync pytest -q tests/test_remerge.py::test_single_iter`
2. Build/install the Rust extension after Rust code changes:
   - `uv run --no-sync maturin develop`
3. Run targeted tests while iterating:
   - `uv run --no-sync pytest -q tests/test_remerge.py -k "<pattern>"`
4. If behavior looks stale, rebuild the extension first:
   - `uv run --no-sync maturin develop`
5. If PyO3 build detection seems wrong, print config and verify interpreter:
   - `PYO3_PRINT_CONFIG=1 uv run --no-sync maturin develop`
   - `PYO3_PYTHON=.venv/bin/python uv run --no-sync maturin develop`
6. If `cargo test` fails with unresolved Python symbols, verify `Cargo.toml` is not forcing `pyo3/extension-module` and rerun:
   - `cargo clean`
   - `cargo test`

## Full Verification Before Handoff
1. Python checks:
   - `uv run ruff format src tests`
   - `uv run ruff check src tests`
   - `uv run ty check src tests`
   - `uv run --no-sync pytest -v`
2. Rust checks:
   - `cargo fmt --all`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test`

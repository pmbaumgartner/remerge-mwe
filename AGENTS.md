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
3. After code changes, run quality checks:
   - `uv run ruff check src tests`
   - `uv run ty check src tests`
4. Run tests before finishing:
   - `uv run pytest -v`


#!/usr/bin/env bash
set -euo pipefail

BUILD_MODE="skip"
RUNS=1
ITERATIONS=5
INGEST="rust"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD_MODE="${2:-}"
      shift 2
      ;;
    --runs)
      RUNS="${2:-}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:-}"
      shift 2
      ;;
    --ingest)
      INGEST="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bin/benchmark-remerge.sh [--build skip|debug|release] [--runs N] [--iterations N]
       [--ingest rust|python|both]

Examples:
  bin/benchmark-remerge.sh --build release --runs 3 --iterations 5 --ingest rust
  bin/benchmark-remerge.sh --build debug --runs 1 --iterations 2
  bin/benchmark-remerge.sh --build skip --runs 5 --iterations 5 --ingest both
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$BUILD_MODE" != "skip" && "$BUILD_MODE" != "debug" && "$BUILD_MODE" != "release" ]]; then
  echo "--build must be one of: skip, debug, release" >&2
  exit 1
fi
if [[ "$INGEST" != "rust" && "$INGEST" != "python" && "$INGEST" != "both" ]]; then
  echo "--ingest must be one of: rust, python, both" >&2
  exit 1
fi

if [[ "$BUILD_MODE" == "debug" ]]; then
  echo "[build] uv run --no-sync maturin develop"
  uv run --no-sync maturin develop
elif [[ "$BUILD_MODE" == "release" ]]; then
  echo "[build] uv run --no-sync maturin develop --release"
  uv run --no-sync maturin develop --release
fi

echo "[bench] runs=$RUNS iterations=$ITERATIONS ingest=$INGEST"

uv run --no-sync python - <<PY
import inspect
from pathlib import Path
from statistics import mean
from time import perf_counter

import remerge

runs = int(${RUNS})
iterations = int(${ITERATIONS})
ingest_mode = "${INGEST}"


def load_sample_docs() -> list[str]:
    docs: list[str] = []
    root = Path("tests/sample_corpus")
    for txt in sorted(root.glob("*.TXT")):
        docs.append(txt.read_text())
    return docs


def python_tokenize_docs(docs: list[str]) -> list[list[str]]:
    # Mirrors the old fixture behavior (split by newline, then " ").
    corpus: list[list[str]] = []
    for doc in docs:
        for line in doc.split("\n"):
            if line:
                corpus.append(line.split(" "))
    return corpus

docs = load_sample_docs()
run_sig = inspect.signature(remerge.run)
supports_rust_ingest = "line_delimiter" in run_sig.parameters
supports_python_ingest = not supports_rust_ingest

requested_modes = (
    ["rust", "python"] if ingest_mode == "both" else [ingest_mode]
)

configs = [
    ("log_likelihood", {"method": "log_likelihood", "min_count": 0}),
    ("frequency", {"method": "frequency", "min_count": 0}),
    ("npmi", {"method": "npmi", "min_count": 25}),
]

for mode in requested_modes:
    if mode == "rust" and not supports_rust_ingest:
        print("[skip] ingest=rust unsupported in this checkout (run() has no line_delimiter)")
        continue
    if mode == "python" and not supports_python_ingest:
        print("[skip] ingest=python unsupported in this checkout (run() expects document strings)")
        continue

    print(f"\\n[mode] ingest={mode}")
    for label, kwargs in configs:
        samples = []
        first_winner = None
        for _ in range(runs):
            t0 = perf_counter()
            if mode == "rust":
                winners = remerge.run(
                    docs,
                    iterations=iterations,
                    line_delimiter="\\n",
                    **kwargs,
                )
            else:
                corpus = python_tokenize_docs(docs)
                winners = remerge.run(
                    corpus,
                    iterations=iterations,
                    **kwargs,
                )
            dt = perf_counter() - t0
            samples.append(dt)
            if first_winner is None and winners:
                first_winner = winners[0].merged_lexeme.word
        print(
            f"{label:14s} avg={mean(samples):8.3f}s min={min(samples):8.3f}s max={max(samples):8.3f}s first={first_winner}"
        )
PY

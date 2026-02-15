#!/usr/bin/env bash
set -euo pipefail

BUILD_MODE="skip"
RUNS=1
ITERATIONS=5

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
    -h|--help)
      cat <<'USAGE'
Usage: bin/benchmark-remerge.sh [--build skip|debug|release] [--runs N] [--iterations N]

Examples:
  bin/benchmark-remerge.sh --build release --runs 3 --iterations 5
  bin/benchmark-remerge.sh --build debug --runs 1 --iterations 2
  bin/benchmark-remerge.sh --build skip --runs 5 --iterations 5
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

if [[ "$BUILD_MODE" == "debug" ]]; then
  echo "[build] uv run --no-sync maturin develop"
  uv run --no-sync maturin develop
elif [[ "$BUILD_MODE" == "release" ]]; then
  echo "[build] uv run --no-sync maturin develop --release"
  uv run --no-sync maturin develop --release
fi

echo "[bench] runs=$RUNS iterations=$ITERATIONS"

uv run --no-sync python - <<PY
from pathlib import Path
from statistics import mean
from time import perf_counter

import remerge

runs = int(${RUNS})
iterations = int(${ITERATIONS})


def load_sample_corpus() -> list[list[str]]:
    corpus: list[list[str]] = []
    root = Path("tests/sample_corpus")
    for txt in sorted(root.glob("*.TXT")):
        for line in txt.read_text().split("\n"):
            if line:
                corpus.append(line.split(" "))
    return corpus

corpus = load_sample_corpus()

configs = [
    ("log_likelihood", {"method": "log_likelihood", "min_count": 0}),
    ("frequency", {"method": "frequency", "min_count": 0}),
    ("npmi", {"method": "npmi", "min_count": 25}),
]

for label, kwargs in configs:
    samples = []
    first_winner = None
    for _ in range(runs):
        t0 = perf_counter()
        winners = remerge.run(corpus, iterations=iterations, progress_bar="none", **kwargs)
        dt = perf_counter() - t0
        samples.append(dt)
        if first_winner is None and winners:
            first_winner = winners[0].merged_lexeme.word
    print(
        f"{label:14s} avg={mean(samples):8.3f}s min={min(samples):8.3f}s max={max(samples):8.3f}s first={first_winner}"
    )
PY

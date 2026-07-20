# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Emit a same-machine unfiltered-core baseline for the fixed POS workload."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter, time

import remerge

from pos_release_fixture import project_authored_workload


TOKENS = 100_000
WARMUPS = 3
REPETITIONS = 15


@dataclass(frozen=True)
class Measurement:
    durations_seconds: tuple[float, ...]
    median_seconds: float
    p95_seconds: float
    iqr_over_median: float


def measure(operation) -> Measurement:
    for _ in range(WARMUPS):
        operation()
    durations: list[float] = []
    for _ in range(REPETITIONS):
        started = perf_counter()
        operation()
        durations.append(perf_counter() - started)
    ordered = sorted(durations)
    median = statistics.median(ordered)
    lower = ordered[: len(ordered) // 2]
    upper = ordered[(len(ordered) + 1) // 2 :]
    iqr = statistics.median(upper) - statistics.median(lower)
    return Measurement(
        tuple(durations),
        median,
        ordered[14],
        iqr / median if median else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    workload = project_authored_workload(TOKENS)

    def operation():
        return remerge.run([workload.raw], 1, method="frequency", min_count=1)

    measurement = measure(operation)
    winner = operation()[0]
    evidence = {
        "schema_version": 1,
        "generated_at_unix": time(),
        "source_revision": args.source_revision,
        "fixture_sha256": workload.digest,
        "winner_signature": [
            list(winner.merged_lexeme.word),
            winner.score,
            winner.merge_token_count,
        ],
        "protocol": {
            "tokens": TOKENS,
            "warmups": WARMUPS,
            "measured_repetitions": REPETITIONS,
        },
        "measurement": asdict(measurement),
        "unfiltered_core_tokens_per_second": TOKENS / measurement.median_seconds,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "power_mode": os.environ.get("REMERGE_POS_BENCHMARK_POWER_MODE"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

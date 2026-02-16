import os
from time import perf_counter

import pytest

from remerge import run


def _timed_frequency_run(
    sample_corpus: list[str], iterations: int
) -> tuple[list, float]:
    start = perf_counter()
    winners = run(sample_corpus, iterations, method="frequency", min_count=0)
    elapsed = perf_counter() - start
    return winners, elapsed


@pytest.mark.fast
@pytest.mark.benchmark(group="performance-guard")
@pytest.mark.skipif(
    os.getenv("REMERGE_PERF_GUARD") != "1",
    reason="Performance guard is disabled. Set REMERGE_PERF_GUARD=1 to enable.",
)
def test_frequency_runtime_guard(sample_corpus, benchmark):
    iterations = int(os.getenv("REMERGE_PERF_ITERATIONS", "5"))
    max_seconds = float(os.getenv("REMERGE_PERF_MAX_SECONDS", "25.0"))

    # Warm up once to avoid counting one-time initialization overhead.
    run(sample_corpus, 1, method="frequency", min_count=0)

    winners, elapsed = benchmark.pedantic(
        _timed_frequency_run,
        args=(sample_corpus, iterations),
        iterations=1,
        rounds=1,
        warmup_rounds=1,
    )
    benchmark.extra_info["elapsed_seconds"] = elapsed
    benchmark.extra_info["limit_seconds"] = max_seconds
    benchmark.extra_info["iterations"] = iterations

    assert winners
    assert elapsed <= max_seconds, (
        f"Performance regression detected: elapsed={elapsed:.3f}s "
        f"(limit={max_seconds:.3f}s, iterations={iterations})."
    )

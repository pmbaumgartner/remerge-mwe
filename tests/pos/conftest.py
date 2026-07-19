"""Shared, project-owned sensors for supplied-tag POS evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter
from typing import Any

import pytest

import remerge


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_719
MEASURED_REPETITIONS = 15
WARMUP_REPETITIONS = 3
REFERENCE_WORKLOAD_TOKENS = 100_000


@dataclass(frozen=True, slots=True)
class GoldDocument:
    """Project-owned evaluation unit; tags stay aligned to explicit words."""

    document_id: str
    domain: str
    sentences: tuple[tuple[remerge.TaggedToken, ...], ...]


@dataclass(frozen=True, slots=True)
class Interval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class TimedMeasurement:
    durations_seconds: tuple[float, ...]
    median_seconds: float
    p95_seconds: float
    iqr_over_median: float


@dataclass(frozen=True, slots=True)
class UtilityMetrics:
    precision_unfiltered: float
    precision_generated: float
    recall_unfiltered: float
    recall_generated: float
    recall_gold_filtered: float
    unfiltered_count: int
    generated_overlap_with_unfiltered: int
    precision_improvement_interval: Interval
    recall_loss_unfiltered_interval: Interval
    recall_loss_gold_interval: Interval


CandidateOccurrence = tuple[int, int, int, int]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("supplied-tag POS release")
    try:
        group.addoption(
            "--pos-pretagged-evidence",
            metavar="PATH",
            help="Required JSON evidence destination for a pretagged release run.",
        )
        group.addoption(
            "--pos-pretagged-core-baseline-evidence",
            metavar="PATH",
            help="Frozen pre-POS core baseline for the pretagged release fixture.",
        )
    except ValueError as error:
        if "--pos-pretagged-evidence" not in str(error):
            raise


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "pos_pretagged_release: opt-in supplied-tag technical release gate",
    )


def raw_documents(documents: Sequence[GoldDocument]) -> tuple[str, ...]:
    """The canonical raw representation used only for end-to-end timing."""

    return tuple(
        "\n".join(
            " ".join(token.form for token in sentence)
            for sentence in document.sentences
        )
        for document in documents
    )


def assert_utility_gates(metrics: UtilityMetrics) -> None:
    """Enforce e6wr's selectivity and recall gates from exact occurrences."""

    failures: list[str] = []
    if metrics.precision_generated - metrics.precision_unfiltered < 0.10:
        failures.append("precision improvement is below 10.0 percentage points")
    if metrics.unfiltered_count <= 0:
        failures.append("unfiltered candidate count is zero")
    elif (
        1 - metrics.generated_overlap_with_unfiltered / metrics.unfiltered_count < 0.10
    ):
        failures.append("generated filter removes less than 10.0% of candidates")
    if metrics.recall_generated < 0.70:
        failures.append("generated-filter recall is below 70.0%")
    if metrics.recall_unfiltered - metrics.recall_generated > 0.05:
        failures.append("generated-filter recall loss exceeds 5.0 points")
    if metrics.recall_gold_filtered - metrics.recall_generated > 0.05:
        failures.append("tagger recall loss exceeds 5.0 points")
    if metrics.precision_improvement_interval.lower <= 0:
        failures.append("precision improvement lower bound is not positive")
    if metrics.recall_loss_unfiltered_interval.upper > 0.05:
        failures.append("unfiltered recall loss upper bound exceeds 5.0 points")
    if metrics.recall_loss_gold_interval.upper > 0.05:
        failures.append("gold-filtered recall loss upper bound exceeds 5.0 points")
    if failures:
        raise AssertionError(
            "POS utility gate rejected candidate: " + "; ".join(failures)
        )


def exact_occurrences(winners: Sequence[Any]) -> set[CandidateOccurrence]:
    """Read structured POS result coordinates; rendered annotations are invalid evidence."""

    occurrences: set[CandidateOccurrence] = set()
    for winner in winners:
        winner_occurrences = getattr(winner, "occurrences", None)
        if winner_occurrences is None:
            raise AssertionError(
                "POS utility evaluation requires WinnerWithOccurrences.occurrences; "
                "do not infer positions from annotation strings"
            )
        for occurrence in winner_occurrences:
            values = (
                occurrence.document_index,
                occurrence.sentence_index,
                occurrence.start_token,
                occurrence.end_token,
            )
            if not all(isinstance(value, int) for value in values):
                raise AssertionError(
                    "POS occurrence diagnostic contains non-integer coordinates"
                )
            occurrences.add(values)
    return occurrences


def calculate_utility_metrics(
    unfiltered: set[CandidateOccurrence],
    generated: set[CandidateOccurrence],
    gold_filtered: set[CandidateOccurrence],
    gold: set[CandidateOccurrence],
    *,
    document_count: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> UtilityMetrics:
    """Score exact MWE spans and cluster all uncertainty by document."""

    if (
        document_count <= 0
        or not gold
        or not unfiltered
        or not generated
        or not gold_filtered
    ):
        raise AssertionError(
            "POS utility evaluation has an empty required occurrence set"
        )

    def values(
        selected_documents: Sequence[int],
    ) -> tuple[float, float, float, float, float, int, int]:
        def resample(source: set[CandidateOccurrence]) -> set[CandidateOccurrence]:
            return {
                (draw, sentence, start, end)
                for draw, document_index in enumerate(selected_documents)
                for source_document, sentence, start, end in source
                if source_document == document_index
            }

        selected_gold = resample(gold)
        selected_unfiltered = resample(unfiltered)
        selected_generated = resample(generated)
        selected_gold_filtered = resample(gold_filtered)
        if (
            not selected_gold
            or not selected_unfiltered
            or not selected_generated
            or not selected_gold_filtered
        ):
            raise AssertionError(
                "bootstrap sample has an empty required occurrence set"
            )
        precision_unfiltered = len(selected_unfiltered & selected_gold) / len(
            selected_unfiltered
        )
        precision_generated = len(selected_generated & selected_gold) / len(
            selected_generated
        )
        recall_unfiltered = len(selected_unfiltered & selected_gold) / len(
            selected_gold
        )
        recall_generated = len(selected_generated & selected_gold) / len(selected_gold)
        recall_gold_filtered = len(selected_gold_filtered & selected_gold) / len(
            selected_gold
        )
        return (
            precision_unfiltered,
            precision_generated,
            recall_unfiltered,
            recall_generated,
            recall_gold_filtered,
            len(selected_unfiltered),
            len(selected_generated & selected_unfiltered),
        )

    point = values(list(range(document_count)))
    rng = random.Random(seed)
    improvements: list[float] = []
    unfiltered_losses: list[float] = []
    gold_losses: list[float] = []
    for _ in range(resamples):
        sample = [rng.randrange(document_count) for _ in range(document_count)]
        sampled = values(sample)
        improvements.append(sampled[1] - sampled[0])
        unfiltered_losses.append(sampled[2] - sampled[3])
        gold_losses.append(sampled[4] - sampled[3])

    def interval(values: list[float]) -> Interval:
        values.sort()
        return Interval(
            values[math.floor(0.025 * (len(values) - 1))],
            values[math.ceil(0.975 * (len(values) - 1))],
        )

    return UtilityMetrics(
        precision_unfiltered=point[0],
        precision_generated=point[1],
        recall_unfiltered=point[2],
        recall_generated=point[3],
        recall_gold_filtered=point[4],
        unfiltered_count=point[5],
        generated_overlap_with_unfiltered=point[6],
        precision_improvement_interval=interval(improvements),
        recall_loss_unfiltered_interval=interval(unfiltered_losses),
        recall_loss_gold_interval=interval(gold_losses),
    )


def project_authored_workload(token_count: int) -> tuple[GoldDocument, ...]:
    """Generate a seed-free, MIT project-authored workload of exactly N tokens."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    cycle = (
        ("bright", "ADJ"),
        ("river", "NOUN"),
        ("flows", "VERB"),
        ("swiftly", "ADV"),
        ("quiet", "ADJ"),
        ("garden", "NOUN"),
        ("grows", "VERB"),
        ("today", "ADV"),
    )
    tokens = [
        remerge.TaggedToken(*cycle[index % len(cycle)]) for index in range(token_count)
    ]
    sentence_size = 32
    sentences = tuple(
        tuple(tokens[offset : offset + sentence_size])
        for offset in range(0, token_count, sentence_size)
    )
    return (GoldDocument("project-authored-0", "project-authored", sentences),)


def workload_digest(documents: Sequence[GoldDocument]) -> str:
    payload = [
        {
            "document_id": document.document_id,
            "domain": document.domain,
            "sentences": [
                [[token.form, token.upos] for token in sentence]
                for sentence in document.sentences
            ],
        }
        for document in documents
    ]
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def timed_measurement(operation: Callable[[], Any]) -> TimedMeasurement:
    for _ in range(WARMUP_REPETITIONS):
        operation()
    durations: list[float] = []
    for _ in range(MEASURED_REPETITIONS):
        started = perf_counter()
        operation()
        durations.append(perf_counter() - started)
    ordered = sorted(durations)
    median = statistics.median(ordered)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    lower_half = ordered[: len(ordered) // 2]
    upper_half = ordered[(len(ordered) + 1) // 2 :]
    iqr = statistics.median(upper_half) - statistics.median(lower_half)
    return TimedMeasurement(
        tuple(durations), median, p95, iqr / median if median else 0.0
    )


def require_single_threaded_environment() -> None:
    required = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    incorrect = [
        f"{name}={os.environ.get(name)!r}"
        for name, value in required.items()
        if os.environ.get(name) != value
    ]
    if incorrect:
        raise AssertionError(
            "single-thread benchmark required; set " + ", ".join(incorrect)
        )


def machine_metadata() -> dict[str, str | int | None]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "compiler": platform.python_compiler(),
        "power_mode": os.environ.get("REMERGE_POS_BENCHMARK_POWER_MODE"),
    }


def write_evidence(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

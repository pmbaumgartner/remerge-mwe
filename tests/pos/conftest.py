"""Shared, project-owned sensors for opt-in POS evaluation.

This module intentionally contains no third-party corpus or model.  It supplies
deterministic fixtures, metric/gate logic, and a narrow adapter protocol so that
an evaluation run cannot silently change token boundaries or its workload.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Protocol

import pytest

import remerge


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_719
MEASURED_REPETITIONS = 15
WARMUP_REPETITIONS = 3
REFERENCE_WORKLOAD_TOKENS = 100_000
SMALL_CALL_TOKENS = 256
MEDIUM_BATCH_TOKENS = 4_096
LONG_SEGMENT_TOKENS = 16_384
UPOS_TAGS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)


@dataclass(frozen=True, slots=True)
class GoldDocument:
    """Project-owned evaluation unit; tags stay aligned to explicit words."""

    document_id: str
    domain: str
    sentences: tuple[tuple[remerge.TaggedToken, ...], ...]


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    overall_accuracy: float
    macro_f1: float
    oov_accuracy: float | None
    ambiguous_accuracy: float | None
    domain_accuracy: dict[str, float]
    class_f1: dict[str, float]


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


class PosTagger(Protocol):
    """Adapter expected by the benchmark command.

    ``tag`` receives the immutable canonical token boundaries.  Returning a
    different form, sentence, document count, or model identity is rejection,
    rather than a request for the harness to re-tokenize.
    """

    model_id: str
    tokenizer_id: str
    artifact_bytes: int
    artifact_sha256: str

    def tag(
        self, documents: tuple[tuple[tuple[str, ...], ...], ...]
    ) -> Sequence[remerge.TaggedDocument]: ...

    def tag_text(
        self, documents: tuple[str, ...]
    ) -> Sequence[remerge.TaggedDocument]: ...


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pos benchmark")
    try:
        group.addoption(
            "--pos-tagger",
            metavar="MODULE:FACTORY",
            help="Zero-argument factory for a conforming POS benchmark adapter.",
        )
        group.addoption(
            "--pos-benchmark-evidence",
            metavar="PATH",
            help="Required JSON evidence destination for an external benchmark run.",
        )
        group.addoption(
            "--pos-benchmark-manifest",
            metavar="PATH",
            help="Frozen wtp0 manifest to validate before a release-quality run.",
        )
        group.addoption(
            "--pos-benchmark-acquisition-root",
            metavar="PATH",
            help="Offline root holding the checksum-verified POS/MWE source files.",
        )
        group.addoption(
            "--pos-candidate-registration",
            metavar="PATH",
            help="Read-only candidate registration created before protected final evaluation.",
        )
        group.addoption(
            "--pos-core-baseline-evidence",
            metavar="PATH",
            help="Frozen pre-POS unfiltered-core baseline evidence for the same fixture.",
        )
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
        if "--pos-tagger" not in str(error):
            raise


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "pos_benchmark: opt-in POS quality and performance acceptance harness",
    )
    config.addinivalue_line(
        "markers",
        "pos_pretagged_release: opt-in supplied-tag technical release gate",
    )


def project_authored_gold_documents() -> tuple[GoldDocument, ...]:
    """Small public control fixture, never a release corpus or quality claim."""

    def sentence(*items: tuple[str, str]) -> tuple[remerge.TaggedToken, ...]:
        return tuple(remerge.TaggedToken(form, upos) for form, upos in items)

    return (
        GoldDocument(
            "control-a",
            "control-a",
            (
                sentence(("bright", "ADJ"), ("river", "NOUN"), ("runs", "VERB")),
                sentence(("record", "VERB"), ("record", "NOUN")),
            ),
        ),
        GoldDocument(
            "control-b",
            "control-b",
            (
                sentence(("quiet", "ADJ"), ("garden", "NOUN"), ("grows", "VERB")),
                sentence(("record", "NOUN"), ("falls", "VERB")),
            ),
        ),
        GoldDocument(
            "control-c",
            "control-c",
            (
                sentence(("swift", "ADJ"), ("fox", "NOUN"), ("leaps", "VERB")),
                sentence(("record", "VERB"), ("spins", "VERB")),
            ),
        ),
    )


def project_authored_training_documents() -> tuple[GoldDocument, ...]:
    def sentence(*items: tuple[str, str]) -> tuple[remerge.TaggedToken, ...]:
        return tuple(remerge.TaggedToken(form, upos) for form, upos in items)

    return (
        GoldDocument(
            "control-train",
            "control-train",
            (
                sentence(("record", "NOUN"), ("bright", "ADJ"), ("river", "NOUN")),
                sentence(("record", "VERB"), ("runs", "VERB")),
            ),
        ),
    )


def canonical_documents(
    documents: Sequence[GoldDocument],
) -> tuple[remerge.TaggedDocument, ...]:
    return tuple(
        remerge.TaggedDocument(sentences=document.sentences) for document in documents
    )


def input_boundaries(
    documents: Sequence[GoldDocument],
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(token.form for token in sentence) for sentence in document.sentences
        )
        for document in documents
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


def validate_tagger_output(
    documents: Sequence[GoldDocument],
    output: Sequence[remerge.TaggedDocument],
) -> tuple[remerge.TaggedDocument, ...]:
    """Reject a tagger result that changes the frozen input alignment."""

    if len(output) != len(documents):
        raise AssertionError("tagger changed the number of documents")

    checked: list[remerge.TaggedDocument] = []
    for document_index, (expected, actual) in enumerate(zip(documents, output)):
        if not isinstance(actual, remerge.TaggedDocument):
            raise AssertionError(
                f"tagger output document {document_index} is not TaggedDocument"
            )
        if len(actual.sentences) != len(expected.sentences):
            raise AssertionError(
                f"tagger changed sentence boundaries in document {document_index}"
            )
        for sentence_index, (expected_sentence, actual_sentence) in enumerate(
            zip(expected.sentences, actual.sentences)
        ):
            if len(actual_sentence) != len(expected_sentence):
                raise AssertionError(
                    "tagger changed token boundaries at "
                    f"document {document_index}, sentence {sentence_index}"
                )
            for token_index, (expected_token, actual_token) in enumerate(
                zip(expected_sentence, actual_sentence)
            ):
                if actual_token.form != expected_token.form:
                    raise AssertionError(
                        "tagger changed token form at "
                        f"document {document_index}, sentence {sentence_index}, "
                        f"token {token_index}"
                    )
                if actual_token.upos not in UPOS_TAGS:
                    raise AssertionError(
                        "tagger emitted an invalid UPOS tag at "
                        f"document {document_index}, sentence {sentence_index}, "
                        f"token {token_index}"
                    )
        checked.append(actual)
    return tuple(checked)


def _iter_tokens(
    documents: Sequence[GoldDocument],
) -> Iterable[tuple[str, str, str, str]]:
    for document in documents:
        for sentence in document.sentences:
            for token in sentence:
                yield document.document_id, document.domain, token.form, token.upos


def _iter_predicted_tokens(
    documents: Sequence[remerge.TaggedDocument],
) -> Iterable[str]:
    for document in documents:
        for sentence in document.sentences:
            yield from (token.upos for token in sentence)


def _accuracy(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(gold == predicted for gold, predicted in pairs) / len(pairs)


def _f1(gold: Sequence[str], predicted: Sequence[str], label: str) -> float:
    true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
    false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
    false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def calculate_quality_metrics(
    training_documents: Sequence[GoldDocument],
    final_documents: Sequence[GoldDocument],
    predicted_documents: Sequence[remerge.TaggedDocument],
) -> QualityMetrics:
    """Calculate the frozen token, OOV, ambiguity, and domain metrics."""

    training_tags: dict[str, set[str]] = {}
    for _document_id, _domain, form, upos in _iter_tokens(training_documents):
        training_tags.setdefault(form, set()).add(upos)

    gold_tokens = list(_iter_tokens(final_documents))
    predicted_tags = list(_iter_predicted_tokens(predicted_documents))
    if len(gold_tokens) != len(predicted_tags):
        raise AssertionError("prediction stream length differs from frozen gold stream")

    gold = [upos for _document, _domain, _form, upos in gold_tokens]
    pairs = list(zip(gold, predicted_tags))
    labels = sorted(set(gold))
    domains = sorted({domain for _document, domain, _form, _upos in gold_tokens})

    by_domain: dict[str, list[tuple[str, str]]] = {domain: [] for domain in domains}
    oov_pairs: list[tuple[str, str]] = []
    ambiguous_pairs: list[tuple[str, str]] = []
    for (_document, domain, form, upos), predicted in zip(gold_tokens, predicted_tags):
        pair = (upos, predicted)
        by_domain[domain].append(pair)
        if form not in training_tags:
            oov_pairs.append(pair)
        if len(training_tags.get(form, set())) >= 2:
            ambiguous_pairs.append(pair)

    return QualityMetrics(
        overall_accuracy=_accuracy(pairs) or 0.0,
        macro_f1=statistics.fmean(_f1(gold, predicted_tags, label) for label in labels),
        oov_accuracy=_accuracy(oov_pairs),
        ambiguous_accuracy=_accuracy(ambiguous_pairs),
        domain_accuracy={
            domain: (_accuracy(domain_pairs) or 0.0)
            for domain, domain_pairs in by_domain.items()
        },
        class_f1={label: _f1(gold, predicted_tags, label) for label in labels},
    )


def document_bootstrap_intervals(
    training_documents: Sequence[GoldDocument],
    final_documents: Sequence[GoldDocument],
    predicted_documents: Sequence[remerge.TaggedDocument],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Interval]:
    """Two-sided 95% document-clustered bootstrap intervals for gate metrics."""

    if len(final_documents) != len(predicted_documents):
        raise AssertionError("cannot bootstrap unequal document streams")
    if not final_documents:
        raise AssertionError("cannot bootstrap an empty final split")

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        "overall_accuracy": [],
        "macro_f1": [],
        "oov_accuracy": [],
        "ambiguous_accuracy": [],
    }
    domains = sorted({document.domain for document in final_documents})
    samples.update({f"domain:{domain}": [] for domain in domains})

    for _ in range(resamples):
        indices = [rng.randrange(len(final_documents)) for _ in final_documents]
        metrics = calculate_quality_metrics(
            training_documents,
            [final_documents[index] for index in indices],
            [predicted_documents[index] for index in indices],
        )
        samples["overall_accuracy"].append(metrics.overall_accuracy)
        samples["macro_f1"].append(metrics.macro_f1)
        if metrics.oov_accuracy is not None:
            samples["oov_accuracy"].append(metrics.oov_accuracy)
        if metrics.ambiguous_accuracy is not None:
            samples["ambiguous_accuracy"].append(metrics.ambiguous_accuracy)
        for domain, value in metrics.domain_accuracy.items():
            samples[f"domain:{domain}"].append(value)

    intervals: dict[str, Interval] = {}
    for name, values in samples.items():
        if not values:
            continue
        values.sort()
        lower_index = math.floor(0.025 * (len(values) - 1))
        upper_index = math.ceil(0.975 * (len(values) - 1))
        intervals[name] = Interval(values[lower_index], values[upper_index])
    return intervals


def assert_quality_gates(
    metrics: QualityMetrics,
    intervals: dict[str, Interval],
    *,
    class_counts: Counter[str],
) -> None:
    """Enforce e6wr's protected quality floors without changing thresholds."""

    failures: list[str] = []
    floors = {
        "overall_accuracy": 0.95,
        "macro_f1": 0.80,
        "oov_accuracy": 0.82,
        "ambiguous_accuracy": 0.88,
    }
    for metric, floor in floors.items():
        interval = intervals.get(metric)
        if interval is None or interval.lower < floor:
            failures.append(f"{metric} lower bound is below {floor:.1%}")
    for domain, accuracy in metrics.domain_accuracy.items():
        interval = intervals.get(f"domain:{domain}")
        if interval is None or interval.lower < 0.90:
            failures.append(f"domain {domain} lower bound is below 90.0%")
        if metrics.overall_accuracy - accuracy > 0.05:
            failures.append(f"domain {domain} is more than 5.0 points below overall")
    for label, count in class_counts.items():
        if count >= 50 and metrics.class_f1[label] < 0.50:
            failures.append(f"UPOS {label} F1 is below 50.0%")
    if failures:
        raise AssertionError(
            "POS quality gate rejected candidate: " + "; ".join(failures)
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
                "POS utility evaluation requires TaggedWinnerInfo.occurrences; "
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


def load_tagger(specification: str) -> PosTagger:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("--pos-tagger must be MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), factory_name)
    tagger = factory()
    for attribute in (
        "model_id",
        "tokenizer_id",
        "artifact_bytes",
        "artifact_sha256",
        "tag",
        "tag_text",
    ):
        if not hasattr(tagger, attribute):
            raise TypeError(
                f"POS tagger adapter lacks required {attribute!r} attribute"
            )
    if not isinstance(tagger.model_id, str) or not tagger.model_id:
        raise TypeError("POS tagger adapter model_id must be a non-empty string")
    if not isinstance(tagger.tokenizer_id, str) or not tagger.tokenizer_id:
        raise TypeError("POS tagger adapter tokenizer_id must be a non-empty string")
    if not isinstance(tagger.artifact_bytes, int) or tagger.artifact_bytes < 0:
        raise TypeError(
            "POS tagger adapter artifact_bytes must be a non-negative integer"
        )
    if (
        not isinstance(tagger.artifact_sha256, str)
        or len(tagger.artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in tagger.artifact_sha256
        )
    ):
        raise TypeError(
            "POS tagger adapter artifact_sha256 must be a lowercase SHA-256"
        )
    return tagger


def isolated_load_measurement(specification: str) -> dict[str, float | int]:
    """Measure model load in a fresh process, separate from warm inference."""

    program = """
import importlib, json, platform, resource
from time import perf_counter
spec = __import__('sys').argv[1]
module_name, factory_name = spec.split(':', 1)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
started = perf_counter()
tagger = getattr(importlib.import_module(module_name), factory_name)()
elapsed = perf_counter() - started
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
unit = 1 if platform.system() == 'Darwin' else 1024
print(json.dumps({'load_seconds': elapsed, 'incremental_peak_rss_bytes': max(0, after - before) * unit, 'artifact_bytes': tagger.artifact_bytes}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, specification],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


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


@pytest.fixture(scope="session")
def pos_benchmark_options(pytestconfig: pytest.Config) -> dict[str, str | None]:
    return {
        "tagger": pytestconfig.getoption("--pos-tagger"),
        "evidence": pytestconfig.getoption("--pos-benchmark-evidence"),
        "manifest": pytestconfig.getoption("--pos-benchmark-manifest"),
        "acquisition_root": pytestconfig.getoption("--pos-benchmark-acquisition-root"),
        "registration": pytestconfig.getoption("--pos-candidate-registration"),
        "core_baseline": pytestconfig.getoption("--pos-core-baseline-evidence"),
    }

"""Technical release gates for supplied-tag POS filtering only."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
from time import perf_counter
from time import time
from typing import Any

import pytest

import remerge
from bin.pos_release_fixture import project_authored_workload as canonical_workload
from tests.pos.conftest import (
    GoldDocument,
    REFERENCE_WORKLOAD_TOKENS,
    UtilityMetrics,
    assert_utility_gates,
    calculate_utility_metrics,
    exact_occurrences,
    machine_metadata,
    project_authored_workload,
    raw_documents,
    require_single_threaded_environment,
    timed_measurement,
    write_evidence,
)


PRE_POS_BASELINE_REVISION = "a26662fa5d7ede6f04cadc370ac55d5a9084e695"
REFERENCE_WORKLOAD_SHA256 = (
    "63d91f8a7316d36c2e8006102bafc36a19049d54b5bd3ecf432d7aeb2cd61427"
)
REFERENCE_WINNER_SIGNATURE = [["bright", "river"], 12_500.0, 12_500]
PATTERNS: list[remerge.PosPattern] = [("ADJ", "NOUN")]
SCALING_PATTERNS: list[remerge.PosPattern] = [
    ("ADJ", "NOUN"),
    ("ADV", "VERB"),
    ("ADJ", "NOUN", "ADV"),
]


def _sentence(*items: tuple[str, str]) -> tuple[remerge.TaggedToken, ...]:
    return tuple(remerge.TaggedToken(form, upos) for form, upos in items)


def _utility_documents() -> tuple[GoldDocument, ...]:
    """Project-owned development control; never protected-final evidence."""

    documents: list[GoldDocument] = []
    for document_index in range(20):
        sentences = tuple(
            _sentence(("new", "ADJ"), ("harbor", "NOUN"), (verb, "VERB"))
            for verb in ("opens", "closes")
            for _ in range(10)
        )
        documents.append(
            GoldDocument(
                f"pretagged-dev-{document_index}",
                "project-authored",
                sentences,
            )
        )
    return tuple(documents)


def _supplied_documents(
    documents: tuple[GoldDocument, ...],
) -> list[remerge.TaggedDocument]:
    return [remerge.TaggedDocument(document.sentences) for document in documents]


def _varied_scaling_documents() -> list[remerge.TaggedDocument]:
    sentences: list[tuple[remerge.TaggedToken, ...]] = []
    for sentence_index in range(5_000):
        variant = sentence_index % 64
        tokens: list[remerge.TaggedToken] = []
        for group in range(5):
            identity = f"{variant}-{group}"
            tokens.extend(
                _sentence(
                    (f"bright-{identity}", "ADJ"),
                    (f"river-{identity}", "NOUN"),
                    (f"swiftly-{identity}", "ADV"),
                    (f"opens-{identity}", "VERB"),
                )
            )
        sentences.append(tuple(tokens))
    return [
        remerge.TaggedDocument(tuple(sentences[offset : offset + 50]))
        for offset in range(0, len(sentences), 50)
    ]


def _multiwinner_scaling() -> list[dict[str, float | int]]:
    documents = _varied_scaling_documents()
    measurements: list[dict[str, float | int]] = []
    for iterations in (1, 8, 32):
        durations = []
        for _ in range(3):
            started = perf_counter()
            winners = remerge.run_tagged(
                documents,
                iterations,
                patterns=SCALING_PATTERNS,
                method="frequency",
                min_count=1,
            )
            durations.append(perf_counter() - started)
            assert len(winners) == iterations
        measurements.append(
            {
                "requested_winners": iterations,
                "median_seconds": statistics.median(durations),
            }
        )
    if measurements[2]["median_seconds"] > 6 * measurements[1]["median_seconds"]:
        raise AssertionError(
            "32-winner POS filtering scales superlinearly relative to 8 winners"
        )
    return measurements


def _utility_metrics(*, resamples: int = 1_001) -> UtilityMetrics:
    documents = _utility_documents()
    unfiltered = remerge.run_with_occurrences(
        list(raw_documents(documents)),
        3,
        method="frequency",
        min_count=1,
    )
    filtered = remerge.run_tagged(
        _supplied_documents(documents),
        3,
        patterns=PATTERNS,
        method="frequency",
        min_count=1,
    )
    gold = {
        (document_index, sentence_index, 0, 2)
        for document_index, document in enumerate(documents)
        for sentence_index, _sentence_value in enumerate(document.sentences)
    }
    return calculate_utility_metrics(
        exact_occurrences(unfiltered),
        exact_occurrences(filtered),
        exact_occurrences(filtered),
        gold,
        document_count=len(documents),
        resamples=resamples,
    )


def _winner_signature(winner: remerge.WinnerInfo) -> list[Any]:
    return [
        list(winner.merged_lexeme.word),
        winner.score,
        winner.merge_token_count,
    ]


def _load_baseline(path: str) -> dict[str, Any]:
    try:
        baseline = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"cannot load pre-POS core baseline: {error}") from error
    if not isinstance(baseline, dict):
        raise AssertionError("pre-POS core baseline must be a JSON object")
    if baseline.get("source_revision") != PRE_POS_BASELINE_REVISION:
        raise AssertionError("core baseline is not from the frozen pre-POS revision")
    if baseline.get("fixture_sha256") != REFERENCE_WORKLOAD_SHA256:
        raise AssertionError(
            "core baseline fixture digest differs from the release fixture"
        )
    throughput = baseline.get("unfiltered_core_tokens_per_second")
    if not isinstance(throughput, (int, float)) or throughput <= 0:
        raise AssertionError("core baseline lacks positive unfiltered throughput")
    if baseline.get("winner_signature") != REFERENCE_WINNER_SIGNATURE:
        raise AssertionError(
            "core baseline winner signature differs from the frozen result"
        )
    return baseline


def test_project_authored_pretagged_utility_fixture_is_useful() -> None:
    metrics = _utility_metrics()
    assert metrics.precision_unfiltered == 0.5
    assert metrics.precision_generated == 1.0
    assert metrics.recall_unfiltered == 1.0
    assert metrics.recall_generated == 1.0
    assert metrics.unfiltered_count == 800
    assert metrics.generated_overlap_with_unfiltered == 400
    assert_utility_gates(metrics)


@pytest.mark.performance
@pytest.mark.pos_pretagged_release
@pytest.mark.skipif(
    os.getenv("REMERGE_POS_PRETAGGED_BENCHMARK") != "1",
    reason="Set REMERGE_POS_PRETAGGED_BENCHMARK=1 to run the supplied-tag gate.",
)
def test_pretagged_release_benchmark(pytestconfig: pytest.Config) -> None:
    evidence_path = pytestconfig.getoption("--pos-pretagged-evidence")
    baseline_path = pytestconfig.getoption("--pos-pretagged-core-baseline-evidence")
    source_revision = os.getenv("REMERGE_POS_SOURCE_REVISION")
    if not evidence_path or not baseline_path:
        pytest.fail(
            "--pos-pretagged-evidence and "
            "--pos-pretagged-core-baseline-evidence are required"
        )
    if (
        source_revision is None
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        pytest.fail("REMERGE_POS_SOURCE_REVISION must be a full lowercase Git SHA")

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "rejected",
        "generated_at_unix": time(),
        "command": " ".join(pytestconfig.invocation_params.args),
        "source_revision": source_revision,
        "machine": machine_metadata(),
        "protected_final_evaluated": False,
        "protocol": {
            "warmups": 3,
            "measured_repetitions": 15,
            "single_thread_required": True,
            "minimum_filtering_tokens_per_second": 100_000,
            "maximum_unfiltered_core_regression": 0.05,
            "maximum_iqr_over_median": 0.10,
        },
    }
    destination = Path(evidence_path)
    try:
        require_single_threaded_environment()
        reference = project_authored_workload(REFERENCE_WORKLOAD_TOKENS)
        fixture_sha256 = canonical_workload(REFERENCE_WORKLOAD_TOKENS).digest
        if fixture_sha256 != REFERENCE_WORKLOAD_SHA256:
            raise AssertionError("project-authored release workload digest changed")
        supplied = _supplied_documents(reference)
        raw = list(raw_documents(reference))
        baseline = _load_baseline(baseline_path)

        filtering = timed_measurement(
            lambda: remerge.run_tagged(
                supplied,
                1,
                patterns=PATTERNS,
                method="frequency",
                min_count=1,
            )
        )
        discovery = timed_measurement(
            lambda: remerge.run(raw, 1, method="frequency", min_count=1)
        )
        current_winners = remerge.run(raw, 1, method="frequency", min_count=1)
        signature = _winner_signature(current_winners[0])
        if signature != REFERENCE_WINNER_SIGNATURE:
            raise AssertionError(
                "unfiltered winner differs from the frozen pre-POS result"
            )

        filtering_rate = REFERENCE_WORKLOAD_TOKENS / filtering.median_seconds
        core_rate = REFERENCE_WORKLOAD_TOKENS / discovery.median_seconds
        multiwinner_scaling = _multiwinner_scaling()
        evidence.update(
            {
                "fixture_sha256": fixture_sha256,
                "winner_signature": signature,
                "frozen_pre_pos_baseline": baseline,
                "filtering": asdict(filtering),
                "unfiltered_core": asdict(discovery),
                "filtering_tokens_per_second": filtering_rate,
                "unfiltered_core_tokens_per_second": core_rate,
                "multiwinner_scaling": multiwinner_scaling,
            }
        )
        failures: list[str] = []
        if filtering_rate < 100_000:
            failures.append("supplied-tag filtering is below 100,000 tokens/s")
        if core_rate < 0.95 * baseline["unfiltered_core_tokens_per_second"]:
            failures.append("unfiltered core regressed more than 5% from pre-POS")
        if filtering.iqr_over_median > 0.10:
            failures.append("supplied-tag filtering IQR/median exceeds 10%")
        if discovery.iqr_over_median > 0.10:
            failures.append("unfiltered core IQR/median exceeds 10%")
        if failures:
            raise AssertionError(
                "pretagged release gate rejected: " + "; ".join(failures)
            )

        utility = _utility_metrics()
        assert_utility_gates(utility)
        evidence.update({"status": "accepted", "utility": asdict(utility)})
    except Exception as error:
        evidence["failure"] = f"{type(error).__name__}: {error}"
        write_evidence(destination, evidence)
        raise
    write_evidence(destination, evidence)

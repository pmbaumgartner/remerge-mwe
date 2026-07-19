"""Opt-in acceptance harness for a conforming POS tagger adapter.

Run the project-owned self-tests with normal pytest.  A candidate/release run
is opt-in and requires the command documented in ``docs/pos_benchmark_protocol.md``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
from time import time
from typing import Any

import pytest

import remerge
from tests.pos.conftest import (
    BOOTSTRAP_RESAMPLES,
    GoldDocument,
    LONG_SEGMENT_TOKENS,
    MEDIUM_BATCH_TOKENS,
    REFERENCE_WORKLOAD_TOKENS,
    SMALL_CALL_TOKENS,
    Interval,
    UtilityMetrics,
    assert_quality_gates,
    assert_utility_gates,
    calculate_utility_metrics,
    calculate_quality_metrics,
    canonical_documents,
    document_bootstrap_intervals,
    exact_occurrences,
    input_boundaries,
    isolated_load_measurement,
    load_tagger,
    machine_metadata,
    project_authored_gold_documents,
    project_authored_training_documents,
    project_authored_workload,
    raw_documents,
    require_single_threaded_environment,
    timed_measurement,
    validate_tagger_output,
    workload_digest,
    write_evidence,
)


def _class_counts(documents) -> Counter[str]:
    return Counter(
        token.upos
        for document in documents
        for sentence in document.sentences
        for token in sentence
    )


def _rotated_predictions(documents):
    tags = ("ADJ", "NOUN", "VERB")
    replacement = {tag: tags[(index + 1) % len(tags)] for index, tag in enumerate(tags)}
    return tuple(
        remerge.TaggedDocument(
            sentences=tuple(
                tuple(
                    remerge.TaggedToken(token.form, replacement.get(token.upos, "X"))
                    for token in sentence
                )
                for sentence in document.sentences
            ),
            source="builtin",
            model_id="rotated-control",
        )
        for document in documents
    )


def test_project_owned_quality_metric_split_and_alignment_sensors() -> None:
    """Known-good and deliberately broken controls prove the oracle can reject."""

    training_documents = project_authored_training_documents()
    gold_documents = project_authored_gold_documents()
    known_good = canonical_documents(gold_documents, model_id="known-good-control")
    checked = validate_tagger_output(
        gold_documents,
        known_good,
        expected_model_id="known-good-control",
    )
    metrics = calculate_quality_metrics(
        training_documents,
        gold_documents,
        checked,
    )
    assert metrics.overall_accuracy == 1.0
    assert metrics.oov_accuracy == 1.0
    assert metrics.ambiguous_accuracy == 1.0

    first = document_bootstrap_intervals(
        training_documents,
        gold_documents,
        checked,
        resamples=101,
    )
    second = document_bootstrap_intervals(
        training_documents,
        gold_documents,
        checked,
        resamples=101,
    )
    assert first == second

    rotated = _rotated_predictions(gold_documents)
    broken_metrics = calculate_quality_metrics(
        training_documents,
        gold_documents,
        rotated,
    )
    broken_intervals = document_bootstrap_intervals(
        training_documents,
        gold_documents,
        rotated,
        resamples=101,
    )
    with pytest.raises(AssertionError, match="quality gate rejected"):
        assert_quality_gates(
            broken_metrics,
            broken_intervals,
            class_counts=_class_counts(gold_documents),
        )


def test_project_owned_utility_sensor_rejects_a_noop_filter() -> None:
    noop = UtilityMetrics(
        precision_unfiltered=0.40,
        precision_generated=0.40,
        recall_unfiltered=0.90,
        recall_generated=0.90,
        recall_gold_filtered=0.90,
        unfiltered_count=300,
        generated_overlap_with_unfiltered=300,
        precision_improvement_interval=Interval(0.0, 0.0),
        recall_loss_unfiltered_interval=Interval(0.0, 0.0),
        recall_loss_gold_interval=Interval(0.0, 0.0),
    )
    with pytest.raises(AssertionError, match="utility gate rejected"):
        assert_utility_gates(noop)


@pytest.mark.parametrize("token_count", [SMALL_CALL_TOKENS, MEDIUM_BATCH_TOKENS])
def test_project_authored_workload_is_deterministic(token_count: int) -> None:
    first = project_authored_workload(token_count)
    second = project_authored_workload(token_count)
    assert workload_digest(first) == workload_digest(second)
    assert (
        sum(len(sentence) for document in first for sentence in document.sentences)
        == token_count
    )


def _manifest_from_path(path: str) -> Any:
    """Load the wtp0 manifest through its project-owned validator only."""

    try:
        from tests.pos.evaluation.loader import load_manifest
    except ImportError as error:  # pragma: no cover - prevents accidental fallback.
        raise AssertionError(
            "POS manifest loader is unavailable; use the committed wtp0 loader."
        ) from error
    manifest = load_manifest(Path(path))
    if manifest.get("release_adequate") is not True:
        raise AssertionError(
            "manifest is not release adequate; the project-authored control fixture "
            "cannot be used as final evaluation data"
        )
    return manifest


def _documents_from_sentences(sentences: Any) -> tuple[GoldDocument, ...]:
    documents: dict[str, list[Any]] = {}
    for sentence in sentences:
        documents.setdefault(sentence.document_id, []).append(sentence)
    return tuple(
        GoldDocument(
            document_id,
            document_sentences[0].tokens[0].domain,
            tuple(
                tuple(
                    remerge.TaggedToken(token.form, token.upos)
                    for token in sentence.tokens
                )
                for sentence in document_sentences
            ),
        )
        for document_id, document_sentences in documents.items()
    )


def _load_frozen_splits(manifest: Any, acquisition_root: Path):
    try:
        from tests.pos.evaluation.loader import load_final_gold, load_tagged_split
    except ImportError as error:  # pragma: no cover - protects the release boundary.
        raise AssertionError("POS final-gold loader is unavailable") from error
    train_sentences = load_tagged_split(manifest, acquisition_root, "train")
    final_gold = load_final_gold(manifest, acquisition_root, allow_final=True)
    final_documents = _documents_from_sentences(final_gold.sentences)
    train_documents = _documents_from_sentences(train_sentences)
    coordinates: dict[tuple[str, str], tuple[int, int]] = {}
    for document_index, document in enumerate(final_documents):
        sentence_index = 0
        for sentence in final_gold.sentences:
            if sentence.document_id == document.document_id:
                coordinates[(document.document_id, sentence.sentence_id)] = (
                    document_index,
                    sentence_index,
                )
                sentence_index += 1
    gold_spans = {
        (
            *coordinates[(span.document_id, span.sentence_id)],
            span.start_token,
            span.end_token,
        )
        for span in final_gold.mwe_spans
    }
    return train_documents, final_documents, gold_spans


def _require_occurrence_diagnostics(winners: list[Any]) -> None:
    if any(not hasattr(winner, "occurrences") for winner in winners):
        raise AssertionError(
            "POS benchmark requires TaggedWinnerInfo.occurrences exact canonical "
            "coordinates; upgrade the POS discovery interface before evaluation."
        )


def _matches_pattern(tags: tuple[str, ...], pattern: remerge.PosPattern) -> bool:
    if len(tags) != len(pattern):
        return False
    for tag, position in zip(tags, pattern):
        if isinstance(position, frozenset):
            if tag not in position:
                return False
        elif position != "*" and tag != position:
            return False
    return True


def _in_scope_gold_spans(
    documents: tuple[GoldDocument, ...],
    spans: set[tuple[int, int, int, int]],
    patterns: list[remerge.PosPattern],
) -> set[tuple[int, int, int, int]]:
    in_scope: set[tuple[int, int, int, int]] = set()
    for document_index, sentence_index, start, end in spans:
        tags = tuple(
            token.upos
            for token in documents[document_index].sentences[sentence_index][start:end]
        )
        if any(_matches_pattern(tags, pattern) for pattern in patterns):
            in_scope.add((document_index, sentence_index, start, end))
    if not in_scope:
        raise AssertionError("frozen final manifest has no in-scope MWE spans")
    return in_scope


def _load_object(path: str, *, description: str) -> dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"cannot load {description}: {error}") from error
    if not isinstance(loaded, dict):
        raise AssertionError(f"{description} must be a JSON object")
    return loaded


def _validate_candidate_registration(path: str, tagger: Any) -> dict[str, Any]:
    registration = _load_object(path, description="candidate registration")
    required = {
        "candidate_id",
        "source_revision",
        "trainer_command",
        "seed",
        "model_id",
        "artifact_sha256",
        "dev_report",
        "final_evaluated",
    }
    missing = required - set(registration)
    if missing:
        raise AssertionError(f"candidate registration is missing {sorted(missing)}")
    if registration["model_id"] != tagger.model_id:
        raise AssertionError(
            "candidate registration model_id differs from tagger adapter"
        )
    if registration["artifact_sha256"] != tagger.artifact_sha256:
        raise AssertionError(
            "candidate registration digest differs from tagger adapter"
        )
    if registration["final_evaluated"] is not False:
        raise AssertionError(
            "candidate has already consumed its one protected final run"
        )
    return registration


def _validate_core_baseline(path: str, workload_sha256: str) -> dict[str, Any]:
    baseline = _load_object(path, description="pre-POS core baseline evidence")
    if baseline.get("fixture_sha256") != workload_sha256:
        raise AssertionError("pre-POS core baseline uses a different fixture digest")
    throughput = baseline.get("unfiltered_core_tokens_per_second")
    if not isinstance(throughput, (int, float)) or throughput <= 0:
        raise AssertionError("pre-POS core baseline lacks a positive throughput")
    return baseline


def _raw_to_filtered(
    tagger: Any, workload, *, patterns: list[remerge.PosPattern]
) -> list[Any]:
    tagged = validate_tagger_output(
        workload,
        tagger.tag_text(raw_documents(workload)),
        expected_model_id=tagger.model_id,
    )
    winners = remerge.run_tagged(
        list(tagged), 1, patterns=patterns, method="frequency", min_count=1
    )
    _require_occurrence_diagnostics(winners)
    return winners


@pytest.mark.performance
@pytest.mark.pos_benchmark
@pytest.mark.skipif(
    os.getenv("REMERGE_POS_BENCHMARK") != "1",
    reason="POS benchmark is opt-in. Set REMERGE_POS_BENCHMARK=1 to enable.",
)
def test_pos_tagger_release_benchmark(pytestconfig: pytest.Config) -> None:
    """Evaluate an external tagger and emit accepting or rejecting JSON evidence."""

    specification = pytestconfig.getoption("--pos-tagger")
    evidence_path = pytestconfig.getoption("--pos-benchmark-evidence")
    manifest_path = pytestconfig.getoption("--pos-benchmark-manifest")
    acquisition_root = pytestconfig.getoption("--pos-benchmark-acquisition-root")
    registration_path = pytestconfig.getoption("--pos-candidate-registration")
    baseline_path = pytestconfig.getoption("--pos-core-baseline-evidence")
    if (
        not specification
        or not evidence_path
        or not manifest_path
        or not acquisition_root
        or not registration_path
        or not baseline_path
    ):
        pytest.fail(
            "--pos-tagger, --pos-benchmark-evidence, --pos-benchmark-manifest, and "
            "--pos-benchmark-acquisition-root, --pos-candidate-registration, and "
            "--pos-core-baseline-evidence are required for a release benchmark"
        )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "command": " ".join(pytestconfig.invocation_params.args),
        "generated_at_unix": time(),
        "machine": machine_metadata(),
        "protocol": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "repetitions": 15,
            "warmups": 3,
            "single_thread_required": True,
        },
        "status": "rejected",
    }
    destination = Path(evidence_path)
    try:
        require_single_threaded_environment()
        manifest = _manifest_from_path(manifest_path)
        tagger = load_tagger(specification)
        evidence["tagger"] = {
            "model_id": tagger.model_id,
            "tokenizer_id": tagger.tokenizer_id,
            "artifact_bytes": tagger.artifact_bytes,
            "artifact_sha256": tagger.artifact_sha256,
        }
        evidence["candidate_registration"] = _validate_candidate_registration(
            registration_path, tagger
        )

        training, final, gold_spans = _load_frozen_splits(
            manifest, Path(acquisition_root)
        )
        tagged_final = validate_tagger_output(
            final,
            tagger.tag(input_boundaries(final)),
            expected_model_id=tagger.model_id,
        )
        quality = calculate_quality_metrics(training, final, tagged_final)
        intervals = document_bootstrap_intervals(training, final, tagged_final)
        assert_quality_gates(quality, intervals, class_counts=_class_counts(final))
        evidence["quality"] = {
            "metrics": asdict(quality),
            "intervals": {name: asdict(value) for name, value in intervals.items()},
        }

        filter_configuration = manifest["filter_configuration"]
        if not isinstance(filter_configuration, dict):
            raise AssertionError("manifest filter_configuration must be an object")
        raw_patterns = filter_configuration.get("patterns")
        if not isinstance(raw_patterns, list):
            raise AssertionError(
                "manifest filter_configuration.patterns must be a list"
            )
        patterns: list[remerge.PosPattern] = [
            tuple(str(position) for position in pattern)
            for pattern in raw_patterns
            if isinstance(pattern, list)
        ]
        if len(patterns) != len(raw_patterns):
            raise AssertionError("manifest POS patterns must be position lists")
        score_method = filter_configuration.get("score_method")
        if not isinstance(score_method, str):
            raise AssertionError("manifest score_method must be a string")
        method = "log_likelihood" if score_method == "ll" else score_method
        iterations = filter_configuration.get("requested_winners")
        min_count = filter_configuration.get("min_count")
        min_score = filter_configuration.get("min_score")
        on_exhausted = filter_configuration.get("on_exhausted")
        if not isinstance(iterations, int) or not isinstance(min_count, int):
            raise AssertionError(
                "manifest requested_winners and min_count must be integers"
            )
        if not isinstance(min_score, (int, float)):
            raise AssertionError("manifest min_score must be numeric")
        if not isinstance(on_exhausted, str):
            raise AssertionError("manifest on_exhausted must be a string")
        run_with_occurrences = getattr(remerge, "run_with_occurrences", None)
        if run_with_occurrences is None:
            raise AssertionError(
                "POS utility evaluation requires remerge.run_with_occurrences exact "
                "unfiltered coordinates; rendered annotations are not valid evidence"
            )
        unfiltered_winners = run_with_occurrences(
            list(raw_documents(final)),
            iterations,
            method=method,
            min_count=min_count,
            min_score=float(min_score),
            on_exhausted=on_exhausted,
        )
        gold_winners = remerge.run_tagged(
            list(canonical_documents(final, model_id="gold-final")),
            iterations,
            patterns=patterns,
            method=method,
            min_count=min_count,
            min_score=float(min_score),
            on_exhausted=on_exhausted,
        )
        generated_winners = remerge.run_tagged(
            list(tagged_final),
            iterations,
            patterns=patterns,
            method=method,
            min_count=min_count,
            min_score=float(min_score),
            on_exhausted=on_exhausted,
        )
        utility = calculate_utility_metrics(
            exact_occurrences(unfiltered_winners),
            exact_occurrences(generated_winners),
            exact_occurrences(gold_winners),
            _in_scope_gold_spans(final, gold_spans, patterns),
            document_count=len(final),
        )
        assert_utility_gates(utility)
        evidence["utility"] = asdict(utility)

        # The manifest is intentionally responsible for protected gold MWE
        # labels and split provenance.  This harness does not read labels until
        # its loader declares a release-adequate, frozen evaluation set.
        workload_shapes = {
            "small": project_authored_workload(SMALL_CALL_TOKENS),
            "medium": project_authored_workload(MEDIUM_BATCH_TOKENS),
            "long": project_authored_workload(LONG_SEGMENT_TOKENS),
            "reference": project_authored_workload(REFERENCE_WORKLOAD_TOKENS),
        }
        reference = workload_shapes["reference"]
        reference_sha256 = workload_digest(reference)
        core_baseline = _validate_core_baseline(baseline_path, reference_sha256)
        patterns: list[remerge.PosPattern] = [("ADJ", "NOUN")]
        tagged_reference = validate_tagger_output(
            reference,
            tagger.tag(input_boundaries(reference)),
            expected_model_id=tagger.model_id,
        )
        inference = timed_measurement(lambda: tagger.tag(input_boundaries(reference)))
        filtering = timed_measurement(
            lambda: remerge.run_tagged(
                list(tagged_reference),
                1,
                patterns=patterns,
                method="frequency",
                min_count=1,
            )
        )
        raw = raw_documents(reference)
        discovery = timed_measurement(
            lambda: remerge.run(list(raw), 1, method="frequency", min_count=1)
        )
        end_to_end = timed_measurement(
            lambda: _raw_to_filtered(tagger, reference, patterns=patterns)
        )
        small_end_to_end = timed_measurement(
            lambda: _raw_to_filtered(
                tagger, workload_shapes["small"], patterns=patterns
            )
        )
        cold_loads = [isolated_load_measurement(specification) for _ in range(15)]
        cold_seconds = sorted(float(item["load_seconds"]) for item in cold_loads)
        cold_rss = sorted(
            int(item["incremental_peak_rss_bytes"]) for item in cold_loads
        )

        inference_rate = REFERENCE_WORKLOAD_TOKENS / inference.median_seconds
        end_to_end_rate = REFERENCE_WORKLOAD_TOKENS / end_to_end.median_seconds
        core_rate = REFERENCE_WORKLOAD_TOKENS / discovery.median_seconds
        failures: list[str] = []
        if tagger.artifact_bytes > 20 * 1024 * 1024:
            failures.append("compressed model artifact exceeds 20 MiB")
        if max(cold_rss) > 128 * 1024 * 1024:
            failures.append("incremental cold-load peak RSS exceeds 128 MiB")
        if statistics.median(cold_seconds) > 0.250:
            failures.append("cold-load median exceeds 250 ms")
        if inference_rate < 50_000:
            failures.append("inference-only throughput is below 50,000 tokens/s")
        if end_to_end_rate < 35_000:
            failures.append("end-to-end throughput is below 35,000 tokens/s")
        if end_to_end_rate < 0.35 * core_rate:
            failures.append("end-to-end throughput is below 35% of unfiltered core")
        if core_rate < 0.95 * core_baseline["unfiltered_core_tokens_per_second"]:
            failures.append(
                "unfiltered core regressed more than 5% from frozen baseline"
            )
        if small_end_to_end.p95_seconds > 0.015:
            failures.append("small-call warm end-to-end p95 exceeds 15 ms")
        for name, measurement in {
            "inference": inference,
            "filtering": filtering,
            "discovery": discovery,
            "end_to_end": end_to_end,
            "small_end_to_end": small_end_to_end,
        }.items():
            if measurement.iqr_over_median > 0.10:
                failures.append(f"{name} IQR/median exceeds 10.0%")
        if failures:
            raise AssertionError(
                "POS resource gate rejected candidate: " + "; ".join(failures)
            )

        evidence["workloads"] = {
            name: {
                "tokens": sum(len(s) for d in docs for s in d.sentences),
                "sha256": workload_digest(docs),
            }
            for name, docs in workload_shapes.items()
        }
        evidence["performance"] = {
            "frozen_pre_pos_baseline": core_baseline,
            "cold_load_seconds": cold_seconds,
            "cold_incremental_peak_rss_bytes": cold_rss,
            "inference": asdict(inference),
            "filtering": asdict(filtering),
            "discovery": asdict(discovery),
            "end_to_end": asdict(end_to_end),
            "small_end_to_end": asdict(small_end_to_end),
            "inference_tokens_per_second": inference_rate,
            "end_to_end_tokens_per_second": end_to_end_rate,
            "unfiltered_core_tokens_per_second": core_rate,
        }
        evidence["status"] = "accepted"
    except Exception as error:
        evidence["failure"] = f"{type(error).__name__}: {error}"
        write_evidence(destination, evidence)
        raise
    write_evidence(destination, evidence)

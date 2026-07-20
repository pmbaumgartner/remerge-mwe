"""Focused controls for the development-only data sufficiency study."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from tests.pos.evaluation.loader import Sentence, Token


MODULE_PATH = (
    Path(__file__).parents[2]
    / "experiments"
    / "pos-tagger"
    / "evaluate_data_sufficiency.py"
)
SPEC = importlib.util.spec_from_file_location("pos_data_sufficiency", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _sentence(document: str, form: str, tag: str, domain: str) -> Sentence:
    return Sentence(
        document,
        document + "-s1",
        (Token(document, document + "-s1", 0, form, tag, domain),),
    )


def test_stratified_sample_is_deterministic_and_keeps_complete_documents() -> None:
    sentences = (
        _sentence("answers-a", "one", "NOUN", "answers"),
        _sentence("answers-b", "two", "VERB", "answers"),
        _sentence("reviews-a", "three", "NOUN", "reviews"),
        _sentence("reviews-b", "four", "VERB", "reviews"),
    )
    domains = study._domain_documents(sentences)

    first, first_summary = study._sample_stratified(
        domains, fraction=0.5, target_tokens=None, seed=7
    )
    second, second_summary = study._sample_stratified(
        domains, fraction=0.5, target_tokens=None, seed=7
    )

    assert first == second
    assert first_summary == second_summary
    assert {sentence.document_id for sentence in first} <= {
        sentence.document_id for sentence in sentences
    }
    assert set(first_summary) == {"answers", "reviews"}


def test_error_slices_use_only_the_condition_training_sample() -> None:
    train = (_sentence("answers-a", "known", "NOUN", "answers"),)
    dev = (
        _sentence("answers-b", "known", "NOUN", "answers"),
        _sentence("reviews-a", "novel", "VERB", "reviews"),
    )
    predictions = ((("NOUN",),), (("NOUN",),))

    slices = study._error_slices(train, dev, predictions)

    assert slices["token_frequency"]["1"] == {"count": 1, "accuracy": 1.0}
    assert slices["token_frequency"]["0"] == {"count": 1, "accuracy": 0.0}
    assert slices["tag_support"]["1-9"] == {"count": 1, "accuracy": 1.0}
    assert slices["tag_support"]["0"] == {"count": 1, "accuracy": 0.0}


def test_diagnosis_prefers_inconclusive_when_data_signals_conflict() -> None:
    conditions = []
    for kind, fraction, held_out, accuracy in (
        ("learning_curve", 0.75, None, 0.80),
        ("learning_curve", 1.00, None, 0.90),
        ("mixed_control", 1.00, "reviews", 0.95),
        ("leave_one_domain_out", 1.00, "reviews", 0.80),
    ):
        for seed in (1, 2):
            conditions.append(
                {
                    "kind": kind,
                    "fraction": fraction,
                    "held_out_domain": held_out,
                    "model_seed": seed,
                    "quality": {
                        "overall_accuracy": accuracy,
                        "oov_accuracy": accuracy,
                    },
                    "held_out_domain_accuracy": accuracy,
                    "error_slices": {"token_frequency": {}, "tag_support": {}},
                }
            )

    assert (
        study._diagnosis(conditions, (0.75, 1.0), calibration=False)["diagnosis"]
        == "inconclusive"
    )


def test_mixed_control_uses_all_domains_at_the_leave_out_requested_budget() -> None:
    jobs = study._jobs(
        (1.0,),
        replicates=1,
        seed=7,
        domain_tokens={"answers": 20, "reviews": 80},
        diversity_fraction=1.0,
        include_diversity=True,
        mwe_fractions=(),
    )
    leave_out = next(item for item in jobs if item["id"] == "loo-answers-r0")
    mixed = next(item for item in jobs if item["id"] == "mixed-answers-r0")

    assert leave_out["requested_token_budget"] == mixed["requested_token_budget"] == 80
    assert leave_out["sample_fraction"] == 1.0
    assert mixed["sample_fraction"] == 0.8


def test_matched_diversity_pair_records_actual_budget_delta_and_rejects_large_gap() -> (
    None
):
    def document(name: str, count: int, domain: str) -> tuple[Sentence, ...]:
        return tuple(
            _sentence(name, f"{name}-{index}", "NOUN", domain) for index in range(count)
        )

    train = document("answers", 5, "answers") + document("reviews", 7, "reviews")
    jobs = study._jobs(
        (1.0,),
        replicates=1,
        seed=7,
        domain_tokens={"answers": 5, "reviews": 7},
        diversity_fraction=1.0,
        include_diversity=True,
        mwe_fractions=(),
    )
    planned = study._plan_conditions(jobs, train, match_tolerance_fraction=0.0)
    mixed = next(item for item in planned if item["id"] == "mixed-answers-r0")
    leave_out = next(item for item in planned if item["id"] == "loo-answers-r0")

    assert leave_out["planned_actual_tokens"] == 7
    assert mixed["matching"]["target_tokens"] == 7
    assert mixed["matching"]["actual_delta_tokens"] == 5
    assert mixed["matching"]["status"] == "unmatched"


def test_repeated_sample_t_interval_is_not_a_percentile_interval() -> None:
    interval = study._t_interval((1.0, 2.0, 3.0, 4.0, 5.0))

    assert interval["method"] == "two-sided Student-t 95% interval across repetitions"
    assert interval["lower"] == pytest.approx(1.0371, abs=0.0001)
    assert interval["upper"] == pytest.approx(4.9629, abs=0.0001)


def test_negative_interval_is_not_treated_as_flat() -> None:
    assert (
        study._interval_state({"lower": 0.003, "upper": 0.004, "count": 5})
        == "positive"
    )
    assert study._interval_state({"lower": 0.0, "upper": 0.002, "count": 5}) == "flat"
    assert (
        study._interval_state({"lower": -0.001, "upper": 0.001, "count": 5}) == "wide"
    )
    assert (
        study._interval_state({"lower": -0.002, "upper": -0.001, "count": 5})
        == "negative"
    )
    assert (
        study._interval_state({"lower": None, "upper": None, "count": 1})
        == "unavailable"
    )


def test_calibration_diagnosis_is_always_unavailable() -> None:
    diagnosis = study._diagnosis((), (1.0,), calibration=True)

    assert diagnosis["diagnosis"] == "unavailable"
    assert diagnosis["interval_states"] == {}


def test_absent_concentration_bucket_does_not_block_diagnosis() -> None:
    conditions = []
    for seed in range(5):
        for fraction, accuracy in ((0.75, 0.80), (1.0, 0.81)):
            conditions.append(
                {
                    "kind": "learning_curve",
                    "fraction": fraction,
                    "held_out_domain": None,
                    "model_seed": seed,
                    "quality": {
                        "overall_accuracy": accuracy,
                        "oov_accuracy": accuracy - 0.10,
                    },
                    "error_slices": {"token_frequency": {}, "tag_support": {}},
                }
            )
        for kind in ("mixed_control", "leave_one_domain_out"):
            conditions.append(
                {
                    "kind": kind,
                    "fraction": 1.0,
                    "held_out_domain": "reviews",
                    "model_seed": seed,
                    "quality": {"overall_accuracy": 0.80, "oov_accuracy": 0.70},
                    "held_out_domain_accuracy": 0.80,
                    "matching": {"status": "matched"},
                    "error_slices": {"token_frequency": {}, "tag_support": {}},
                }
            )

    diagnosis = study._diagnosis(conditions, (0.75, 1.0), calibration=False)

    assert diagnosis["diagnosis"] == "volume_limited"
    assert diagnosis["interval_states"]["concentration"][
        "tag_support_0_error_excess"
    ] == ("unavailable")
    assert diagnosis["interval_states"]["applicable_concentration"] == [
        "oov_error_excess"
    ]


def test_planned_study_diagnosis_is_unavailable() -> None:
    diagnosis = study._diagnosis((), study.DEFAULT_FRACTIONS, calibration=False)

    assert diagnosis["diagnosis"] == "unavailable"
    assert diagnosis["interval_states"] == {}


def test_interval_rejects_more_than_thirty_repetitions() -> None:
    with pytest.raises(study.StudyError, match="at most 30"):
        study._t_interval([1.0] * 31)


def test_fraction_index_keeps_close_fraction_condition_ids_distinct() -> None:
    jobs = study._jobs(
        (0.051, 0.052),
        replicates=1,
        seed=7,
        domain_tokens={"reviews": 10},
        diversity_fraction=1.0,
        include_diversity=False,
        mwe_fractions=(),
    )

    assert len({job["id"] for job in jobs}) == 2


def test_evidence_integrity_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "evaluator.py"
    source.write_text("initial", encoding="utf-8")
    integrity = {
        "paths": {"evaluator": str(source)},
        "hashes": {"evaluator": study._sha(source)},
    }
    source.write_text("changed", encoding="utf-8")

    with pytest.raises(study.StudyError, match="drift"):
        study._verify_evidence_integrity(integrity)


def test_aggregate_contains_slice_tag_domain_and_mwe_uncertainty() -> None:
    conditions = []
    for seed in (1, 2):
        conditions.append(
            {
                "id": f"curve-{seed}",
                "kind": "learning_curve",
                "fraction": 1.0,
                "model_seed": seed,
                "sample": {"tokens": 10},
                "quality": {
                    "overall_accuracy": 0.9,
                    "oov_accuracy": 0.8,
                    "ambiguous_accuracy": 0.7,
                    "macro_f1": 0.6,
                    "per_domain_accuracy": {"reviews": 0.85},
                    "supported_tag_f1": {"NOUN": {"f1": 0.75, "count": 8}},
                },
                "error_slices": {
                    "token_frequency": {"0": {"accuracy": 0.5, "count": 2}},
                    "tag_support": {"0": {"accuracy": 0.4, "count": 1}},
                },
                "mwe_utility": {
                    "precision_generated": 0.5,
                    "unfiltered_count": 4,
                    "precision_improvement_interval": {"lower": 0.1, "upper": 0.2},
                },
            }
        )

    aggregate = study._aggregate(conditions)[0]

    assert aggregate["quality"]["per_domain_accuracy.reviews"]["count"] == 2
    assert aggregate["quality"]["supported_tag_f1.NOUN.f1"]["count"] == 2
    assert aggregate["error_slices"]["token_frequency.0.accuracy"]["count"] == 2
    assert aggregate["error_slices"]["tag_support.0.accuracy"]["count"] == 2
    assert aggregate["mwe_utility"]["precision_generated"]["count"] == 2
    assert (
        aggregate["mwe_utility"]["precision_improvement_interval.lower"]["count"] == 2
    )


def test_study_command_has_no_protected_split_option() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'add_argument("--final"' not in source


def test_dirty_normal_run_writes_machine_readable_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "rejected.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--acquisition-root",
            str(tmp_path / "data"),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        study,
        "_require_clean_implementation",
        lambda: (_ for _ in ()).throw(study.StudyError("dirty checkout")),
    )

    with pytest.raises(SystemExit, match="2"):
        study.main()

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "failure": "StudyError: dirty checkout",
        "protected_final_evaluated": False,
        "status": "rejected",
    }

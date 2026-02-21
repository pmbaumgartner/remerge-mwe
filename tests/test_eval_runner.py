from dataclasses import dataclass
from pathlib import Path
import shutil

import pytest

from scripts.mwe_eval.run_eval import EvalRunConfig, _normalize_run_spec, evaluate_run


@dataclass(frozen=True)
class _DummyLexeme:
    word: tuple[str, ...]


@dataclass(frozen=True)
class _DummyWinner:
    merged_lexeme: _DummyLexeme


def _copy_fixture(src_name: str, dst: Path) -> None:
    src = Path("tests/eval_fixtures") / src_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


@pytest.mark.fast
def test_evaluate_run_reports_expected_counts_for_parseme(monkeypatch, tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"
    _copy_fixture(
        "parseme_sample.cupt",
        data_root / "parseme" / "sharedtask-data" / "1.2" / "EN" / "dev.cupt",
    )

    def fake_run(*_args, **_kwargs):
        return [
            _DummyWinner(_DummyLexeme(("make", "up"))),
            _DummyWinner(_DummyLexeme(("give", "up"))),
        ]

    monkeypatch.setattr("scripts.mwe_eval.run_eval.remerge_run", fake_run)

    config = EvalRunConfig(dataset="parseme12", split="dev", iterations=5)
    result = evaluate_run(config, data_root=data_root, download_missing=False)

    assert result["counts"]["gold_total"] == 3
    assert result["counts"]["gold_contiguous"] == 2
    assert result["counts"]["gold_discontinuous"] == 1
    assert result["counts"]["predicted_total"] == 1
    assert result["counts"]["matched"] == 1
    assert result["metrics"]["mention"]["recall"] == 0.5


@pytest.mark.fast
def test_evaluate_run_all_returns_per_dataset_results(monkeypatch, tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"

    _copy_fixture(
        "parseme_sample.cupt",
        data_root / "parseme" / "sharedtask-data" / "1.2" / "EN" / "dev.cupt",
    )
    _copy_fixture(
        "streusle_sample.json",
        data_root / "streusle" / "streusle-5.0" / "dev.json",
    )
    _copy_fixture(
        "coam_sample.json",
        data_root / "coam" / "coam_sample.json",
    )

    def fake_run(*_args, **_kwargs):
        return [
            _DummyWinner(_DummyLexeme(("make", "up"))),
            _DummyWinner(_DummyLexeme(("give", "up"))),
            _DummyWinner(_DummyLexeme(("kick", "the", "bucket"))),
        ]

    monkeypatch.setattr("scripts.mwe_eval.run_eval.remerge_run", fake_run)

    config = EvalRunConfig(dataset="all", split="dev", iterations=5)
    result = evaluate_run(config, data_root=data_root, download_missing=False)

    assert result["dataset"] == "all"
    assert len(result["results"]) == 3
    assert result["summary"]["counts"]["gold_total"] >= 1


@pytest.mark.fast
def test_evaluate_run_decode_policy_changes_predicted_mentions(monkeypatch, tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"
    _copy_fixture(
        "parseme_sample.cupt",
        data_root / "parseme" / "sharedtask-data" / "1.2" / "EN" / "dev.cupt",
    )

    def fake_run(*_args, **_kwargs):
        return [
            _DummyWinner(_DummyLexeme(("make", "up"))),
            _DummyWinner(_DummyLexeme(("make", "up", "for"))),
        ]

    monkeypatch.setattr("scripts.mwe_eval.run_eval.remerge_run", fake_run)

    all_config = EvalRunConfig(
        dataset="parseme12",
        split="dev",
        iterations=5,
        decode_policy="all",
    )
    longest_config = EvalRunConfig(
        dataset="parseme12",
        split="dev",
        iterations=5,
        decode_policy="longest_per_start",
    )

    all_result = evaluate_run(all_config, data_root=data_root, download_missing=False)
    longest_result = evaluate_run(
        longest_config,
        data_root=data_root,
        download_missing=False,
    )

    assert (
        all_result["counts"]["predicted_total"]
        >= longest_result["counts"]["predicted_total"]
    )


@pytest.mark.fast
def test_evaluate_run_forwards_new_core_arguments(monkeypatch, tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"
    _copy_fixture(
        "parseme_sample.cupt",
        data_root / "parseme" / "sharedtask-data" / "1.2" / "EN" / "dev.cupt",
    )

    captured_kwargs: list[dict[str, object]] = []

    def fake_run(*_args, **kwargs):
        captured_kwargs.append(kwargs)
        return [_DummyWinner(_DummyLexeme(("make", "up")))]

    monkeypatch.setattr("scripts.mwe_eval.run_eval.remerge_run", fake_run)

    config = EvalRunConfig(
        dataset="parseme12",
        split="dev",
        iterations=3,
        min_range=2,
        range_alpha=0.5,
        min_p_ab=0.2,
        min_p_ba=0.3,
        min_merge_count=2,
        min_winner_score_output=0.0,
        min_winner_range_output=2,
        search_strategy="beam",
        beam_width=3,
        beam_top_m=5,
        consensus_runs=({"method": "frequency"}, {"method": "npmi"}),
        consensus_min_run_support=2,
        consensus_min_method_support=1,
    )
    evaluate_run(config, data_root=data_root, download_missing=False)

    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs["min_range"] == 2
    assert kwargs["range_alpha"] == 0.5
    assert kwargs["min_p_ab"] == 0.2
    assert kwargs["min_p_ba"] == 0.3
    assert kwargs["min_merge_count"] == 2
    assert kwargs["min_winner_score_output"] == 0.0
    assert kwargs["min_winner_range_output"] == 2
    assert kwargs["search_strategy"] == "beam"
    assert kwargs["beam_width"] == 3
    assert kwargs["beam_top_m"] == 5
    assert kwargs["consensus_runs"] == [
        {"method": "frequency"},
        {"method": "npmi"},
    ]
    assert kwargs["consensus_min_run_support"] == 2
    assert kwargs["consensus_min_method_support"] == 1


@pytest.mark.fast
def test_normalize_run_spec_new_fields_and_validation():
    normalized = _normalize_run_spec(
        {
            "dataset": "parseme12",
            "split": "dev",
            "iterations": 1,
            "min_range": 2,
            "range_alpha": 0.25,
            "min_p_ab": 0.2,
            "min_p_ba": 0.3,
            "min_merge_count": 2,
            "search_strategy": "beam",
            "beam_width": 2,
            "beam_top_m": 4,
            "consensus_methods": ["frequency", "npmi"],
            "decode_policy": "maximal_non_overlapping",
            "suppress_subspan_types": True,
        }
    )
    assert normalized.min_range == 2
    assert normalized.range_alpha == 0.25
    assert normalized.min_p_ab == 0.2
    assert normalized.min_p_ba == 0.3
    assert normalized.min_merge_count == 2
    assert normalized.search_strategy == "beam"
    assert normalized.beam_width == 2
    assert normalized.beam_top_m == 4
    assert normalized.consensus_runs == (
        {"method": "frequency"},
        {"method": "npmi"},
    )
    assert normalized.decode_policy == "maximal_non_overlapping"
    assert normalized.suppress_subspan_types is True

    with pytest.raises(ValueError, match="min_p_ab"):
        _normalize_run_spec(
            {
                "dataset": "parseme12",
                "split": "dev",
                "iterations": 1,
                "min_p_ab": 1.1,
            }
        )

    with pytest.raises(ValueError, match="beam_width"):
        _normalize_run_spec(
            {
                "dataset": "parseme12",
                "split": "dev",
                "iterations": 1,
                "search_strategy": "beam",
                "beam_width": 0,
            }
        )

    with pytest.raises(ValueError, match="decode_policy"):
        _normalize_run_spec(
            {
                "dataset": "parseme12",
                "split": "dev",
                "iterations": 1,
                "decode_policy": "not-valid",
            }
        )

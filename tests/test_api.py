import importlib.metadata
import inspect

import pytest
from remerge import __version__, annotate, run
from remerge.core import Lexeme, Splitter


def test_version_matches_installed_metadata():
    assert __version__ == importlib.metadata.version("remerge-mwe")


def test_winner_shape():
    winners = run(["a b c"], 1, method="frequency")
    winner = winners[0]
    assert winner.bigram[0] == Lexeme(("a",), 0)
    assert winner.bigram[1] == Lexeme(("b",), 0)
    assert winner.merged_lexeme == Lexeme(("a", "b"), 0)
    assert winner.n_lexemes == 2
    assert isinstance(winner.score, float)
    assert winner.merge_token_count == 1


def test_lexeme_and_winner_string_helpers():
    lexeme = Lexeme(("a", "b"), 0)
    assert str(lexeme) == "a b"
    assert lexeme.text == "a b"
    assert lexeme.token_count == 2

    winner = run(["a b c"], 1, method="frequency")[0]
    assert str(winner) == "a b"
    assert winner.text == "a b"
    assert winner.token_count == 2
    assert winner.n_lexemes == 2


def test_root_exports_include_types():
    from remerge import (
        Bigram,
        Lexeme as RootLexeme,
        WinnerInfo as RootWinnerInfo,
        WinnerWithOccurrences,
    )

    assert RootLexeme is Lexeme
    assert RootWinnerInfo is not None
    assert WinnerWithOccurrences is not None
    assert Bigram is not None


def test_run_and_annotate_common_signatures_are_aligned():
    run_params = list(inspect.signature(run).parameters.values())
    annotate_params = list(inspect.signature(annotate).parameters.values())
    annotate_common = annotate_params[: len(run_params)]

    assert [param.name for param in annotate_common] == [
        param.name for param in run_params
    ]
    for run_param, annotate_param in zip(run_params, annotate_common):
        assert run_param.kind == annotate_param.kind
        assert run_param.default == annotate_param.default


@pytest.mark.parametrize(
    "method,min_count",
    [
        ("frequency", 0),
        ("log_likelihood", 0),
        ("npmi", 0),
    ],
)
def test_run_progress_preserves_results(method, min_count, capsys):
    corpus = ["a b a c a b a c", "a b a c"]
    baseline = run(corpus, 3, method=method, min_count=min_count)
    capsys.readouterr()
    progress_winners = run(
        corpus,
        3,
        method=method,
        min_count=min_count,
        progress=True,
    )
    stderr = capsys.readouterr().err

    assert progress_winners == baseline
    if progress_winners:
        assert "remerge progress:" in stderr
        assert f"{len(progress_winners)}/3" in stderr
    else:
        assert stderr == ""


def test_annotate_progress_preserves_results(capsys):
    corpus = ["a b a b c d c d"]
    baseline = annotate(corpus, 2, method="frequency")
    capsys.readouterr()
    progress_result = annotate(
        corpus,
        2,
        method="frequency",
        progress=True,
    )
    stderr = capsys.readouterr().err

    assert progress_result == baseline
    if progress_result[0]:
        assert "remerge progress:" in stderr
        assert f"{len(progress_result[0])}/2" in stderr
    else:
        assert stderr == ""


def test_invalid_splitter_raises():
    with pytest.raises(ValueError):
        run(["a b"], 1, splitter="not-a-splitter")


def test_rescore_interval_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], 1, rescore_interval=0)

    with pytest.raises(ValueError):
        annotate(["a b a b"], 1, rescore_interval=0)


def test_iterations_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], -1)

    with pytest.raises(ValueError):
        annotate(["a b a b"], -1)


def test_progress_validation_for_run_and_annotate():
    with pytest.raises(TypeError):
        run(["a b a b"], 1, progress=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        annotate(["a b a b"], 1, progress=1)  # type: ignore[arg-type]


def test_min_count_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], 1, min_count=-1)

    with pytest.raises(ValueError):
        annotate(["a b a b"], 1, min_count=-1)


def test_sentencex_language_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], 1, splitter=Splitter.sentencex, sentencex_language="   ")

    with pytest.raises(ValueError):
        annotate(
            ["a b a b"],
            1,
            splitter=Splitter.sentencex,
            sentencex_language="   ",
        )

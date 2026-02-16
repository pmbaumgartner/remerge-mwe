import json
from pathlib import Path

import pytest
from remerge import __version__, run
from remerge.core import Lexeme, NoCandidateBigramError, SelectionMethod, Splitter


def _summarize_winners(winners):
    return [
        {
            "merged_word": list(winner.merged_lexeme.word),
            "merge_token_count": winner.merge_token_count,
        }
        for winner in winners
    ]


def test_version():
    assert __version__ == "0.2.1"


@pytest.mark.corpus
def test_sample_corpus_single_iter(sample_corpus):
    winners = run(sample_corpus, 1)
    assert winners[0].merged_lexeme == Lexeme(("you", "know"), 0)


@pytest.mark.fast
def test_winner_shape():
    winners = run(["a b c"], 1, method="frequency")
    winner = winners[0]
    assert winner.bigram[0] == Lexeme(("a",), 0)
    assert winner.bigram[1] == Lexeme(("b",), 0)
    assert winner.merged_lexeme == Lexeme(("a", "b"), 0)
    assert winner.n_lexemes == 2
    assert winner.merge_token_count == 1


@pytest.mark.fast
def test_consecutive_single():
    """Ensure greedy bigram merge avoids overlapping middle bigram."""
    corpus = ["a a a a"]
    winners = run(corpus, 2)
    assert winners[0].merge_token_count == 2
    assert winners[1].merged_lexeme == Lexeme(("a", "a", "a", "a"), 0)


@pytest.mark.fast
def test_consecutive_remainder():
    """Ensure greedy bigram merge avoids overlapping trailing bigram."""
    corpus = ["c a b a b a b d"]
    winners = run(corpus, 2, method="frequency")
    assert winners[0].merge_token_count == 3
    assert winners[1].merge_token_count == 1


@pytest.mark.fast
@pytest.mark.parametrize(
    "method,min_count",
    [
        (SelectionMethod.log_likelihood, 0),
        (SelectionMethod.npmi, 0),
        (SelectionMethod.frequency, 0),
        ("frequency", 0),
    ],
)
def test_methods(method, min_count):
    corpus = ["a b a b c", "a b d e"]
    winners = run(corpus, 2, method=method, min_count=min_count)
    assert len(winners) == 2


@pytest.mark.fast
def test_output(tmp_path):
    output = tmp_path / "tmp.json"
    winners = run(["a b a b"], 1, output=output)
    results = json.loads(output.read_text())
    assert tuple(results["0"]) == winners[0].merged_lexeme.word


@pytest.mark.fast
def test_output_writes_once_by_default(monkeypatch, tmp_path):
    output = tmp_path / "tmp.json"
    corpus = ["a a a a"]
    writes = []
    original_write_text = Path.write_text

    def _capture_write(path_obj, text, *args, **kwargs):
        if path_obj == output:
            writes.append(text)
        return original_write_text(path_obj, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _capture_write)
    winners = run(corpus, 2, output=output)

    assert len(writes) == 1
    expected = {str(j): list(w.merged_lexeme.word) for j, w in enumerate(winners)}
    assert json.loads(writes[0]) == expected
    assert json.loads(output.read_text()) == expected


@pytest.mark.fast
def test_output_debug_writes_each_iteration_with_cumulative_content(
    monkeypatch, tmp_path
):
    output = tmp_path / "tmp.json"
    corpus = ["a a a a"]
    writes = []
    original_write_text = Path.write_text

    def _capture_write(path_obj, text, *args, **kwargs):
        if path_obj == output:
            writes.append(text)
        return original_write_text(path_obj, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _capture_write)
    winners = run(
        corpus,
        2,
        output=output,
        output_debug_each_iteration=True,
    )

    assert len(writes) == len(winners)
    for i, payload in enumerate(writes):
        decoded = json.loads(payload)
        expected = {
            str(j): list(w.merged_lexeme.word) for j, w in enumerate(winners[: i + 1])
        }
        assert decoded == expected


@pytest.mark.fast
def test_empty_or_single_token_corpus_stops_cleanly():
    assert run([], 1) == []
    assert run(["a"], 5) == []


@pytest.mark.fast
def test_exhausted_policy_raise():
    with pytest.raises(NoCandidateBigramError):
        run(["a"], 1, on_exhausted="raise")


@pytest.mark.fast
def test_iterations_larger_than_available_merges_stop():
    winners = run(["a b c"], 99)
    assert len(winners) == 2


@pytest.mark.fast
def test_frequency_respects_min_count():
    corpus = ["a b", "a c"]
    winners = run(
        corpus,
        1,
        method="frequency",
        min_count=2,
        on_exhausted="stop",
    )
    assert winners == []


@pytest.mark.fast
@pytest.mark.parametrize("method", ["frequency", "log_likelihood", "npmi"])
def test_deterministic_tie_breaking_is_order_independent(method):
    corpus_a = ["a b", "c d"]
    corpus_b = ["c d", "a b"]

    winner_a = run(corpus_a, 1, method=method)[0].merged_lexeme.word
    winner_b = run(corpus_b, 1, method=method)[0].merged_lexeme.word
    assert winner_a == winner_b == ("a", "b")


@pytest.mark.fast
def test_legacy_tie_breaking_keeps_first_seen_behavior():
    corpus_a = ["a b", "c d"]
    corpus_b = ["c d", "a b"]

    winner_a = run(
        corpus_a,
        1,
        method="frequency",
        tie_breaker="legacy_first_seen",
    )[0].merged_lexeme.word
    winner_b = run(
        corpus_b,
        1,
        method="frequency",
        tie_breaker="legacy_first_seen",
    )[0].merged_lexeme.word

    assert winner_a != winner_b


@pytest.mark.fast
def test_min_score_stops_or_raises():
    corpus = ["a b c"]
    assert run(corpus, 2, min_score=1e9, on_exhausted="stop") == []

    with pytest.raises(NoCandidateBigramError):
        run(corpus, 2, min_score=1e9, on_exhausted="raise")


@pytest.mark.fast
def test_sentencex_splitter_breaks_cross_sentence_bigrams():
    corpus = ["hi! bye! hi! bye!"]

    baseline = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.delimiter,
        line_delimiter=None,
    )
    assert baseline[0].merged_lexeme.word == ("hi!", "bye!")

    sentencex = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.sentencex,
        sentencex_language="en",
    )
    assert sentencex == []


@pytest.mark.fast
def test_invalid_splitter_raises():
    with pytest.raises(ValueError):
        run(["a b"], 1, splitter="not-a-splitter")


@pytest.mark.fast
def test_sentencex_defaults_to_en_language():
    corpus = ["hi! bye! hi! bye!"]

    implicit_en = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.sentencex,
    )
    explicit_en = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.sentencex,
        sentencex_language="en",
    )

    assert implicit_en == explicit_en == []


@pytest.mark.fast
def test_sentencex_ignores_line_delimiter():
    corpus = ["hi! bye! hi! bye!"]

    with_none = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.sentencex,
        line_delimiter=None,
    )
    with_custom = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.sentencex,
        line_delimiter="__this_delimiter_is_ignored__",
    )

    assert with_none == with_custom == []


@pytest.mark.parity
def test_parity_fixture(sample_corpus):
    parity = json.loads(Path("tests/parity_expected.json").read_text())

    ll = run(sample_corpus, 5, method="log_likelihood", min_count=0)
    freq = run(sample_corpus, 5, method="frequency", min_count=0)
    npmi = run(sample_corpus, 5, method="npmi", min_count=25)

    assert _summarize_winners(ll) == parity["log_likelihood"]
    assert _summarize_winners(freq) == parity["frequency"]
    assert _summarize_winners(npmi) == parity["npmi"]

    tie_corpus = ["a b", "c d"]
    for method, expected in parity["tie_deterministic"].items():
        winner = run(tie_corpus, 1, method=method)[0]
        assert list(winner.merged_lexeme.word) == expected

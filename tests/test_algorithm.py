import pytest
from remerge import annotate, run
from remerge.core import (
    ExhaustionPolicy,
    Lexeme,
    NoCandidateBigramError,
    SelectionMethod,
)


def test_consecutive_single():
    """Ensure greedy bigram merge avoids overlapping middle bigram."""
    corpus = ["a a a a"]
    winners = run(corpus, 2)
    assert winners[0].merge_token_count == 2
    assert winners[1].merged_lexeme == Lexeme(("a", "a", "a", "a"), 0)


def test_consecutive_remainder():
    """Ensure greedy bigram merge avoids overlapping trailing bigram."""
    corpus = ["c a b a b a b d"]
    winners = run(corpus, 2, method="frequency")
    assert winners[0].merge_token_count == 3
    assert winners[1].merge_token_count == 1


@pytest.mark.parametrize(
    "enum_method,string_method",
    [
        (SelectionMethod.log_likelihood, "log_likelihood"),
        (SelectionMethod.npmi, "npmi"),
        (SelectionMethod.frequency, "frequency"),
    ],
)
def test_selection_method_enum_and_string_are_equivalent(enum_method, string_method):
    corpus = ["a b a b c", "a b d e"]

    enum_winners = run(corpus, 2, method=enum_method, min_count=0)
    string_winners = run(corpus, 2, method=string_method, min_count=0)

    assert len(enum_winners) == 2
    assert enum_winners == string_winners


def test_empty_or_single_token_corpus_stops_cleanly():
    assert run([], 1) == []
    assert run(["a"], 5) == []


def test_exhausted_policy_raise_with_string_value():
    with pytest.raises(NoCandidateBigramError):
        run(["a"], 1, on_exhausted="raise")


def test_exhausted_policy_raise_with_enum_value():
    with pytest.raises(NoCandidateBigramError):
        run(["a"], 1, on_exhausted=ExhaustionPolicy.raise_)


def test_iterations_larger_than_available_merges_stop():
    winners = run(["a b c"], 99)
    assert len(winners) == 2


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


@pytest.mark.parametrize("method", ["frequency", "log_likelihood", "npmi"])
def test_deterministic_tie_breaking_is_order_independent(method):
    corpus_a = ["a b", "c d"]
    corpus_b = ["c d", "a b"]

    winner_a = run(corpus_a, 1, method=method)[0].merged_lexeme.word
    winner_b = run(corpus_b, 1, method=method)[0].merged_lexeme.word
    assert winner_a == winner_b == ("a", "b")


def test_min_score_stops_or_raises():
    corpus = ["a b c"]
    assert run(corpus, 2, min_score=1e9, on_exhausted="stop") == []

    with pytest.raises(NoCandidateBigramError):
        run(corpus, 2, min_score=1e9, on_exhausted="raise")


@pytest.mark.parametrize(
    "method",
    [SelectionMethod.log_likelihood, SelectionMethod.npmi],
)
def test_rescore_interval_one_preserves_run_annotate_winner_equivalence(method):
    corpus = ["a b a c a b a c", "a b a c"]
    run_winners = run(corpus, 3, method=method, min_count=0, rescore_interval=1)
    annotate_winners, _annotated, _labels = annotate(
        corpus,
        3,
        method=method,
        min_count=0,
        rescore_interval=1,
    )

    assert len(run_winners) == 3
    assert annotate_winners == run_winners
    assert all(
        winner.merged_lexeme.word == winner.bigram[0].word + winner.bigram[1].word
        for winner in run_winners
    )


def test_run_can_build_mwes_longer_than_two_tokens():
    tokens = ("a", "b", "c", "d") * 3
    winners = run([" ".join(tokens)], 6, method="frequency")

    assert winners[-1].merged_lexeme == Lexeme(tokens, 0)

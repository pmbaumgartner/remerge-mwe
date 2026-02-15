import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from src.remerge import __version__, run
from src.remerge.core import (
    BigramCandidateArrays,
    BigramData,
    Lexeme,
    LexemeData,
    NoCandidateBigramError,
    ScoredBigram,
    SelectionMethod,
    TieBreaker,
    WinnerInfo,
    _calculate_log_likelihood,
    _calculate_npmi,
    _select_candidate,
    calculate_candidates_frequency,
    merge_winner,
)
def test_version():
    assert __version__ == "0.2.1"


def test_single_iter(sample_corpus):
    winners = run(sample_corpus, 1, progress_bar="none")
    assert winners[0].merged_lexeme == Lexeme(("you", "know"), 0)


def test_consecutive_single():
    """Ensure greedy bigram merge avoids overlapping middle bigram."""
    corpus = [["a", "a", "a", "a"]]
    winners = run(corpus, 2, progress_bar="none")
    assert winners[0].merge_token_count == 2  # count == 3 is incorrect
    assert winners[1].merged_lexeme == Lexeme(("a", "a", "a", "a"), 0)


def test_consecutive_remainder():
    """Ensure greedy bigram merge avoids overlapping trailing bigram."""
    corpus = [["c", "a", "b", "a", "b", "a", "b", "d"]]
    winners = run(corpus, 2, method="frequency", progress_bar="none")
    assert winners[0].merge_token_count == 3
    assert winners[1].merge_token_count == 1


@pytest.mark.parametrize("progress_bar", ["all", "iterations", "none"])
def test_progress_bar(progress_bar, sample_corpus):
    winners = run(sample_corpus, 2, progress_bar=progress_bar)
    assert len(winners) == 2


@pytest.mark.parametrize(
    "method,min_count",
    [
        (SelectionMethod.log_likelihood, 0),
        (SelectionMethod.npmi, 25),
        (SelectionMethod.frequency, 0),
        ("frequency", 0),
    ],
)
def test_methods(method, min_count, sample_corpus):
    winners = run(sample_corpus, 2, method=method, min_count=min_count, progress_bar="none")
    assert len(winners) == 2


def test_output(sample_corpus, tmp_path):
    output = tmp_path / "tmp.json"
    winners = run(sample_corpus, 1, output=output, progress_bar="none")
    results = json.loads(output.read_text())
    # json keys are strings
    assert tuple(results["0"]) == winners[0].merged_lexeme.word


def test_output_writes_each_iteration_with_cumulative_content(monkeypatch, tmp_path):
    output = tmp_path / "tmp.json"
    corpus = [["a", "a", "a", "a"]]
    writes = []
    original_write_text = Path.write_text

    def _capture_write(path_obj, text, *args, **kwargs):
        if path_obj == output:
            writes.append(text)
        return original_write_text(path_obj, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _capture_write)
    winners = run(corpus, 2, output=output, progress_bar="none")

    assert len(writes) == len(winners)
    for i, payload in enumerate(writes):
        decoded = json.loads(payload)
        expected = {str(j): list(w.merged_lexeme.word) for j, w in enumerate(winners[: i + 1])}
        assert decoded == expected
    assert json.loads(output.read_text()) == json.loads(writes[-1])


def test_empty_or_single_token_corpus_stops_cleanly():
    assert run([], 1, progress_bar="none") == []
    assert run([["a"]], 5, progress_bar="none") == []


def test_exhausted_policy_raise():
    with pytest.raises(NoCandidateBigramError):
        run([["a"]], 1, on_exhausted="raise", progress_bar="none")


def test_iterations_larger_than_available_merges_stop():
    winners = run([["a", "b", "c"]], 99, progress_bar="none")
    assert len(winners) == 2


def test_frequency_respects_min_count():
    corpus = [["a", "b"], ["a", "c"]]
    winners = run(
        corpus,
        1,
        method="frequency",
        min_count=2,
        on_exhausted="stop",
        progress_bar="none",
    )
    assert winners == []


def test_ll_npmi_arrays_stay_finite_with_min_count_filter():
    corpus = [["a", "b"], ["a", "c"], ["a", "b"]]
    lexemes = LexemeData.from_corpus(corpus)
    bigrams = BigramData.from_lexemes(lexemes)
    data = BigramCandidateArrays.from_bigram_data(bigrams, min_count=2)

    obs_a = data.bigram_freq_array
    obs_b = data.el1_freq_array - obs_a
    obs_c = data.el2_freq_array - obs_a
    obs_d = data.total_bigram_count - obs_a - obs_b - obs_c
    assert np.all(obs_d >= 0)

    ll = _calculate_log_likelihood(data)
    npmi = _calculate_npmi(data)
    assert np.isfinite(ll).all()
    assert np.isfinite(npmi).all()


@pytest.mark.parametrize("method", ["frequency", "log_likelihood", "npmi"])
def test_deterministic_tie_breaking_is_order_independent(method):
    corpus_a = [["a", "b"], ["c", "d"]]
    corpus_b = [["c", "d"], ["a", "b"]]

    winner_a = run(corpus_a, 1, method=method, progress_bar="none")[0].merged_lexeme.word
    winner_b = run(corpus_b, 1, method=method, progress_bar="none")[0].merged_lexeme.word
    assert winner_a == winner_b == ("a", "b")


def test_legacy_tie_breaking_keeps_first_seen_behavior():
    corpus_a = [["a", "b"], ["c", "d"]]
    corpus_b = [["c", "d"], ["a", "b"]]

    winner_a = run(
        corpus_a,
        1,
        method="frequency",
        tie_breaker="legacy_first_seen",
        progress_bar="none",
    )[0].merged_lexeme.word
    winner_b = run(
        corpus_b,
        1,
        method="frequency",
        tie_breaker="legacy_first_seen",
        progress_bar="none",
    )[0].merged_lexeme.word

    assert winner_a != winner_b


def test_min_score_stops_or_raises():
    corpus = [["a", "b", "c"]]
    assert run(corpus, 2, min_score=1e9, on_exhausted="stop", progress_bar="none") == []

    with pytest.raises(NoCandidateBigramError):
        run(corpus, 2, min_score=1e9, on_exhausted="raise", progress_bar="none")


def test_select_candidate_legacy_first_seen_tolerance_boundary():
    ab = (Lexeme(("a",), 0), Lexeme(("b",), 0))
    cd = (Lexeme(("c",), 0), Lexeme(("d",), 0))
    ef = (Lexeme(("e",), 0), Lexeme(("f",), 0))
    candidates = [
        ScoredBigram(bigram=ab, score=1.0, frequency=1),
        ScoredBigram(bigram=cd, score=1.0 + 5e-13, frequency=9),
        ScoredBigram(bigram=ef, score=1.0 + 3e-12, frequency=10),
    ]
    # max score is ef; cd is not close to max and should not be selected.
    assert _select_candidate(candidates, TieBreaker.legacy_first_seen) == candidates[2]

    close_tie = [
        ScoredBigram(bigram=ab, score=1.0, frequency=1),
        ScoredBigram(bigram=cd, score=1.0 + 5e-13, frequency=9),
    ]
    # both are within tolerance of max score; first seen wins.
    assert _select_candidate(close_tie, TieBreaker.legacy_first_seen) == close_tie[0]


def test_select_candidate_deterministic_tie_break_order():
    ab = (Lexeme(("a",), 0), Lexeme(("b",), 0))
    ac = (Lexeme(("a",), 0), Lexeme(("c",), 0))
    ad = (Lexeme(("a",), 0), Lexeme(("d",), 0))
    tied = [
        ScoredBigram(bigram=ad, score=7.0, frequency=2),
        ScoredBigram(bigram=ac, score=7.0 + 2e-13, frequency=3),
        ScoredBigram(bigram=ab, score=7.0 + 1e-13, frequency=3),
    ]
    # All scores are tied within tolerance. Frequency wins first, then lexicographic merged word.
    selected = _select_candidate(tied, TieBreaker.deterministic)
    assert selected == tied[2]


def _assert_invariants(lexemes: LexemeData, bigrams: BigramData):
    assert all(freq > 0 for freq in lexemes.lexemes_to_freqs.values())
    assert all(freq > 0 for freq in bigrams.bigrams_to_freqs.values())
    assert all(locations for locations in lexemes.lexemes_to_locations.values())
    assert all(locations for locations in bigrams.bigrams_to_locations.values())

    for lexeme, freq in lexemes.lexemes_to_freqs.items():
        assert lexeme.ix == 0
        assert freq == len(lexemes.lexemes_to_locations[lexeme])

    for bigram, freq in bigrams.bigrams_to_freqs.items():
        assert freq == len(bigrams.bigrams_to_locations[bigram])

    expected_left = Counter()
    expected_right = Counter()
    for (left, right), freq in bigrams.bigrams_to_freqs.items():
        expected_left[left] += freq
        expected_right[right] += freq
    assert expected_left == bigrams.left_lex_freqs
    assert expected_right == bigrams.right_lex_freqs


def test_merge_invariants_hold_each_iteration():
    corpus = [["a", "a", "a", "a"], ["c", "a", "b", "a", "b", "a", "b", "d"]]
    lexemes = LexemeData.from_corpus(corpus)
    bigrams = BigramData.from_lexemes(lexemes)

    for _ in range(8):
        candidates = calculate_candidates_frequency(bigrams)
        if not candidates:
            break
        top_frequency = max(candidate.frequency for candidate in candidates)
        tied = [c for c in candidates if c.frequency == top_frequency]
        selected = sorted(tied, key=lambda c: (-c.frequency, c.merged_word))[0]

        winner = WinnerInfo.from_bigram_with_data(selected.bigram, bigrams)
        lexemes, bigrams = merge_winner(winner, lexemes, bigrams)

        assert winner.bigram not in bigrams.bigrams_to_freqs
        _assert_invariants(lexemes, bigrams)


def test_merge_prunes_touched_zero_or_empty_entries():
    corpus = [["a", "a", "a", "a"], ["b", "a", "a", "c"]]
    lexemes = LexemeData.from_corpus(corpus)
    bigrams = BigramData.from_lexemes(lexemes)
    winner = run(corpus, 1, method=SelectionMethod.frequency, progress_bar="none")[0]

    lexemes_after, bigrams_after = merge_winner(winner, lexemes, bigrams)

    assert all(freq > 0 for freq in lexemes_after.lexemes_to_freqs.values())
    assert all(freq > 0 for freq in bigrams_after.bigrams_to_freqs.values())
    assert all(freq > 0 for freq in bigrams_after.left_lex_freqs.values())
    assert all(freq > 0 for freq in bigrams_after.right_lex_freqs.values())
    assert all(lexemes_after.lexemes_to_locations.values())
    assert all(bigrams_after.bigrams_to_locations.values())

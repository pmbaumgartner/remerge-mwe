from dataclasses import dataclass

import pytest

from scripts.mwe_eval.match import (
    build_trie,
    decode_spans,
    find_spans,
    lexicon_from_winners,
    suppress_subspan_types,
)


@dataclass(frozen=True)
class _DummyLexeme:
    word: tuple[str, ...]


@dataclass(frozen=True)
class _DummyWinner:
    merged_lexeme: _DummyLexeme


@pytest.mark.fast
def test_find_spans_supports_nested_and_overlapping_matches():
    trie = build_trie({("a", "b"), ("a", "b", "c"), ("b", "c")})
    spans = find_spans(("a", "b", "c"), trie)
    assert spans == {(0, 2), (0, 3), (1, 3)}


@pytest.mark.fast
def test_decode_spans_longest_per_start_keeps_only_longest_start_match():
    spans = {(0, 2), (0, 3), (1, 3)}
    assert decode_spans(spans, policy="longest_per_start") == {(0, 3), (1, 3)}


@pytest.mark.fast
def test_decode_spans_maximal_non_overlapping_enforces_no_overlap():
    spans = {(0, 2), (0, 3), (1, 4), (4, 6)}
    decoded = decode_spans(spans, policy="maximal_non_overlapping")
    assert decoded == {(0, 3), (4, 6)}


@pytest.mark.fast
def test_suppress_subspan_types_drops_contiguous_subphrases():
    types = {
        ("kick", "the"),
        ("the", "bucket"),
        ("kick", "the", "bucket"),
        ("make", "up"),
    }
    assert suppress_subspan_types(types) == {
        ("kick", "the", "bucket"),
        ("make", "up"),
    }


@pytest.mark.fast
def test_lexicon_from_winners_filters_singletons():
    winners = [
        _DummyWinner(_DummyLexeme(("a",))),
        _DummyWinner(_DummyLexeme(("make", "up"))),
        _DummyWinner(_DummyLexeme(("give", "it", "up"))),
    ]
    assert lexicon_from_winners(winners) == {("make", "up"), ("give", "it", "up")}

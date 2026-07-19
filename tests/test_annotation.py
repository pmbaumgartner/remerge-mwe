import pytest
from remerge import annotate, run
from remerge.core import NoCandidateBigramError, Splitter


def test_annotate_basic_single_mwe():
    winners, annotated, labels = annotate(
        ["a b c a b"],
        1,
        method="frequency",
    )
    assert winners[0].merged_lexeme.word == ("a", "b")
    assert annotated == ["<mwe:a_b> c <mwe:a_b>"]
    assert labels == ["<mwe:a_b>"]


def test_annotate_agglomerative_longest_merge():
    winners, annotated, labels = annotate(
        ["a b a b a b a b"],
        2,
        method="frequency",
    )
    assert winners[1].merged_lexeme.word == ("a", "b", "a", "b")
    assert annotated == ["<mwe:a_b_a_b> <mwe:a_b_a_b>"]
    assert labels == ["<mwe:a_b_a_b>"]


def test_annotate_multiple_distinct_mwes():
    _winners, annotated, labels = annotate(
        ["a b a b c d c d"],
        2,
        method="frequency",
    )
    assert annotated == ["<mwe:a_b> <mwe:a_b> <mwe:c_d> <mwe:c_d>"]
    assert labels == ["<mwe:a_b>", "<mwe:c_d>"]


def test_annotate_preserves_document_count():
    corpus = ["a b a b", "", "c d c d"]
    _winners, annotated, _labels = annotate(corpus, 1, method="frequency")
    assert len(annotated) == len(corpus)


def test_annotate_multiline_delimiter_roundtrip():
    _winners, annotated, labels = annotate(
        ["a b\nc d\na b"],
        1,
        method="frequency",
        splitter=Splitter.delimiter,
        line_delimiter="\n",
    )
    assert annotated == ["<mwe:a_b>\nc d\n<mwe:a_b>"]
    assert labels == ["<mwe:a_b>"]


def test_annotate_custom_formatting():
    _winners, annotated, labels = annotate(
        ["a b c a b"],
        1,
        method="frequency",
        mwe_prefix="[",
        mwe_suffix="]",
        token_separator="+",
    )
    assert annotated == ["[a+b] c [a+b]"]
    assert labels == ["[a+b]"]


def test_annotate_no_matches_passthrough():
    winners, annotated, labels = annotate(["a"], 5, method="frequency")
    assert winners == []
    assert annotated == ["a"]
    assert labels == []


def test_annotate_winners_match_run():
    corpus = ["a b a b c d c d"]
    run_winners = run(
        corpus,
        2,
        method="frequency",
        min_count=0,
        splitter=Splitter.delimiter,
        line_delimiter="\n",
        sentencex_language="en",
        on_exhausted="stop",
        min_score=None,
    )
    annotate_winners, _annotated, _labels = annotate(
        corpus,
        2,
        method="frequency",
        min_count=0,
        splitter=Splitter.delimiter,
        line_delimiter="\n",
        sentencex_language="en",
        on_exhausted="stop",
        min_score=None,
    )
    assert annotate_winners == run_winners


def test_annotate_labels_sorted_deduped():
    _winners, _annotated, labels = annotate(
        ["c d c d a b a b"],
        2,
        method="frequency",
    )
    assert labels == ["<mwe:a_b>", "<mwe:c_d>"]


def test_annotate_exhausted_policy_raise():
    with pytest.raises(NoCandidateBigramError):
        annotate(["a"], 1, on_exhausted="raise")


def test_annotate_min_score_stops_or_raises():
    corpus = ["a b c"]
    winners, annotated, labels = annotate(
        corpus,
        2,
        min_score=1e9,
        on_exhausted="stop",
    )
    assert winners == []
    assert annotated == ["a b c"]
    assert labels == []

    with pytest.raises(NoCandidateBigramError):
        annotate(corpus, 2, min_score=1e9, on_exhausted="raise")


def test_annotate_line_delimiter_none_preserves_document_boundaries():
    _winners, annotated, _labels = annotate(
        ["a b", "c d"],
        1,
        method="frequency",
        splitter=Splitter.delimiter,
        line_delimiter=None,
    )
    assert annotated == ["<mwe:a_b>", "c d"]

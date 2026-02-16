import importlib.metadata
import inspect
import json
from pathlib import Path

import pytest
from remerge import __version__, annotate, run
from remerge.core import (
    Engine,
    ExhaustionPolicy,
    Lexeme,
    NoCandidateBigramError,
    SelectionMethod,
    Splitter,
)


def _summarize_winners(winners):
    return [
        {
            "merged_word": list(winner.merged_lexeme.word),
            "merge_token_count": winner.merge_token_count,
        }
        for winner in winners
    ]


@pytest.mark.fast
def test_version_matches_installed_metadata():
    assert __version__ == importlib.metadata.version("remerge-mwe")


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
    assert isinstance(winner.score, float)
    assert winner.merge_token_count == 1


@pytest.mark.fast
def test_winnerinfo_redundant_location_helpers_removed():
    winner = run(["a b c"], 1, method="frequency")[0]
    assert not hasattr(winner, "bigram_locations")
    assert winner.merge_token_count == 1
    assert not hasattr(winner, "cleaned_bigram_locations")
    assert not hasattr(winner, "clean_bigram_locations")


@pytest.mark.fast
def test_root_exports_include_types():
    from remerge import Bigram, Lexeme as RootLexeme, WinnerInfo as RootWinnerInfo

    assert RootLexeme is Lexeme
    assert RootWinnerInfo is not None
    assert Bigram is not None


@pytest.mark.fast
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
def test_corpus_length_matches_segment_count():
    corpus = ["a b\nc d\ne f"]
    engine = Engine(corpus, "frequency", 0, "delimiter", "\n", "en", 25)
    assert engine.corpus_length() == 3

    engine_default_delim = Engine(corpus, "frequency", 0)
    assert engine_default_delim.corpus_length() == 3

    engine_no_delim = Engine(["a b c"], "frequency", 0, "delimiter", None, "en", 25)
    assert engine_no_delim.corpus_length() == 1


@pytest.mark.fast
def test_removed_output_parameter_raises_type_error():
    with pytest.raises(TypeError):
        run(["a b a b"], 1, output=Path("/tmp/out.json"))  # type: ignore[call-arg]


@pytest.mark.fast
def test_removed_output_debug_parameter_raises_type_error():
    with pytest.raises(TypeError):
        run(["a b a b"], 1, output_debug_each_iteration=True)  # type: ignore[call-arg]


@pytest.mark.fast
def test_removed_tie_breaker_parameter_raises_type_error():
    with pytest.raises(TypeError):
        run(["a b"], 1, tie_breaker="deterministic")  # type: ignore[call-arg]


@pytest.mark.fast
def test_empty_or_single_token_corpus_stops_cleanly():
    assert run([], 1) == []
    assert run(["a"], 5) == []


@pytest.mark.fast
def test_exhausted_policy_raise_with_string_value():
    with pytest.raises(NoCandidateBigramError):
        run(["a"], 1, on_exhausted="raise")


@pytest.mark.fast
def test_exhausted_policy_raise_with_enum_value_string():
    with pytest.raises(NoCandidateBigramError):
        run(["a"], 1, on_exhausted=ExhaustionPolicy.raise_.value)


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
def test_min_score_stops_or_raises():
    corpus = ["a b c"]
    assert run(corpus, 2, min_score=1e9, on_exhausted="stop") == []

    with pytest.raises(NoCandidateBigramError):
        run(corpus, 2, min_score=1e9, on_exhausted="raise")


@pytest.mark.fast
def test_annotate_basic_single_mwe():
    winners, annotated, labels = annotate(
        ["a b c a b"],
        1,
        method="frequency",
    )
    assert winners[0].merged_lexeme.word == ("a", "b")
    assert annotated == ["<mwe:a_b> c <mwe:a_b>"]
    assert labels == ["<mwe:a_b>"]


@pytest.mark.fast
def test_annotate_agglomerative_longest_merge():
    winners, annotated, labels = annotate(
        ["a b a b a b a b"],
        2,
        method="frequency",
    )
    assert winners[1].merged_lexeme.word == ("a", "b", "a", "b")
    assert annotated == ["<mwe:a_b_a_b> <mwe:a_b_a_b>"]
    assert labels == ["<mwe:a_b_a_b>"]


@pytest.mark.fast
def test_annotate_multiple_distinct_mwes():
    _winners, annotated, labels = annotate(
        ["a b a b c d c d"],
        2,
        method="frequency",
    )
    assert annotated == ["<mwe:a_b> <mwe:a_b> <mwe:c_d> <mwe:c_d>"]
    assert labels == ["<mwe:a_b>", "<mwe:c_d>"]


@pytest.mark.fast
def test_annotate_preserves_document_count():
    corpus = ["a b a b", "", "c d c d"]
    _winners, annotated, _labels = annotate(corpus, 1, method="frequency")
    assert len(annotated) == len(corpus)


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
def test_annotate_no_matches_passthrough():
    winners, annotated, labels = annotate(["a"], 5, method="frequency")
    assert winners == []
    assert annotated == ["a"]
    assert labels == []


@pytest.mark.fast
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
def test_annotate_sentencex_splitter():
    winners, annotated, labels = annotate(
        ["hi! bye! hi! bye!"],
        1,
        method="frequency",
        splitter=Splitter.sentencex,
        sentencex_language="en",
    )
    assert winners == []
    assert annotated == ["hi! bye! hi! bye!"]
    assert labels == []


@pytest.mark.fast
def test_annotate_labels_sorted_deduped():
    _winners, _annotated, labels = annotate(
        ["c d c d a b a b"],
        2,
        method="frequency",
    )
    assert labels == ["<mwe:a_b>", "<mwe:c_d>"]


@pytest.mark.fast
def test_annotate_exhausted_policy_raise():
    with pytest.raises(NoCandidateBigramError):
        annotate(["a"], 1, on_exhausted="raise")


@pytest.mark.fast
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


@pytest.mark.fast
def test_annotate_line_delimiter_none_preserves_document_boundaries():
    _winners, annotated, _labels = annotate(
        ["a b", "c d"],
        1,
        method="frequency",
        splitter=Splitter.delimiter,
        line_delimiter=None,
    )
    assert annotated == ["<mwe:a_b>", "c d"]


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


@pytest.mark.fast
def test_rescore_interval_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], 1, rescore_interval=0)

    with pytest.raises(ValueError):
        annotate(["a b a b"], 1, rescore_interval=0)


@pytest.mark.fast
def test_iterations_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], -1)

    with pytest.raises(ValueError):
        annotate(["a b a b"], -1)


@pytest.mark.fast
def test_min_count_validation_for_run_and_annotate():
    with pytest.raises(ValueError):
        run(["a b a b"], 1, min_count=-1)

    with pytest.raises(ValueError):
        annotate(["a b a b"], 1, min_count=-1)


@pytest.mark.fast
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


@pytest.mark.fast
def test_rescore_interval_one_runs_ll_and_npmi():
    corpus = ["a b a c a b a c", "a b a c"]
    ll = run(corpus, 3, method="log_likelihood", rescore_interval=1)
    npmi = run(corpus, 3, method="npmi", min_count=0, rescore_interval=1)

    assert len(ll) > 0
    assert len(npmi) > 0


@pytest.mark.fast
def test_unicode_multibyte_corpus():
    corpus = ["你好 世界 你好 世界 你好 世界", "😀 😃 😀 😃"]
    winners = run(corpus, 1, method="frequency")
    assert winners[0].merged_lexeme.word == ("你好", "世界")


@pytest.mark.fast
def test_pathological_whitespace_corpus_is_handled():
    corpus = ["   \n\t  ", "", "  a   b  \r\n  ", "\n\n"]
    winners = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.delimiter,
        line_delimiter="\n",
    )
    assert winners[0].merged_lexeme.word == ("a", "b")


@pytest.mark.fast
def test_mixed_newline_styles_with_custom_delimiter():
    corpus = ["a b\r\n\r\nc d\r\na b", "a b\r\nx y"]
    winners = run(
        corpus,
        1,
        method="frequency",
        splitter=Splitter.delimiter,
        line_delimiter="\r\n",
    )
    assert winners[0].merged_lexeme.word == ("a", "b")


@pytest.mark.fast
def test_smallvec_spill_path_produces_long_mwe():
    corpus = ["a b c d a b c d a b c d"]
    winners = run(corpus, 6, method="frequency")
    assert any(len(winner.merged_lexeme.word) >= 4 for winner in winners)


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

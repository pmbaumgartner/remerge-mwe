from remerge import annotate, run
from remerge.core import Splitter


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


def test_unicode_multibyte_corpus():
    corpus = ["你好 世界 你好 世界 你好 世界", "😀 😃 😀 😃"]
    winners = run(corpus, 1, method="frequency")
    assert winners[0].merged_lexeme.word == ("你好", "世界")


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

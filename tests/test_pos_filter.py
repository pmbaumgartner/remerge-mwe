import pytest

import remerge


def token(form: str, upos: str) -> remerge.TaggedToken:
    return remerge.TaggedToken(form, upos)


def document(*sentences: tuple[remerge.TaggedToken, ...]) -> remerge.TaggedDocument:
    return remerge.TaggedDocument(sentences=sentences)


def test_mixed_pos_occurrences_are_counted_and_merged_individually() -> None:
    corpus = [
        document(
            (token("record", "VERB"), token("deal", "NOUN")),
            (token("record", "NOUN"), token("deal", "NOUN")),
        )
    ]

    winners, annotated, labels = remerge.annotate_tagged(
        corpus,
        1,
        patterns=[("VERB", "NOUN")],
        method="frequency",
        min_count=1,
    )

    assert [winner.text for winner in winners] == ["record deal"]
    assert winners[0].score == 1
    assert winners[0].merge_token_count == 1
    assert winners[0].occurrences == (remerge.MweOccurrence(0, 0, 0, 2),)
    assert annotated == ["<mwe:record_deal>\nrecord deal"]
    assert labels == ["<mwe:record_deal>"]


def test_matching_occurrence_count_controls_min_count() -> None:
    corpus = [
        document(
            (token("record", "VERB"), token("deal", "NOUN")),
            (token("record", "NOUN"), token("deal", "NOUN")),
        )
    ]

    assert (
        remerge.run_tagged(
            corpus,
            1,
            patterns=[("VERB", "NOUN")],
            method="frequency",
            min_count=2,
        )
        == []
    )
    with pytest.raises(remerge.NoCandidateBigramError):
        remerge.run_tagged(
            corpus,
            1,
            patterns=[("VERB", "NOUN")],
            method="frequency",
            min_count=2,
            on_exhausted="raise",
        )


def test_long_pattern_support_merges_are_internal() -> None:
    corpus = [
        document(
            (
                token("new", "ADJ"),
                token("york", "NOUN"),
                token("office", "NOUN"),
            ),
            (
                token("new", "ADJ"),
                token("york", "NOUN"),
                token("office", "NOUN"),
            ),
        )
    ]

    winners, annotated, labels = remerge.annotate_tagged(
        corpus,
        1,
        patterns=[("ADJ", "NOUN", "NOUN")],
        method="frequency",
        min_count=2,
    )

    assert [winner.text for winner in winners] == ["new york office"]
    assert winners[0].merge_token_count == 2
    assert winners[0].occurrences == (
        remerge.MweOccurrence(0, 0, 0, 3),
        remerge.MweOccurrence(0, 1, 0, 3),
    )
    assert annotated == ["<mwe:new_york_office>\n<mwe:new_york_office>"]
    assert labels == ["<mwe:new_york_office>"]


def test_full_match_can_continue_as_support_for_longer_pattern() -> None:
    corpus = [
        document(
            (
                token("new", "ADJ"),
                token("york", "NOUN"),
                token("office", "NOUN"),
            )
        )
    ]

    winners = remerge.run_tagged(
        corpus,
        2,
        patterns=[("ADJ", "NOUN"), ("ADJ", "NOUN", "NOUN")],
        method="frequency",
    )

    assert [winner.text for winner in winners] == ["new york", "new york office"]


def test_wildcard_alternatives_and_x_have_explicit_meaning() -> None:
    corpus = [
        document(
            (token("mystery", "X"), token("item", "NOUN")),
            (token("bright", "ADJ"), token("item", "NOUN")),
        )
    ]

    wildcard = remerge.run_tagged(
        corpus,
        1,
        patterns=[("*", "NOUN")],
        method="frequency",
    )
    alternatives = remerge.run_tagged(
        corpus,
        1,
        patterns=[(frozenset({"X", "ADJ"}), "NOUN")],
        method="frequency",
    )
    exact_x = remerge.run_tagged(
        corpus,
        1,
        patterns=[("X", "NOUN")],
        method="frequency",
    )

    assert wildcard[0].merge_token_count == 1
    assert alternatives[0].merge_token_count == 1
    assert exact_x[0].text == "mystery item"


@pytest.mark.parametrize(
    ("corpus", "patterns", "message"),
    [
        ([document((token("bad tag", "NOUN"),))], [("NOUN", "NOUN")], "whitespace"),
        ([document((token("word", "BAD"),))], [("NOUN", "NOUN")], "17 Universal"),
        ([document(())], [("NOUN", "NOUN")], "must not be empty"),
        ([document((token("word", "NOUN"),))], [], "at least one pattern"),
        ([document((token("word", "NOUN"),))], [("NOUN",)], "at least two"),
        ([document((token("word", "NOUN"),))], [("BAD", "NOUN")], "not a Universal"),
    ],
)
def test_tagged_input_and_pattern_validation(
    corpus: list[remerge.TaggedDocument],
    patterns: list[remerge.PosPattern],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        remerge.run_tagged(corpus, 1, patterns=patterns)


def test_empty_document_round_trips() -> None:
    empty = remerge.TaggedDocument(sentences=())
    assert remerge.annotate_tagged([empty], 0, patterns=[("NOUN", "NOUN")]) == (
        [],
        [""],
        [],
    )


def test_strict_conllu_adapter_keeps_word_rows_and_boundaries() -> None:
    conllu = """# newdoc id = first
# text = can't go
1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
1\tca\tcan\tAUX\t_\t_\t0\troot\t_\tSpaceAfter=No
2\tn't\tnot\tPART\t_\t_\t1\tadvmod\t_\t_
2.1\tghost\tghost\tX\t_\t_\t_\t_\t_\t_
3\tgo\tgo\tVERB\t_\t_\t1\txcomp\t_\t_

# newdoc id = second
1\tDone\tdone\tADJ\t_\t_\t0\troot\t_\t_
2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_
"""

    documents = remerge.from_conllu(conllu)

    assert documents == [
        document((token("ca", "AUX"), token("n't", "PART"), token("go", "VERB"))),
        document((token("Done", "ADJ"), token(".", "PUNCT"))),
    ]


def test_conllu_adapter_rejects_repairable_alignment_errors() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        remerge.from_conllu(
            "1\tone\tone\tNUM\t_\t_\t0\troot\t_\t_\n"
            "3\tthree\tthree\tNUM\t_\t_\t1\tdep\t_\t_\n"
        )


def test_min_score_and_progress_use_emitted_winner_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = [
        document(
            (
                token("new", "ADJ"),
                token("york", "NOUN"),
                token("office", "NOUN"),
            )
        )
    ]
    assert (
        remerge.run_tagged(
            corpus,
            1,
            patterns=[("ADJ", "NOUN", "NOUN")],
            method="frequency",
            min_score=2,
        )
        == []
    )

    winners = remerge.run_tagged(
        corpus,
        1,
        patterns=[("ADJ", "NOUN", "NOUN")],
        method="frequency",
        progress=True,
    )
    assert len(winners) == 1
    assert "1/1" in capsys.readouterr().err


def test_unfiltered_diagnostic_reports_exact_occurrence_coordinates() -> None:
    corpus = ["a b a b\nc d", "", "a b"]

    diagnostic = remerge.run_with_occurrences(
        corpus,
        1,
        method="frequency",
        splitter="delimiter",
        line_delimiter="\n",
    )
    ordinary = remerge.run(
        corpus,
        1,
        method="frequency",
        splitter="delimiter",
        line_delimiter="\n",
    )

    assert [winner.text for winner in diagnostic] == [
        winner.text for winner in ordinary
    ]
    assert diagnostic[0].occurrences == (
        remerge.MweOccurrence(0, 0, 0, 2),
        remerge.MweOccurrence(0, 0, 2, 4),
        remerge.MweOccurrence(2, 0, 0, 2),
    )

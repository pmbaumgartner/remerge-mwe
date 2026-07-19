# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "remerge-mwe",
# ]
# ///

from importlib.metadata import version

import remerge


def main() -> None:
    winners = remerge.run(["a b a b"], 1, method="frequency")
    assert winners and winners[0].merged_lexeme.word == ("a", "b")

    document = remerge.TaggedDocument(
        sentences=(
            (
                remerge.TaggedToken("bright", "ADJ"),
                remerge.TaggedToken("river", "NOUN"),
            ),
            (
                remerge.TaggedToken("river", "NOUN"),
                remerge.TaggedToken("bright", "ADJ"),
            ),
        )
    )
    tagged = remerge.run_tagged(
        [document],
        1,
        patterns=[("ADJ", "NOUN")],
        method="frequency",
    )
    assert isinstance(tagged[0], remerge.WinnerWithOccurrences)
    assert tagged[0].merged_lexeme.word == ("bright", "river")
    assert tagged[0].occurrences == (remerge.MweOccurrence(0, 0, 0, 2),)

    annotated, documents, labels = remerge.annotate_tagged(
        [document],
        1,
        patterns=[("ADJ", "NOUN")],
        method="frequency",
    )
    assert annotated == tagged
    assert documents == ["<mwe:bright_river>\nriver bright"]
    assert labels == ["<mwe:bright_river>"]

    parsed = remerge.from_conllu(
        "# sent_id = smoke-1\n"
        "1\tbright\t_\tADJ\t_\t_\t0\troot\t_\t_\n"
        "2\triver\t_\tNOUN\t_\t_\t1\tnsubj\t_\t_\n"
    )
    assert parsed == [
        remerge.TaggedDocument(
            sentences=(
                (
                    remerge.TaggedToken("bright", "ADJ"),
                    remerge.TaggedToken("river", "NOUN"),
                ),
            )
        )
    ]
    diagnostic = remerge.run_with_occurrences(["bright river"], 1, method="frequency")
    assert isinstance(diagnostic[0], remerge.WinnerWithOccurrences)
    assert diagnostic[0].occurrences == (remerge.MweOccurrence(0, 0, 0, 2),)
    print("remerge-mwe version:", version("remerge-mwe"))
    print("first winner:", winners[0].merged_lexeme.word)
    print("supplied-tag winner:", tagged[0].merged_lexeme.word)
    print("ok")


if __name__ == "__main__":
    main()

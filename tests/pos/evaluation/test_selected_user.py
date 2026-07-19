from tests.pos.evaluation.loader import GoldSplit, MweSpan, Sentence, Token
from tests.pos.evaluation.selected_user import _representative_corpus


def test_representative_corpus_preserves_document_and_sentence_coordinates() -> None:
    sentences = (
        _sentence("reviews-a", "reviews-a-1", "bright", "ADJ", "light", "NOUN"),
        _sentence("reviews-a", "reviews-a-2", "runs", "VERB", "out", "PART"),
        _sentence("reviews-b", "reviews-b-1", "stone", "NOUN", "wall", "NOUN"),
    )
    gold = GoldSplit(
        split="dev",
        sentences=sentences,
        mwe_spans=frozenset(
            {
                MweSpan("reviews-a", "reviews-a-2", 0, 2),
                MweSpan("reviews-b", "reviews-b-1", 0, 2),
            }
        ),
    )

    raw, tagged, occurrences = _representative_corpus(gold)

    assert raw == ["bright light\nruns out", "stone wall"]
    assert [len(document.sentences) for document in tagged] == [2, 1]
    assert occurrences == {(0, 1, 0, 2), (1, 0, 0, 2)}


def _sentence(
    document_id: str,
    sentence_id: str,
    left_form: str,
    left_tag: str,
    right_form: str,
    right_tag: str,
) -> Sentence:
    return Sentence(
        document_id,
        sentence_id,
        (
            Token(document_id, sentence_id, 0, left_form, left_tag, "reviews"),
            Token(document_id, sentence_id, 1, right_form, right_tag, "reviews"),
        ),
    )

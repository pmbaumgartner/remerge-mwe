"""Public inference boundary controls for the standalone POS package."""

from __future__ import annotations

import unicodedata

import pytest

from remerge_pos import Tagger, UPOS_TAGS, load_model
from remerge_pos._perceptron import Config, train, write_artifact


DATA = (
    (("The", "DET"), ("watch", "NOUN")),
    (("I", "PRON"), ("watch", "VERB")),
    (("bright", "ADJ"), ("lights", "NOUN")),
    (("lights", "NOUN"), ("shine", "VERB")),
)
CONFIG = Config(epochs=2, feature_cutoff=1, feature_buckets=4096)


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "synthetic.rmsp"
    write_artifact(train(DATA, CONFIG, seed=7), path)
    return path


def test_public_api_loads_explicit_artifact_and_preserves_sentence_token_shape(
    artifact,
) -> None:
    tagger = Tagger.load(artifact)
    sentences = (("The", "watch"), ("I", "watch"), ("Hello",))

    tagged = tagger.tag(sentences, language="en")

    assert tagged == load_model(artifact).tag(sentences, language="en")
    assert isinstance(tagged, tuple)
    assert all(isinstance(sentence, tuple) for sentence in tagged)
    assert tuple(len(sentence) for sentence in tagged) == (2, 2, 1)
    assert {tag for sentence in tagged for tag in sentence} <= set(UPOS_TAGS)
    assert frozenset(UPOS_TAGS) == frozenset(
        {
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X",
        }
    )


@pytest.mark.parametrize(
    "sentences",
    (
        ((),),
        (("",),),
        (("two words",),),
        ((unicodedata.normalize("NFD", "café"),),),
    ),
)
def test_public_api_rejects_inputs_outside_the_alignment_contract(
    artifact, sentences
) -> None:
    with pytest.raises(ValueError):
        Tagger.load(artifact).tag(sentences)


def test_public_api_rejects_unsupported_language(artifact) -> None:
    with pytest.raises(ValueError):
        Tagger.load(artifact).tag((("hello",),), language="fr")


def test_public_api_validates_all_sentences_before_inference(
    artifact, monkeypatch: pytest.MonkeyPatch
) -> None:
    tagger = Tagger.load(artifact)
    original = type(tagger._model).tag_sentence
    calls = 0

    def counted(model, forms):
        nonlocal calls
        calls += 1
        return original(model, forms)

    monkeypatch.setattr(type(tagger._model), "tag_sentence", counted)
    with pytest.raises(ValueError):
        tagger.tag((("valid",), ("",)))
    assert calls == 0

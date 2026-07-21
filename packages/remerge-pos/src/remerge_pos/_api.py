"""Stable, occurrence-aligned inference boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import unicodedata

from ._perceptron import Model, SCHEMA, TAGS


def _validate_sentences(sentences: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
        raise ValueError("sentences must be a sentence-nested sequence of forms")
    validated = []
    for sentence_index, sentence in enumerate(sentences):
        if not isinstance(sentence, Sequence) or isinstance(sentence, (str, bytes)):
            raise ValueError(f"sentence {sentence_index} must be a sequence of forms")
        if not sentence:
            raise ValueError(f"sentence {sentence_index} must not be empty")
        forms = []
        for token_index, form in enumerate(sentence):
            coordinate = f"sentence {sentence_index}, token {token_index}"
            if not isinstance(form, str) or not form:
                raise ValueError(f"{coordinate} form must be a non-empty string")
            if not unicodedata.is_normalized("NFC", form):
                raise ValueError(f"{coordinate} form must be NFC")
            if any(character.isspace() for character in form):
                raise ValueError(f"{coordinate} form must not contain whitespace")
            forms.append(form)
        validated.append(tuple(forms))
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class Tagger:
    """An explicitly loaded English UPOS tagger."""

    _model: Model = field(repr=False)
    artifact_path: Path
    artifact_sha256: str

    model_id = SCHEMA
    language = "en"

    @classmethod
    def load(cls, artifact_path: str | Path) -> Tagger:
        """Load a local RMSP0001 artifact without discovery or network access."""

        path = Path(artifact_path)
        model, artifact = Model.from_path(path)
        return cls(
            _model=model,
            artifact_path=path,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        )

    def tag(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        language: str = "en",
    ) -> tuple[tuple[str, ...], ...]:
        """Tag caller-tokenized sentences while preserving their exact nesting."""

        if language != self.language:
            raise ValueError("remerge-pos supports only declared language 'en'")
        validated = _validate_sentences(sentences)
        return tuple(self._model.tag_sentence(sentence) for sentence in validated)


def load_model(artifact_path: str | Path) -> Tagger:
    """Load an English UPOS model from an explicit local path."""

    return Tagger.load(artifact_path)


UPOS_TAGS = TAGS

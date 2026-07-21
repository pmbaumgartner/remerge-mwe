"""Common-harness adapter for the structured-perceptron pilot artifact."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from remerge_pos import Tagger


class StructuredPerceptronCandidate:
    model_id = "remerge-pos-structured-perceptron-v1"
    tokenizer_id = "canonical-boundaries-v1"

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        artifact = artifact_path.read_bytes()
        self.artifact_bytes = len(artifact)
        self.artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        self._tagger = Tagger.load(artifact_path)

    def tag(self, documents):
        return tuple(self._tagger.tag(document) for document in documents)


def create_candidate() -> StructuredPerceptronCandidate:
    path = os.environ.get("REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT")
    if not path:
        raise ValueError("REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT is required")
    return StructuredPerceptronCandidate(Path(path))

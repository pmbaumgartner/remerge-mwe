"""Common-harness adapter for the structured-perceptron pilot artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys


def _model_module():
    path = Path(__file__).with_name("structured_perceptron.py")
    specification = importlib.util.spec_from_file_location(
        "remerge_pos_structured_perceptron", path
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class StructuredPerceptronCandidate:
    model_id = "remerge-pos-structured-perceptron-v1"
    tokenizer_id = "canonical-boundaries-v1"

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        artifact = artifact_path.read_bytes()
        self.artifact_bytes = len(artifact)
        self.artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        self._model = _model_module().Model.from_artifact(artifact)

    def tag(self, documents):
        return self._model.tag(documents)


def create_candidate() -> StructuredPerceptronCandidate:
    path = os.environ.get("REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT")
    if not path:
        raise ValueError("REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT is required")
    return StructuredPerceptronCandidate(Path(path))

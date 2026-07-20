"""Common-bakeoff adapter for a local first-party TnT pilot artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys


def _model_module():
    path = Path(__file__).with_name("tnt.py")
    specification = importlib.util.spec_from_file_location("remerge_pos_tnt", path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class TntCandidate:
    model_id = "remerge-pos-tnt-v1"
    tokenizer_id = "canonical-boundaries-v1"

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        artifact = artifact_path.read_bytes()
        self.artifact_bytes = len(artifact)
        self.artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        self._model = _model_module().Model.from_artifact(artifact)

    def tag(self, documents):
        return self._model.tag(documents)


def create_candidate() -> TntCandidate:
    path = os.environ.get("REMERGE_POS_TNT_ARTIFACT")
    if not path:
        raise ValueError("REMERGE_POS_TNT_ARTIFACT is required")
    return TntCandidate(Path(path))

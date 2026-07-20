"""Canonical project-authored workload for POS release measurements."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


_CYCLE = (
    ("bright", "ADJ"),
    ("river", "NOUN"),
    ("flows", "VERB"),
    ("swiftly", "ADV"),
    ("quiet", "ADJ"),
    ("garden", "NOUN"),
    ("grows", "VERB"),
    ("today", "ADV"),
)
_SENTENCE_SIZE = 32


@dataclass(frozen=True, slots=True)
class ProjectAuthoredWorkload:
    document_id: str
    domain: str
    sentences: tuple[tuple[tuple[str, str], ...], ...]

    @property
    def raw(self) -> str:
        return "\n".join(
            " ".join(form for form, _upos in sentence)
            for sentence in self.sentences
        )

    @property
    def digest(self) -> str:
        payload = [
            {
                "document_id": self.document_id,
                "domain": self.domain,
                "sentences": [
                    [[form, upos] for form, upos in sentence]
                    for sentence in self.sentences
                ],
            }
        ]
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


def project_authored_workload(token_count: int) -> ProjectAuthoredWorkload:
    """Generate the seed-free, MIT workload used by every release sensor."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    tokens = tuple(_CYCLE[index % len(_CYCLE)] for index in range(token_count))
    sentences = tuple(
        tokens[offset : offset + _SENTENCE_SIZE]
        for offset in range(0, token_count, _SENTENCE_SIZE)
    )
    return ProjectAuthoredWorkload(
        document_id="project-authored-0",
        domain="project-authored",
        sentences=sentences,
    )

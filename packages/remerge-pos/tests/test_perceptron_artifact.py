"""Deterministic synthetic-data and bounded-loader controls."""

from __future__ import annotations

import hashlib
import json
import zlib

import pytest

from remerge_pos import Tagger
from remerge_pos import _perceptron as perceptron
from remerge_pos._perceptron import (
    HEADER,
    MAGIC,
    MAX_PAYLOAD_BYTES,
    Config,
    Model,
    train,
    write_artifact,
)


DATA = (
    (("The", "DET"), ("watch", "NOUN")),
    (("I", "PRON"), ("watch", "VERB")),
    (("bright", "ADJ"), ("lights", "NOUN")),
    (("lights", "NOUN"), ("shine", "VERB")),
)
CONFIG = Config(epochs=2, feature_cutoff=1, feature_buckets=4096)


def test_training_and_artifact_export_are_deterministic_for_fixed_synthetic_input(
    tmp_path,
) -> None:
    first = train(DATA, CONFIG, seed=19)
    second = train(DATA, CONFIG, seed=19)
    first_path = tmp_path / "first.rmsp"
    second_path = tmp_path / "second.rmsp"

    first_digest = write_artifact(first, first_path)
    second_digest = write_artifact(second, second_path)

    assert (
        first.artifact()
        == second.artifact()
        == first_path.read_bytes()
        == second_path.read_bytes()
    )
    assert first_digest == second_digest == hashlib.sha256(first.artifact()).hexdigest()
    assert Model.from_artifact(first.artifact()).artifact() == first.artifact()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda artifact: artifact[:-1],
        lambda artifact: artifact + b"trailing",
        lambda artifact: b"BROKEN!!" + artifact[8:],
        lambda artifact: artifact[: HEADER.size - 1],
        lambda _artifact: HEADER.pack(MAGIC, MAX_PAYLOAD_BYTES + 1, 0, b"\\0" * 32),
    ),
)
def test_loader_rejects_truncated_corrupt_trailing_and_oversize_artifacts(
    mutate,
) -> None:
    artifact = train(DATA, CONFIG, seed=23).artifact()

    with pytest.raises(ValueError):
        Model.from_artifact(mutate(artifact))


def test_explicit_file_loader_rejects_checksum_mismatch(tmp_path) -> None:
    artifact = bytearray(train(DATA, CONFIG, seed=29).artifact())
    magic, raw_size, compressed_size, _digest = HEADER.unpack_from(artifact)
    artifact[: HEADER.size] = HEADER.pack(magic, raw_size, compressed_size, b"\\0" * 32)
    path = tmp_path / "checksum-mismatch.rmsp"
    path.write_bytes(artifact)

    with pytest.raises(ValueError):
        Tagger.load(path)


def test_explicit_file_loader_bounds_the_initial_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(perceptron, "MAX_ARTIFACT_BYTES", 8)
    path = tmp_path / "too-large.rmsp"
    path.write_bytes(b"123456789")

    with pytest.raises(ValueError, match="exceeds 20 MiB"):
        Tagger.load(path)


def _repack(raw: bytes) -> bytes:
    compressed = zlib.compress(raw, level=9)
    return (
        HEADER.pack(MAGIC, len(raw), len(compressed), hashlib.sha256(raw).digest())
        + compressed
    )


def test_loader_rejects_duplicate_json_keys() -> None:
    artifact = train(DATA, CONFIG, seed=31).artifact()
    raw = zlib.decompress(artifact[HEADER.size :])
    duplicated = raw.replace(
        b'"schema":',
        b'"schema":"duplicate","schema":',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        Model.from_artifact(_repack(duplicated))


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema", "wrong-schema"),
        ("tags", ["NOUN"]),
        ("feature_weights", [[1, 2]]),
        ("transition_weights", [[0, 1, float("nan")]]),
    ),
)
def test_loader_rejects_invalid_schema_and_weight_shapes(field, value) -> None:
    artifact = train(DATA, CONFIG, seed=37).artifact()
    payload = json.loads(zlib.decompress(artifact[HEADER.size :]))
    payload[field] = value
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

    with pytest.raises(ValueError):
        Model.from_artifact(_repack(raw))

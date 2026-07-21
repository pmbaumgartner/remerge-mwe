"""Compatibility shim for the completed structured-perceptron pilot.

The maintained implementation now lives in the standalone workspace package.
"""

from remerge_pos._perceptron import (
    ALL_TAGS,
    BOS,
    EOS,
    HEADER,
    MAGIC,
    MAX_ARTIFACT_BYTES,
    MAX_PAYLOAD_BYTES,
    SCHEMA,
    TAG_INDEX,
    TAGS,
    Config,
    Model,
    _Averager,
    _decode_indices,
    _training_digest,
    _transitions,
    feature_ids,
    lexicon_hash,
    stable_hash,
    train,
    write_artifact,
)


__all__ = [
    "ALL_TAGS",
    "BOS",
    "EOS",
    "HEADER",
    "MAGIC",
    "MAX_ARTIFACT_BYTES",
    "MAX_PAYLOAD_BYTES",
    "SCHEMA",
    "TAG_INDEX",
    "TAGS",
    "Config",
    "Model",
    "_Averager",
    "_decode_indices",
    "_training_digest",
    "_transitions",
    "feature_ids",
    "lexicon_hash",
    "stable_hash",
    "train",
    "write_artifact",
]

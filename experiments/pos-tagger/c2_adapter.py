"""Development-only adapter for the retained ``RMPOS001`` c2 artifact.

It is intentionally outside the package and is loaded only by the bakeoff
command.  ``REMERGE_POS_C2_ARTIFACT`` names the locally produced experiment
artifact; no downloads, package entry points, or production APIs are involved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import os
import struct
import unicodedata


TAGS = (
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
)
TAG_INDEX = {tag: index for index, tag in enumerate(TAGS)}
HEADER = struct.Struct("<8sHHIHHHHIII")
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
PUNCTUATION = frozenset("!\"'(),-./:;?[]_{}\\«»‐‑‒–—―‘’‚‛“”„‟…")
NUMBER_CONNECTORS = frozenset("+-−.,_/%‰eE")
HYPHENS = frozenset("-‐‑‒–—−")
APOSTROPHES = frozenset("'’")


def _fnv1a(value: str) -> int:
    result = FNV_OFFSET
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * FNV_PRIME) & MASK64
    return result


def _shape(value: str) -> str:
    if value in {"<BOS>", "<EOS>"}:
        return value
    return "".join(
        "X"
        if char.isupper()
        else "x"
        if char.islower() or char.isalpha()
        else "d"
        if char.isnumeric()
        else char
        for char in value
    )


def _ascii_lower(value: str) -> str:
    return (
        value
        if value in {"<BOS>", "<EOS>"}
        else value.translate(
            str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
        )
    )


def _is_title(value: str) -> bool:
    alphabetic = [char for char in value if char.isalpha()]
    return (
        len(alphabetic) >= 2
        and alphabetic[0].isupper()
        and all(char.islower() for char in alphabetic[1:])
    )


def _forced_tag(form: str) -> int | None:
    if all(char in PUNCTUATION for char in form):
        return TAG_INDEX["PUNCT"]
    seen_digit = False
    for char in form:
        if char.isnumeric():
            seen_digit = True
        elif char not in NUMBER_CONNECTORS:
            break
    else:
        if seen_digit:
            return TAG_INDEX["NUM"]
    if all(
        not char.isspace()
        and not char.isalnum()
        and char not in PUNCTUATION
        and unicodedata.category(char) != "Cc"
        for char in form
    ):
        return TAG_INDEX["SYM"]
    return None


def _features(forms: tuple[str, ...], index: int) -> tuple[str, ...]:
    form = forms[index]
    lower = _ascii_lower(form)
    previous = forms[index - 1] if index else "<BOS>"
    following = forms[index + 1] if index + 1 < len(forms) else "<EOS>"
    previous_lower = _ascii_lower(previous)
    following_lower = _ascii_lower(following)
    values = [f"W={form}", f"L={lower}", f"S={_shape(form)}"]
    values.extend(f"P{width}={lower[:width]}" for width in range(1, 5))
    values.extend(f"U{width}={lower[-width:]}" for width in range(1, 5))
    values.extend(
        (
            f"PL={previous_lower}",
            f"PS={_shape(previous)}",
            f"NL={following_lower}",
            f"NS={_shape(following)}",
            f"PC={previous_lower}|{lower}",
            f"CN={lower}|{following_lower}",
        )
    )
    if any(char.isupper() for char in form):
        values.append("F=upper")
    if _is_title(form):
        values.append("F=title")
    if any(char.isnumeric() for char in form):
        values.append("F=digit")
    if any(char in HYPHENS for char in form):
        values.append("F=hyphen")
    if any(char in APOSTROPHES for char in form):
        values.append("F=apostrophe")
    return tuple(values)


class C2Adapter:
    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        data = artifact_path.read_bytes()
        self.artifact_bytes = len(data)
        self.artifact_sha256 = hashlib.sha256(data).hexdigest()
        if len(data) < HEADER.size + 34:
            raise ValueError("truncated RMPOS001 artifact")
        reader = memoryview(data)
        (
            magic,
            version,
            tag_count,
            buckets,
            feature_limit,
            reserved,
            model_length,
            tokenizer_length,
            direct_count,
            candidate_count,
            weights_length,
        ) = HEADER.unpack_from(reader)
        if (
            magic != b"RMPOS001"
            or version != 1
            or tag_count != len(TAGS)
            or feature_limit != 22
            or reserved
            or not buckets
            or buckets & (buckets - 1)
        ):
            raise ValueError("invalid RMPOS001 artifact header")
        offset = HEADER.size
        table_bytes = direct_count * 9 + candidate_count * 12
        if (
            model_length == 0
            or tokenizer_length == 0
            or weights_length != buckets * len(TAGS)
            or offset
            + model_length
            + tokenizer_length
            + 34
            + table_bytes
            + weights_length
            != len(data)
        ):
            raise ValueError("invalid RMPOS001 artifact lengths")
        try:
            self.model_id = bytes(reader[offset : offset + model_length]).decode(
                "utf-8"
            )
            offset += model_length
            self.tokenizer_id = bytes(
                reader[offset : offset + tokenizer_length]
            ).decode("utf-8")
            offset += tokenizer_length
        except UnicodeDecodeError as error:
            raise ValueError("RMPOS001 identity is not UTF-8") from error
        self.biases = struct.unpack_from("<17h", reader, offset)
        offset += 34
        self.direct: dict[int, int] = {}
        previous = -1
        for _ in range(direct_count):
            key, tag = struct.unpack_from("<QB", reader, offset)
            offset += 9
            if key <= previous or tag >= len(TAGS):
                raise ValueError("invalid RMPOS001 direct table")
            previous = key
            self.direct[key] = tag
        self.candidates: dict[int, int] = {}
        previous = -1
        for _ in range(candidate_count):
            key, mask = struct.unpack_from("<QI", reader, offset)
            offset += 12
            if (
                key <= previous
                or not mask
                or mask & ~((1 << len(TAGS)) - 1)
                or key in self.direct
            ):
                raise ValueError("invalid RMPOS001 candidate table")
            previous = key
            self.candidates[key] = mask
        self.weights = struct.unpack_from(f"<{weights_length}b", reader, offset)
        self.bucket_mask = buckets - 1

    def _tag(self, forms: tuple[str, ...], index: int) -> str:
        form = forms[index]
        forced = _forced_tag(form)
        if forced is not None:
            return TAGS[forced]
        key = _fnv1a(form)
        if key in self.direct:
            return TAGS[self.direct[key]]
        mask = self.candidates.get(key, (1 << len(TAGS)) - 1)
        scores = list(self.biases)
        for feature in _features(forms, index):
            base = (_fnv1a(feature) & self.bucket_mask) * len(TAGS)
            for tag in range(len(TAGS)):
                if mask & (1 << tag):
                    scores[tag] += self.weights[base + tag]
        return TAGS[
            max(
                (tag for tag in range(len(TAGS)) if mask & (1 << tag)),
                key=lambda tag: (scores[tag], -tag),
            )
        ]

    def tag(self, documents):
        return tuple(
            tuple(
                tuple(
                    self._tag(tuple(sentence), index) for index in range(len(sentence))
                )
                for sentence in document
            )
            for document in documents
        )


def create_candidate() -> C2Adapter:
    value = os.environ.get("REMERGE_POS_C2_ARTIFACT")
    if not value:
        raise ValueError("REMERGE_POS_C2_ARTIFACT is required")
    return C2Adapter(Path(value))

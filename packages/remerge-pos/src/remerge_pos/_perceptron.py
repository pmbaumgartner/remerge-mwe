"""Averaged structured-perceptron training, inference, and artifacts.

This private module preserves the completed research pilot's feature family,
Viterbi tie-breaking, update order, averaging, and RMSP0001 artifact bytes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import zlib


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
ALL_TAGS = tuple(range(len(TAGS)))
BOS = len(TAGS)
EOS = BOS + 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
MAGIC = b"RMSP0001"
HEADER = struct.Struct("<8sII32s")
SCHEMA = "remerge-pos-structured-perceptron-v1"
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def stable_hash(value: str) -> int:
    result = FNV_OFFSET
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * FNV_PRIME) & MASK64
    return result


def lexicon_hash(form: str) -> int:
    """Return the full-width identity hash used to reject word collisions."""

    return stable_hash(f"lexicon\0{form}")


def _shape(form: str) -> str:
    classes = []
    for character in form:
        category = (
            "X"
            if character.isupper()
            else "x"
            if character.islower()
            else "d"
            if character.isdigit()
            else "-"
            if character == "-"
            else "o"
        )
        if not classes or classes[-1] != category:
            classes.append(category)
    return "".join(classes[:8])


@dataclass(frozen=True, slots=True)
class Config:
    """Bounded training configuration for the structured perceptron."""

    epochs: int
    feature_cutoff: int
    feature_buckets: int = 131_072

    def validate(self) -> None:
        if type(self.epochs) is not int or not 1 <= self.epochs <= 20:
            raise ValueError("epochs must be an integer in [1, 20]")
        if type(self.feature_cutoff) is not int or not 1 <= self.feature_cutoff <= 20:
            raise ValueError("feature_cutoff must be an integer in [1, 20]")
        if (
            type(self.feature_buckets) is not int
            or self.feature_buckets < 1_024
            or self.feature_buckets > 1_048_576
        ):
            raise ValueError("feature_buckets must be an integer in [1024, 1048576]")


def feature_ids(forms: Sequence[str], index: int, buckets: int) -> tuple[int, ...]:
    """Return the frozen hashed lexical and morphological feature family."""

    form = forms[index]
    lower = form.lower()
    previous = forms[index - 1].lower() if index else "<BOS>"
    following = forms[index + 1].lower() if index + 1 < len(forms) else "<EOS>"
    names = [
        "bias",
        f"word={form}",
        f"lower={lower}",
        f"shape={_shape(form)}",
        f"prev={previous}",
        f"next={following}",
    ]
    for length in range(1, min(4, len(lower)) + 1):
        names.append(f"prefix{length}={lower[:length]}")
        names.append(f"suffix{length}={lower[-length:]}")
    if any(character.isdigit() for character in form):
        names.append("has-digit")
    if "-" in form:
        names.append("has-hyphen")
    if form[:1].isupper():
        names.append("initial-upper")
    return tuple(sorted(stable_hash(name) % buckets for name in names))


@dataclass(frozen=True, slots=True)
class Model:
    """Private in-memory representation of an RMSP0001 model."""

    config: Config
    seed: int
    steps: int
    training_digest: str
    feature_weights: dict[tuple[int, int], float]
    transition_weights: dict[tuple[int, int], float]

    def __post_init__(self) -> None:
        self.config.validate()
        if type(self.seed) is not int or type(self.steps) is not int or self.steps <= 0:
            raise ValueError("invalid structured-perceptron seed or step count")
        if (
            not isinstance(self.training_digest, str)
            or len(self.training_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.training_digest
            )
        ):
            raise ValueError("invalid structured-perceptron training digest")
        for (feature, tag), weight in self.feature_weights.items():
            if (
                type(feature) is not int
                or not 0 <= feature < self.config.feature_buckets
                or type(tag) is not int
                or tag not in ALL_TAGS
                or not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(weight)
                or weight == 0
            ):
                raise ValueError("invalid structured-perceptron feature weight")
        for (previous, tag), weight in self.transition_weights.items():
            if (
                type(previous) is not int
                or previous not in ALL_TAGS + (BOS,)
                or type(tag) is not int
                or tag not in ALL_TAGS + (EOS,)
                or (previous == BOS and tag == EOS)
                or not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(weight)
                or weight == 0
            ):
                raise ValueError("invalid structured-perceptron transition weight")

    def decode_indices(self, forms: Sequence[str]) -> tuple[int, ...]:
        """Run exact 17-tag Viterbi with lexicographic ties and scored EOS."""

        return _decode_indices(
            forms, self.config, self.feature_weights, self.transition_weights
        )

    def tag_sentence(self, forms: Sequence[str]) -> tuple[str, ...]:
        return tuple(TAGS[tag] for tag in self.decode_indices(forms))

    def artifact(self) -> bytes:
        payload = {
            "config": asdict(self.config),
            "feature_weights": [
                [feature, tag, weight]
                for (feature, tag), weight in sorted(self.feature_weights.items())
            ],
            "schema": SCHEMA,
            "seed": self.seed,
            "steps": self.steps,
            "tags": TAGS,
            "training_digest": self.training_digest,
            "transition_weights": [
                [previous, tag, weight]
                for (previous, tag), weight in sorted(self.transition_weights.items())
            ],
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        compressed = zlib.compress(raw, level=9)
        artifact = (
            HEADER.pack(MAGIC, len(raw), len(compressed), hashlib.sha256(raw).digest())
            + compressed
        )
        if len(artifact) > MAX_ARTIFACT_BYTES:
            raise ValueError("structured-perceptron artifact exceeds 20 MiB")
        return artifact

    @classmethod
    def from_artifact(cls, artifact: bytes) -> Model:
        if not isinstance(artifact, bytes):
            raise ValueError("structured-perceptron artifact must be bytes")
        if len(artifact) > MAX_ARTIFACT_BYTES or len(artifact) < HEADER.size:
            raise ValueError("invalid structured-perceptron artifact size")
        magic, raw_size, compressed_size, expected_digest = HEADER.unpack_from(artifact)
        compressed = artifact[HEADER.size :]
        if (
            magic != MAGIC
            or compressed_size != len(compressed)
            or raw_size > MAX_PAYLOAD_BYTES
        ):
            raise ValueError("invalid structured-perceptron artifact header")
        decompressor = zlib.decompressobj()
        try:
            raw = decompressor.decompress(compressed, raw_size + 1)
        except zlib.error as error:
            raise ValueError("invalid structured-perceptron compression") from error
        if (
            len(raw) != raw_size
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
            or hashlib.sha256(raw).digest() != expected_digest
        ):
            raise ValueError("invalid structured-perceptron payload")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        try:
            payload = json.loads(raw, object_pairs_hook=reject_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid structured-perceptron JSON") from error
        required = {
            "config",
            "feature_weights",
            "schema",
            "seed",
            "steps",
            "tags",
            "training_digest",
            "transition_weights",
        }
        try:
            valid_schema = (
                isinstance(payload, dict)
                and set(payload) == required
                and payload["schema"] == SCHEMA
                and tuple(payload["tags"]) == TAGS
            )
        except (KeyError, TypeError):
            valid_schema = False
        if not valid_schema:
            raise ValueError("invalid structured-perceptron schema")
        try:
            config_value = payload["config"]
            if not isinstance(config_value, dict):
                raise TypeError
            config = Config(**config_value)
        except (TypeError, KeyError) as error:
            raise ValueError("invalid structured-perceptron configuration") from error

        def weights(name: str) -> dict[tuple[int, int], float]:
            rows = payload[name]
            if not isinstance(rows, list):
                raise ValueError(f"invalid {name}")
            result = {}
            for row in rows:
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError(f"invalid {name} row")
                key = (row[0], row[1])
                try:
                    duplicate = key in result
                except TypeError as error:
                    raise ValueError(f"invalid {name} key") from error
                if duplicate:
                    raise ValueError(f"duplicate {name} row")
                try:
                    result[key] = row[2]
                except TypeError as error:
                    raise ValueError(f"invalid {name} key") from error
            return result

        return cls(
            config=config,
            seed=payload["seed"],
            steps=payload["steps"],
            training_digest=payload["training_digest"],
            feature_weights=weights("feature_weights"),
            transition_weights=weights("transition_weights"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> tuple[Model, bytes]:
        artifact_path = Path(path)
        with artifact_path.open("rb") as source:
            artifact = source.read(MAX_ARTIFACT_BYTES + 1)
        if len(artifact) > MAX_ARTIFACT_BYTES:
            raise ValueError("structured-perceptron artifact exceeds 20 MiB")
        return cls.from_artifact(artifact), artifact


def _decode_indices(
    forms: Sequence[str],
    config: Config,
    feature_weights: dict[tuple[int, int], float],
    transition_weights: dict[tuple[int, int], float],
) -> tuple[int, ...]:
    """Decode without constructing a model during every training update."""

    if not forms:
        return ()
    observations = []
    for index in range(len(forms)):
        features = feature_ids(forms, index, config.feature_buckets)
        observations.append(
            tuple(
                sum(feature_weights.get((feature, tag), 0.0) for feature in features)
                for tag in ALL_TAGS
            )
        )
    scores = tuple(
        observations[0][tag] + transition_weights.get((BOS, tag), 0.0)
        for tag in ALL_TAGS
    )
    ranks = ALL_TAGS
    backpointers: list[tuple[int, ...]] = []
    for position in range(1, len(forms)):
        parents = []
        next_scores = []
        for tag in ALL_TAGS:
            parent = max(
                ALL_TAGS,
                key=lambda previous: (
                    scores[previous] + transition_weights.get((previous, tag), 0.0),
                    -ranks[previous],
                ),
            )
            parents.append(parent)
            next_scores.append(
                scores[parent]
                + transition_weights.get((parent, tag), 0.0)
                + observations[position][tag]
            )
        ordered = sorted(ALL_TAGS, key=lambda tag: (ranks[parents[tag]], tag))
        next_ranks = [0] * len(TAGS)
        for rank, tag in enumerate(ordered):
            next_ranks[tag] = rank
        scores = tuple(next_scores)
        ranks = tuple(next_ranks)
        backpointers.append(tuple(parents))
    final = max(
        ALL_TAGS,
        key=lambda tag: (
            scores[tag] + transition_weights.get((tag, EOS), 0.0),
            -ranks[tag],
        ),
    )
    path = [final]
    for parents in reversed(backpointers):
        path.append(parents[path[-1]])
    return tuple(reversed(path))


class _Averager:
    def __init__(self) -> None:
        self.weights: dict[tuple[int, int], float] = {}
        self.totals: dict[tuple[int, int], float] = {}
        self.timestamps: dict[tuple[int, int], int] = {}

    def update(self, key: tuple[int, int], delta: float, step: int) -> None:
        weight = self.weights.get(key, 0.0)
        self.totals[key] = (
            self.totals.get(key, 0.0) + (step - self.timestamps.get(key, 0)) * weight
        )
        self.timestamps[key] = step
        updated = weight + delta
        if updated:
            self.weights[key] = updated
        else:
            self.weights.pop(key, None)

    def averaged(self, steps: int) -> dict[tuple[int, int], float]:
        if steps <= 0:
            raise ValueError("cannot average zero steps")
        keys = self.weights.keys() | self.totals.keys()
        result = {}
        for key in keys:
            total = self.totals.get(key, 0.0) + (
                steps - self.timestamps.get(key, 0)
            ) * self.weights.get(key, 0.0)
            average = total / steps
            if average:
                result[key] = average
        return result


def _transitions(path: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not path:
        return ()
    return ((BOS, path[0]),) + tuple(zip(path, path[1:])) + ((path[-1], EOS),)


def _training_digest(sentences: Sequence[Sequence[tuple[str, str]]]) -> str:
    digest = hashlib.sha256()
    for sentence in sentences:
        digest.update(struct.pack("<I", len(sentence)))
        for form, label in sentence:
            encoded = form.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            digest.update(bytes((TAG_INDEX[label],)))
    return digest.hexdigest()


def train(
    sentences: Iterable[Sequence[tuple[str, str]]], config: Config, seed: int
) -> Model:
    """Train the frozen averaged structured-perceptron recipe."""

    config.validate()
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    data = tuple(tuple(sentence) for sentence in sentences)
    if not data or any(not sentence for sentence in data):
        raise ValueError("training requires non-empty sentences")
    lexicon_sources: dict[int, str] = {}
    for sentence in data:
        for form, label in sentence:
            if not isinstance(form, str) or not form or label not in TAG_INDEX:
                raise ValueError("training token has invalid form or UPOS")
            identity = lexicon_hash(form)
            prior = lexicon_sources.setdefault(identity, form)
            if prior != form:
                raise ValueError("lexicon identity hash collision")
    feature_counts: Counter[int] = Counter()
    cached = []
    for sentence in data:
        forms = tuple(form for form, _label in sentence)
        features = tuple(
            feature_ids(forms, index, config.feature_buckets)
            for index in range(len(forms))
        )
        cached.append(
            (forms, tuple(TAG_INDEX[label] for _form, label in sentence), features)
        )
        feature_counts.update(feature for position in features for feature in position)
    active = frozenset(
        feature
        for feature, count in feature_counts.items()
        if count >= config.feature_cutoff
    )
    features_by_sentence = tuple(
        (
            forms,
            gold,
            tuple(
                tuple(feature for feature in position if feature in active)
                for position in features
            ),
        )
        for forms, gold, features in cached
    )
    feature_averager = _Averager()
    transition_averager = _Averager()
    step = 0
    indices = list(range(len(data)))
    generator = random.Random(seed)
    for _epoch in range(config.epochs):
        generator.shuffle(indices)
        for sentence_index in indices:
            forms, gold, features = features_by_sentence[sentence_index]
            predicted = _decode_indices(
                forms,
                config,
                feature_averager.weights,
                transition_averager.weights,
            )
            if predicted != gold:
                delta: Counter[tuple[str, int, int]] = Counter()
                for position, (expected, actual) in enumerate(zip(gold, predicted)):
                    if expected != actual:
                        for feature in features[position]:
                            delta["feature", feature, expected] += 1
                            delta["feature", feature, actual] -= 1
                expected_transitions = Counter(_transitions(gold))
                actual_transitions = Counter(_transitions(predicted))
                for key in expected_transitions.keys() | actual_transitions.keys():
                    delta["transition", key[0], key[1]] += (
                        expected_transitions[key] - actual_transitions[key]
                    )
                for (kind, first, second), change in sorted(delta.items()):
                    if not change:
                        continue
                    target = (
                        feature_averager if kind == "feature" else transition_averager
                    )
                    target.update((first, second), float(change), step)
            step += 1
    return Model(
        config=config,
        seed=seed,
        steps=step,
        training_digest=_training_digest(data),
        feature_weights=feature_averager.averaged(step),
        transition_weights=transition_averager.averaged(step),
    )


def write_artifact(model: Model, path: str | Path) -> str:
    """Write a canonical model artifact and return its SHA-256 digest."""

    artifact = model.artifact()
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact)
    return hashlib.sha256(artifact).hexdigest()

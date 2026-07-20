"""Deterministic, first-party TnT-style trigram UPOS HMM for dev research."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


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
BOS = len(TAGS)
EOS = BOS + 1
EVENT_COUNT = EOS + 1
EVENTS = tuple(range(len(TAGS))) + (EOS,)
ALL_TAGS = tuple(range(len(TAGS)))
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def token_hash(value: str) -> int:
    """Stable first-party hash; collisions are rejected while training."""

    result = FNV_OFFSET
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * FNV_PRIME) & MASK64
    return result


def suffix_hash(form: str, length: int) -> int:
    return token_hash(form[-length:])


@dataclass(frozen=True)
class Config:
    transition_alpha: float
    emission_alpha: float
    suffix_alpha: float
    suffix_length: int
    rare_word_max_count: int

    def validate(self) -> None:
        numeric = (self.transition_alpha, self.emission_alpha, self.suffix_alpha)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in numeric
        ):
            raise ValueError("smoothing values must be finite and positive")
        if (
            type(self.suffix_length) is not int
            or type(self.rare_word_max_count) is not int
        ):
            raise ValueError("suffix configuration values must be integers")
        if not 1 <= self.suffix_length <= 8:
            raise ValueError("suffix_length must be in [1, 8]")
        if not 1 <= self.rare_word_max_count <= 10:
            raise ValueError("rare_word_max_count must be in [1, 10]")


@dataclass(frozen=True)
class Model:
    config: Config
    # The deleted-interpolation weights are learned only from the gold train
    # counts; they are deliberately not a tunable grid parameter.
    lambdas: tuple[float, float, float]
    event_counts: tuple[int, ...]
    bigram: dict[tuple[int, int], int]
    context: dict[tuple[int, int], int]
    trigram: dict[tuple[int, int, int], int]
    emissions: dict[tuple[int, int], int]
    tag_totals: tuple[int, ...]
    suffixes: dict[tuple[int, int, int], int]
    suffix_totals: dict[tuple[int, int], int]
    vocabulary_size: int
    bigram_totals: tuple[int, ...] = ()
    word_candidate_masks: dict[int, int] = field(default_factory=dict)
    known_words: frozenset[int] = field(init=False, repr=False, compare=False)
    word_candidates: dict[int, tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )
    event_total: int = field(init=False, repr=False, compare=False)
    tag_total_count: int = field(init=False, repr=False, compare=False)
    tag_vocabulary_sizes: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Materialize all aggregate lookups used inside the Viterbi loop."""

        derived_totals = tuple(
            sum(
                value
                for (history, _event), value in self.bigram.items()
                if history == index
            )
            for index in range(EVENT_COUNT)
        )
        if self.bigram_totals and self.bigram_totals != derived_totals:
            raise ValueError("invalid TnT bigram totals")
        object.__setattr__(self, "bigram_totals", derived_totals)
        derived_masks: dict[int, int] = {}
        for word, tag in self.emissions:
            derived_masks[word] = derived_masks.get(word, 0) | (1 << tag)
        if self.word_candidate_masks and self.word_candidate_masks != derived_masks:
            raise ValueError("invalid TnT word candidate masks")
        object.__setattr__(self, "word_candidate_masks", derived_masks)
        object.__setattr__(self, "known_words", frozenset(derived_masks))
        object.__setattr__(
            self,
            "word_candidates",
            {
                word: tuple(tag for tag in ALL_TAGS if mask & (1 << tag))
                for word, mask in derived_masks.items()
            },
        )
        object.__setattr__(self, "event_total", sum(self.event_counts))
        object.__setattr__(self, "tag_total_count", sum(self.tag_totals))
        object.__setattr__(
            self,
            "tag_vocabulary_sizes",
            tuple(
                sum(1 for _word, candidate in self.emissions if candidate == tag)
                for tag in range(len(TAGS))
            ),
        )

    def transition(self, first: int, second: int, event: int) -> float:
        """Interpolated add-alpha transition probability, including EOS."""

        if event not in EVENTS:
            raise ValueError("invalid transition event")
        alpha = self.config.transition_alpha
        unigram = (self.event_counts[event] + alpha) / (
            self.event_total + alpha * len(EVENTS)
        )
        bigram_total = self.bigram_totals[second]
        second_order = (self.bigram.get((second, event), 0) + alpha) / (
            bigram_total + alpha * len(EVENTS)
        )
        third_order = (self.trigram.get((first, second, event), 0) + alpha) / (
            self.context.get((first, second), 0) + alpha * len(EVENTS)
        )
        one, two, three = self.lambdas
        return one * unigram + two * second_order + three * third_order

    def _prior(self, tag: int) -> float:
        alpha = self.config.emission_alpha
        return (self.tag_totals[tag] + alpha) / (
            self.tag_total_count + alpha * len(TAGS)
        )

    def emission(self, form: str, tag: int) -> float:
        word = token_hash(form)
        alpha = self.config.emission_alpha
        # A seen word is always lexical, for every candidate tag.  Letting its
        # unattested tags use suffix evidence would leak the unknown-word path.
        if word in self.known_words:
            if not self.word_candidate_masks[word] & (1 << tag):
                return 0.0
            return (self.emissions[word, tag] + alpha) / (
                self.tag_totals[tag] + alpha * self.tag_vocabulary_sizes[tag]
            )
        prior = self._prior(tag)
        # TnT-style successive abstraction: longest observed suffix first, then
        # progressively shorter suffixes, then the tag prior.
        for length in range(min(self.config.suffix_length, len(form)), 0, -1):
            suffix = suffix_hash(form, length)
            total = self.suffix_totals.get((length, suffix), 0)
            if total:
                count = self.suffixes.get((length, suffix, tag), 0)
                beta = self.config.suffix_alpha
                return (count + beta * prior) / (total + beta)
        return prior

    def candidate_tags(self, form: str) -> tuple[int, ...]:
        """Return the mathematically non-zero lexical candidates for ``form``."""

        return self.word_candidates.get(token_hash(form), ALL_TAGS)

    def tag_sentence(self, forms: Sequence[str]) -> tuple[str, ...]:
        if not forms:
            return ()
        states: dict[tuple[int, int], tuple[float, int, tuple[int, int]]] = {
            (BOS, BOS): (0.0, 0, (BOS, BOS))
        }
        layers: list[dict[tuple[int, int], tuple[float, int, tuple[int, int]]]] = []
        for form in forms:
            next_states: dict[tuple[int, int], tuple[float, int, tuple[int, int]]] = {}
            for (first, second), (score, rank, _parent) in states.items():
                for tag in self.candidate_tags(form):
                    candidate = (
                        score
                        + math.log(self.transition(first, second, tag))
                        + math.log(self.emission(form, tag))
                    )
                    key = (second, tag)
                    current = next_states.get(key)
                    if (
                        current is None
                        or candidate > current[0]
                        or (candidate == current[0] and rank < current[1])
                    ):
                        next_states[key] = candidate, rank, (first, second)
            ordered = sorted(
                next_states,
                key=lambda key: (next_states[key][1], key[1]),
            )
            for rank, key in enumerate(ordered):
                score, _old_rank, parent = next_states[key]
                next_states[key] = score, rank, parent
            states = next_states
            layers.append(states)
        # EOS is a scored transition, so a sentence does not merely choose the
        # best pre-termination state.
        best_key = max(
            states,
            key=lambda key: (
                states[key][0] + math.log(self.transition(key[0], key[1], EOS)),
                -states[key][1],
            ),
        )
        tags = []
        key = best_key
        for layer in reversed(layers):
            tags.append(key[1])
            key = layer[key][2]
        return tuple(TAGS[tag] for tag in reversed(tags))

    def tag(
        self, documents: tuple[tuple[tuple[str, ...], ...], ...]
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        return tuple(
            tuple(self.tag_sentence(sentence) for sentence in document)
            for document in documents
        )

    def artifact(self) -> bytes:
        payload = {
            "schema": "remerge-pos-tnt-v2",
            "tags": TAGS,
            "config": asdict(self.config),
            "lambdas": self.lambdas,
            "event_counts": self.event_counts,
            "bigram": [[*key, value] for key, value in sorted(self.bigram.items())],
            "context": [[*key, value] for key, value in sorted(self.context.items())],
            "trigram": [[*key, value] for key, value in sorted(self.trigram.items())],
            "emissions": [
                [*key, value] for key, value in sorted(self.emissions.items())
            ],
            "tag_totals": self.tag_totals,
            "suffixes": [[*key, value] for key, value in sorted(self.suffixes.items())],
            "suffix_totals": [
                [*key, value] for key, value in sorted(self.suffix_totals.items())
            ],
            "vocabulary_size": self.vocabulary_size,
            "bigram_totals": self.bigram_totals,
            "word_candidate_masks": [
                [word, mask] for word, mask in sorted(self.word_candidate_masks.items())
            ],
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

    @classmethod
    def from_artifact(cls, artifact: bytes) -> "Model":
        try:

            def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    result[key] = value
                return result

            payload = json.loads(artifact, object_pairs_hook=reject_duplicates)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid TnT artifact JSON") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "remerge-pos-tnt-v2"
            or tuple(payload.get("tags", ())) != TAGS
        ):
            raise ValueError("invalid TnT artifact schema or tag inventory")
        try:
            config = Config(**payload["config"])
        except (KeyError, TypeError) as error:
            raise ValueError("invalid TnT artifact configuration") from error
        config.validate()

        def table(name: str, width: int) -> dict[tuple[int, ...], int]:
            rows = payload.get(name)
            if not isinstance(rows, list):
                raise ValueError(f"invalid {name} table")
            result: dict[tuple[int, ...], int] = {}
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != width + 1
                    or not all(
                        type(value) is int and value >= 0 for value in row[:-1]
                    )
                    or type(row[-1]) is not int
                    or row[-1] <= 0
                ):
                    raise ValueError(f"invalid {name} row")
                key, value = tuple(row[:-1]), row[-1]
                if key in result:
                    raise ValueError(f"duplicate {name} row")
                result[key] = value
            return result

        try:
            event_counts = tuple(payload["event_counts"])
            tag_totals = tuple(payload["tag_totals"])
            lambdas = tuple(payload["lambdas"])
            vocabulary_size = payload["vocabulary_size"]
            bigram_totals = tuple(payload["bigram_totals"])
            masks = table("word_candidate_masks", 1)
        except (KeyError, TypeError) as error:
            raise ValueError("invalid TnT aggregate counts") from error
        bigram = table("bigram", 2)
        context = table("context", 2)
        trigram = table("trigram", 3)
        emissions = table("emissions", 2)
        suffixes = table("suffixes", 3)
        suffix_totals = table("suffix_totals", 2)
        history_values = set(ALL_TAGS) | {BOS}
        if (
            any(
                first not in history_values or event not in EVENTS
                for first, event in bigram
            )
            or any(
                first not in history_values or second not in history_values
                for first, second in context
            )
            or any(
                first not in history_values
                or second not in history_values
                or event not in EVENTS
                for first, second, event in trigram
            )
            or any(word > MASK64 or tag not in ALL_TAGS for word, tag in emissions)
            or any(
                length not in range(1, config.suffix_length + 1) or tag not in ALL_TAGS
                for length, _suffix, tag in suffixes
            )
            or any(
                length not in range(1, config.suffix_length + 1)
                for length, _suffix in suffix_totals
            )
            or any(word > MASK64 for (word,), _mask in masks.items())
        ):
            raise ValueError("invalid TnT table structure")
        model = cls(
            config,
            lambdas,
            event_counts,
            bigram,
            context,
            trigram,
            emissions,
            tag_totals,
            suffixes,
            suffix_totals,
            vocabulary_size,
            bigram_totals,
            {word: mask for (word,), mask in masks.items()},
        )
        history_values = set(ALL_TAGS) | {BOS}
        valid_structure = (
            all(
                first in history_values and event in EVENTS
                for first, event in model.bigram
            )
            and all(
                first in history_values and second in history_values
                for first, second in model.context
            )
            and all(
                first in history_values and second in history_values and event in EVENTS
                for first, second, event in model.trigram
            )
            and all(0 <= tag < len(TAGS) for _word, tag in model.emissions)
            and all(
                1 <= length <= model.config.suffix_length and 0 <= tag < len(TAGS)
                for length, _suffix, tag in model.suffixes
            )
            and all(
                1 <= length <= model.config.suffix_length
                for length, _suffix in model.suffix_totals
            )
        )
        if (
            len(model.event_counts) != EVENT_COUNT
            or len(model.tag_totals) != len(TAGS)
            or len(model.lambdas) != 3
            or len(model.bigram_totals) != EVENT_COUNT
            or not all(
                type(value) is int and value >= 0 for value in model.bigram_totals
            )
            or not all(
                type(value) is int and value >= 0 for value in model.event_counts
            )
            or not all(type(value) is int and value >= 0 for value in model.tag_totals)
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
                for value in model.lambdas
            )
            or not math.isclose(sum(model.lambdas), 1.0)
            or type(model.vocabulary_size) is not int
            or model.vocabulary_size <= 0
            or model.event_counts[BOS] != 0
            or not valid_structure
            or any(word > MASK64 for word, _tag in model.emissions)
            or any(word > MASK64 for word in model.word_candidate_masks)
            or any(
                mask <= 0 or mask >= 1 << len(TAGS)
                for mask in model.word_candidate_masks.values()
            )
            or model.vocabulary_size != len(model.word_candidate_masks)
            or tuple(
                sum(
                    value
                    for (_word, tag), value in model.emissions.items()
                    if tag == index
                )
                for index in ALL_TAGS
            )
            != model.tag_totals
            or tuple(
                sum(
                    value
                    for (_history, event), value in model.bigram.items()
                    if event == index
                )
                for index in range(EVENT_COUNT)
            )
            != model.event_counts
            or any(
                sum(
                    value
                    for (first, current, event), value in model.trigram.items()
                    if first == history and current == second and event in EVENTS
                )
                != count
                for (history, second), count in model.context.items()
            )
            or any(
                sum(
                    value
                    for (first, current, candidate), value in model.trigram.items()
                    if current == history and candidate == event
                )
                != count
                for (history, event), count in model.bigram.items()
            )
            or any(
                sum(
                    value
                    for (
                        candidate_length,
                        candidate_suffix,
                        _tag,
                    ), value in model.suffixes.items()
                    if candidate_length == length and candidate_suffix == suffix
                )
                != count
                for (length, suffix), count in model.suffix_totals.items()
            )
            or any(
                value > model.tag_totals[tag]
                for (_length, _suffix, tag), value in model.suffixes.items()
            )
            or any(
                (first, second) not in model.context
                for first, second, _event in model.trigram
            )
            or any(
                (length, suffix) not in model.suffix_totals
                for length, suffix, _tag in model.suffixes
            )
        ):
            raise ValueError("invalid TnT aggregate counts")
        return model


def _deleted_interpolation(
    event_counts: Counter[int],
    bigram: Counter[tuple[int, int]],
    context: Counter[tuple[int, int]],
    trigram: Counter[tuple[int, int, int]],
) -> tuple[float, float, float]:
    """Estimate normalized interpolation weights by leave-one-out counts."""

    total = sum(event_counts.values())
    weights = [0, 0, 0]
    for (first, second, event), count in sorted(trigram.items()):
        third = (
            (count - 1) / (context[first, second] - 1)
            if context[first, second] > 1
            else -1
        )
        second_order = (
            (bigram[second, event] - 1)
            / (
                sum(
                    value for (history, _), value in bigram.items() if history == second
                )
                - 1
            )
            if sum(value for (history, _), value in bigram.items() if history == second)
            > 1
            else -1
        )
        first_order = (event_counts[event] - 1) / (total - 1) if total > 1 else -1
        # Ties intentionally favour the highest-order available estimate.
        winner = max(
            range(3),
            key=lambda index: ((first_order, second_order, third)[index], index),
        )
        weights[winner] += count
    weight_total = sum(weights)
    if not weight_total:
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(weight / weight_total for weight in weights)  # type: ignore[return-value]


def train(sentences: Iterable[Sequence[tuple[str, str]]], config: Config) -> Model:
    config.validate()
    data = tuple(tuple(sentence) for sentence in sentences)
    word_counts: Counter[str] = Counter()
    hash_sources: dict[int, str] = {}
    for sentence in data:
        for form, label in sentence:
            if label not in TAG_INDEX:
                raise ValueError(f"unsupported UPOS {label!r}")
            hashed = token_hash(form)
            existing = hash_sources.setdefault(hashed, form)
            if existing != form:
                raise ValueError("token hash collision; refusing ambiguous artifact")
            word_counts[form] += 1
    if not word_counts:
        raise ValueError("cannot train TnT model on no tokens")

    event_counts: Counter[int] = Counter()
    bigram: Counter[tuple[int, int]] = Counter()
    context: Counter[tuple[int, int]] = Counter()
    trigram: Counter[tuple[int, int, int]] = Counter()
    emissions: Counter[tuple[int, int]] = Counter()
    suffixes: Counter[tuple[int, int, int]] = Counter()
    suffix_totals: Counter[tuple[int, int]] = Counter()
    for sentence in data:
        first, second = BOS, BOS
        for form, label in sentence:
            tag = TAG_INDEX[label]
            event_counts[tag] += 1
            bigram[second, tag] += 1
            context[first, second] += 1
            trigram[first, second, tag] += 1
            emissions[token_hash(form), tag] += 1
            if word_counts[form] <= config.rare_word_max_count:
                for length in range(1, min(config.suffix_length, len(form)) + 1):
                    suffix = suffix_hash(form, length)
                    suffixes[length, suffix, tag] += 1
                    suffix_totals[length, suffix] += 1
            first, second = second, tag
        # Count and score termination in the same second-order model.
        event_counts[EOS] += 1
        bigram[second, EOS] += 1
        context[first, second] += 1
        trigram[first, second, EOS] += 1

    tag_totals = tuple(
        sum(value for (_word, tag), value in emissions.items() if tag == index)
        for index in range(len(TAGS))
    )
    return Model(
        config=config,
        lambdas=_deleted_interpolation(event_counts, bigram, context, trigram),
        event_counts=tuple(event_counts[index] for index in range(EVENT_COUNT)),
        bigram=dict(bigram),
        context=dict(context),
        trigram=dict(trigram),
        emissions=dict(emissions),
        tag_totals=tag_totals,
        suffixes=dict(suffixes),
        suffix_totals=dict(suffix_totals),
        vocabulary_size=len(word_counts),
    )


def write_artifact(model: Model, path: Path) -> str:
    artifact = model.artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(artifact)
    return hashlib.sha256(artifact).hexdigest()

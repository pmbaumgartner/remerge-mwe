#!/usr/bin/env python3
"""Train the isolated linear POS research model.

The script intentionally accepts only project-controlled train and development
CoNLL-U inputs.  It has no download path and deliberately has no ``--final``
option: final evaluation belongs to the release oracle, not model fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import struct
import unicodedata
from typing import Iterable, Mapping, Sequence


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
MAGIC = b"RMPOS001"
VERSION = 1
FEATURE_LIMIT = 22
HEADER = struct.Struct("<8sHHIHHHHIII")
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
DIRECT_TARGET_ACCURACY = 0.99
FEATURE_SCHEMA = (
    "W={form}",
    "L={ascii_lower(form)}",
    "S={shape(form)}",
    "P1..P4={ascii_lower(form)[:width]}",
    "U1..U4={ascii_lower(form)[-width:]}",
    "PL={ascii_lower(previous)}",
    "PS={shape(previous)}",
    "NL={ascii_lower(following)}",
    "NS={shape(following)}",
    "PC={ascii_lower(previous)}|{ascii_lower(form)}",
    "CN={ascii_lower(form)}|{ascii_lower(following)}",
    "F=upper|title|digit|hyphen|apostrophe when true",
)
PUNCTUATION = frozenset("!\"'(),-./:;?[]_{}\\«»‐‑‒–—―‘’‚‛“”„‟…")
NUMBER_CONNECTORS = frozenset("+-−.,_/%‰eE")
HYPHENS = frozenset("-‐‑‒–—−")
APOSTROPHES = frozenset("'’")


@dataclass(frozen=True)
class Token:
    form: str
    tag: int


@dataclass(frozen=True)
class Example:
    tokens: tuple[Token, ...]


def fnv1a(value: str) -> int:
    result = FNV_OFFSET
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * FNV_PRIME) & MASK64
    return result


def read_conllu(path: Path) -> tuple[Example, ...]:
    """Read canonical integer-word CoNLL-U, ignoring comments and MWT rows."""
    sentences: list[Example] = []
    current: list[Token] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if not line:
            if current:
                sentences.append(Example(tuple(current)))
                current = []
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError(f"{path}:{number}: expected 10 CoNLL-U fields")
        word_id, form, upos = fields[0], fields[1], fields[3]
        if "-" in word_id or "." in word_id:
            continue
        if not word_id.isdecimal() or int(word_id) < 1:
            raise ValueError(f"{path}:{number}: expected positive integer word ID")
        if unicodedata.normalize("NFC", form) != form:
            raise ValueError(f"{path}:{number}: FORM must be NFC-normalized")
        if upos not in TAG_INDEX:
            raise ValueError(f"{path}:{number}: unsupported UPOS {upos!r}")
        current.append(Token(form, TAG_INDEX[upos]))
    if current:
        sentences.append(Example(tuple(current)))
    if not sentences:
        raise ValueError(f"{path}: no canonical CoNLL-U words")
    return tuple(sentences)


def shape(form: str) -> str:
    if form in {"<BOS>", "<EOS>"}:
        return form
    return "".join(
        "X"
        if char.isupper()
        else "x"
        if char.islower() or char.isalpha()
        else "d"
        if char.isnumeric()
        else char
        for char in form
    )


def lexical_class(form: str) -> str:
    if all(is_punctuation(char) for char in form):
        return "punct"
    if is_number_like(form):
        return "number"
    if all(is_symbol(char) for char in form):
        return "symbol"
    if form.isupper():
        return "upper"
    if form.istitle():
        return "title"
    if form.islower():
        return "lower"
    return "mixed"


def is_punctuation(character: str) -> bool:
    return character in PUNCTUATION


def is_number_like(form: str) -> bool:
    seen_digit = False
    for character in form:
        if character.isnumeric():
            seen_digit = True
        elif character not in NUMBER_CONNECTORS:
            return False
    return seen_digit


def is_symbol(character: str) -> bool:
    return (
        not character.isspace()
        and not character.isalnum()
        and not is_punctuation(character)
        and unicodedata.category(character) != "Cc"
    )


def forced_tag(form: str) -> int | None:
    token_class = lexical_class(form)
    if token_class == "punct":
        return TAG_INDEX["PUNCT"]
    if token_class == "number":
        return TAG_INDEX["NUM"]
    if token_class == "symbol":
        return TAG_INDEX["SYM"]
    return None


def ascii_lower(value: str) -> str:
    if value in {"<BOS>", "<EOS>"}:
        return value
    return value.translate(
        str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    )


def is_title(form: str) -> bool:
    alphabetic = [char for char in form if char.isalpha()]
    return (
        len(alphabetic) >= 2
        and alphabetic[0].isupper()
        and all(char.islower() for char in alphabetic[1:])
    )


def features(forms: tuple[str, ...], index: int) -> tuple[str, ...]:
    """The 22 fixed feature families; false flags consume no feature slot."""
    form = forms[index]
    lower = ascii_lower(form)
    previous = forms[index - 1] if index else "<BOS>"
    following = forms[index + 1] if index + 1 < len(forms) else "<EOS>"
    previous_ascii = ascii_lower(previous)
    following_ascii = ascii_lower(following)
    result = [f"W={form}", f"L={lower}", f"S={shape(form)}"]
    result.extend(f"P{width}={lower[:width]}" for width in range(1, 5))
    result.extend(f"U{width}={lower[-width:]}" for width in range(1, 5))
    result.extend(
        (
            f"PL={previous_ascii}",
            f"PS={shape(previous)}",
            f"NL={following_ascii}",
            f"NS={shape(following)}",
            f"PC={previous_ascii}|{lower}",
            f"CN={lower}|{following_ascii}",
        )
    )
    flags = (
        ("upper", any(char.isupper() for char in form)),
        ("title", is_title(form)),
        ("digit", any(char.isnumeric() for char in form)),
        ("hyphen", any(char in HYPHENS for char in form)),
        ("apostrophe", any(char in APOSTROPHES for char in form)),
    )
    result.extend(f"F={name}" for name, enabled in flags if enabled)
    assert len(result) <= FEATURE_LIMIT
    return tuple(result)


def feature_buckets(
    forms: tuple[str, ...], index: int, buckets: int
) -> tuple[int, ...]:
    return tuple(fnv1a(feature) & (buckets - 1) for feature in features(forms, index))


class AveragedPerceptron:
    def __init__(self, buckets: int) -> None:
        self.buckets = buckets
        self.weights: dict[tuple[int, int], int] = {}
        self.totals: dict[tuple[int, int], int] = {}
        self.timestamps: dict[tuple[int, int], int] = {}
        self.biases = [0] * len(TAGS)
        self.bias_totals = [0] * len(TAGS)
        self.bias_timestamps = [0] * len(TAGS)
        self.step = 0

    def _touch(self, key: tuple[int, int]) -> None:
        previous = self.timestamps.get(key, 0)
        self.totals[key] = self.totals.get(key, 0) + (
            self.step - previous
        ) * self.weights.get(key, 0)
        self.timestamps[key] = self.step

    def _touch_bias(self, tag: int) -> None:
        self.bias_totals[tag] += (self.step - self.bias_timestamps[tag]) * self.biases[
            tag
        ]
        self.bias_timestamps[tag] = self.step

    def score(self, bucket_ids: Iterable[int], tag: int) -> int:
        return self.biases[tag] + sum(
            self.weights.get((bucket, tag), 0) for bucket in bucket_ids
        )

    def predict(
        self, bucket_ids: tuple[int, ...], allowed: Iterable[int] = range(17)
    ) -> int:
        return max(allowed, key=lambda tag: (self.score(bucket_ids, tag), -tag))

    def update(self, bucket_ids: tuple[int, ...], truth: int, prediction: int) -> None:
        self.step += 1
        if truth == prediction:
            return
        for tag, delta in ((truth, 1), (prediction, -1)):
            self._touch_bias(tag)
            self.biases[tag] += delta
            for bucket in bucket_ids:
                key = (bucket, tag)
                self._touch(key)
                self.weights[key] = self.weights.get(key, 0) + delta

    def averaged(self) -> tuple[list[float], dict[tuple[int, int], float]]:
        denominator = max(self.step, 1)
        biases: list[float] = []
        for tag, current in enumerate(self.biases):
            total = (
                self.bias_totals[tag]
                + (self.step - self.bias_timestamps[tag]) * current
            )
            biases.append(total / denominator)
        values: dict[tuple[int, int], float] = {}
        for key, current in self.weights.items():
            total = (
                self.totals.get(key, 0)
                + (self.step - self.timestamps.get(key, 0)) * current
            )
            average = total / denominator
            if average:
                values[key] = average
        return biases, values


def train(
    examples: tuple[Example, ...],
    epochs: int,
    buckets: int,
    seed: int,
    candidates: Mapping[int, int],
) -> AveragedPerceptron:
    model = AveragedPerceptron(buckets)
    order = list(range(len(examples)))
    random_source = random.Random(seed)
    for _ in range(epochs):
        random_source.shuffle(order)
        for example_index in order:
            example = examples[example_index]
            forms = tuple(token.form for token in example.tokens)
            for index, token in enumerate(example.tokens):
                forced = forced_tag(token.form)
                if forced is not None:
                    continue
                bucket_ids = feature_buckets(forms, index, buckets)
                mask = candidates.get(fnv1a(token.form))
                allowed = allowed_tags(mask) if mask is not None else range(17)
                model.update(
                    bucket_ids,
                    token.tag,
                    model.predict(bucket_ids, allowed),
                )
    return model


def lexicons(
    examples: tuple[Example, ...],
) -> tuple[dict[int, tuple[int, int]], dict[int, int], int]:
    forms: dict[int, set[str]] = defaultdict(set)
    labels: dict[int, Counter[int]] = defaultdict(Counter)
    for example in examples:
        for token in example.tokens:
            key = fnv1a(token.form)
            forms[key].add(token.form)
            labels[key][token.tag] += 1
    collisions = {key for key, variants in forms.items() if len(variants) > 1}
    direct: dict[int, tuple[int, int]] = {}
    candidates: dict[int, int] = {}
    for key, counts in labels.items():
        if key in collisions:
            continue
        mask = sum(1 << tag for tag in counts)
        if mask & (mask - 1):
            candidates[key] = mask
        else:
            direct[key] = (next(iter(counts)), sum(counts.values()))
    return direct, candidates, len(collisions)


def select_direct_lexicon(
    train_direct: Mapping[int, tuple[int, int]],
    dev_examples: tuple[Example, ...],
) -> tuple[dict[int, int], int | None, float | None, float]:
    """Pick the most-covering pure train lexicon threshold meeting the dev gate."""
    thresholds = sorted({support for _, support in train_direct.values()})
    best: tuple[dict[int, int], int, float, float] | None = None
    dev_tokens = sum(len(example.tokens) for example in dev_examples)
    for threshold in thresholds:
        trial = {
            key: tag
            for key, (tag, support) in train_direct.items()
            if support >= threshold
        }
        accepted = correct = 0
        for example in dev_examples:
            for token in example.tokens:
                if forced_tag(token.form) is not None:
                    continue
                tag = trial.get(fnv1a(token.form))
                if tag is not None:
                    accepted += 1
                    correct += tag == token.tag
        if not accepted:
            continue
        accepted_accuracy = correct / accepted
        coverage = accepted / dev_tokens
        if accepted_accuracy >= DIRECT_TARGET_ACCURACY and (
            best is None
            or coverage > best[3]
            or (coverage == best[3] and threshold < best[1])
        ):
            best = (trial, threshold, accepted_accuracy, coverage)
    if best is None:
        return {}, None, None, 0.0
    return best


def allowed_tags(mask: int) -> tuple[int, ...]:
    return tuple(tag for tag in range(17) if mask & (1 << tag))


def predict(
    biases: Sequence[float],
    weights: Mapping[tuple[int, int], float],
    examples: tuple[Example, ...],
    buckets: int,
    direct: dict[int, int] | None = None,
    candidates: dict[int, int] | None = None,
) -> tuple[list[int], int, int]:
    output: list[int] = []
    direct_total = direct_correct = 0
    for example in examples:
        forms = tuple(token.form for token in example.tokens)
        for index, token in enumerate(example.tokens):
            forced = forced_tag(token.form)
            key = fnv1a(token.form)
            direct_tag = direct.get(key) if direct is not None else None
            if forced is not None:
                result = forced
            elif direct_tag is not None:
                result = direct_tag
                direct_total += 1
                direct_correct += result == token.tag
            else:
                ids = feature_buckets(forms, index, buckets)
                pool = range(17)
                if candidates is not None and key in candidates:
                    pool = allowed_tags(candidates[key])
                result = max(
                    pool,
                    key=lambda tag: (
                        biases[tag]
                        + sum(weights.get((bucket, tag), 0) for bucket in ids),
                        -tag,
                    ),
                )
            output.append(result)
    return output, direct_total, direct_correct


def accuracy(predictions: list[int], examples: tuple[Example, ...]) -> float:
    truth = [token.tag for example in examples for token in example.tokens]
    return sum(
        prediction == expected
        for prediction, expected in zip(predictions, truth, strict=True)
    ) / len(truth)


def quantize(
    biases: list[float], weights: dict[tuple[int, int], float]
) -> tuple[list[int], dict[tuple[int, int], int], float]:
    max_weight = max((abs(value) for value in weights.values()), default=0.0)
    max_bias = max((abs(value) for value in biases), default=0.0)
    limits = []
    if max_weight:
        limits.append(127.0 / max_weight)
    if max_bias:
        limits.append(32767.0 / max_bias)
    scale = min(limits) if limits else 1.0
    return (
        [max(-32768, min(32767, round(value * scale))) for value in biases],
        {
            key: max(-128, min(127, round(value * scale)))
            for key, value in weights.items()
        },
        scale,
    )


def compile_model(
    path: Path,
    *,
    biases: list[int],
    weights: dict[tuple[int, int], int],
    buckets: int,
    direct: dict[int, int],
    candidates: dict[int, int],
    model_id: str,
    tokenizer_id: str,
) -> str:
    model_bytes = model_id.encode("utf-8")
    tokenizer_bytes = tokenizer_id.encode("utf-8")
    if len(model_bytes) > 65535 or len(tokenizer_bytes) > 65535:
        raise ValueError("model_id and tokenizer_id must fit u16")
    flat = bytearray(buckets * 17)
    for (bucket, tag), value in weights.items():
        flat[bucket * 17 + tag] = value & 0xFF
    header = HEADER.pack(
        MAGIC,
        VERSION,
        17,
        buckets,
        FEATURE_LIMIT,
        0,
        len(model_bytes),
        len(tokenizer_bytes),
        len(direct),
        len(candidates),
        len(flat),
    )
    payload = bytearray(header + model_bytes + tokenizer_bytes)
    payload.extend(struct.pack("<17h", *biases))
    for key, tag in sorted(direct.items()):
        payload.extend(struct.pack("<QB", key, tag))
    for key, mask in sorted(candidates.items()):
        payload.extend(struct.pack("<QI", key, mask))
    payload.extend(flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def registration_template(
    args: argparse.Namespace, digest: str
) -> dict[str, str | int | bool]:
    return {
        "candidate_id": args.candidate_id,
        "source_revision": args.source_revision,
        "trainer_command": " ".join(__import__("sys").argv),
        "seed": args.seed,
        "model_id": args.model_id,
        "artifact_sha256": digest,
        "dev_report": str(args.report),
        "final_evaluated": False,
        "tokenizer_id": args.tokenizer_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, required=True, help="canonical CoNLL-U training split"
    )
    parser.add_argument(
        "--dev", type=Path, required=True, help="canonical CoNLL-U development split"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--registration-template", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--buckets", type=int, default=1 << 18)
    parser.add_argument("--candidate-id", default="remerge-pos-linear-candidate")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-id", default="remerge-pos-linear-v1")
    parser.add_argument("--tokenizer-id", default="unicode-whitespace-v1")
    arguments = parser.parse_args()
    if arguments.epochs < 1:
        parser.error("--epochs must be positive")
    if arguments.buckets < 2 or arguments.buckets & (arguments.buckets - 1):
        parser.error("--buckets must be a power of two")
    return arguments


def main() -> None:
    args = parse_args()
    train_examples = read_conllu(args.train)
    dev_examples = read_conllu(args.dev)
    train_direct, candidates, collision_count = lexicons(train_examples)
    model = train(
        train_examples,
        args.epochs,
        args.buckets,
        args.seed,
        candidates,
    )
    biases, weights = model.averaged()
    direct, direct_threshold, direct_accuracy, direct_coverage = select_direct_lexicon(
        train_direct, dev_examples
    )
    full, _, _ = predict(biases, weights, dev_examples, args.buckets)
    quantized_biases, quantized_weights, quantization_scale = quantize(biases, weights)
    quantized, _, _ = predict(
        quantized_biases, quantized_weights, dev_examples, args.buckets
    )
    pruned, direct_total, direct_correct = predict(
        quantized_biases,
        quantized_weights,
        dev_examples,
        args.buckets,
        direct,
        candidates,
    )
    full_accuracy = accuracy(full, dev_examples)
    quantized_accuracy = accuracy(quantized, dev_examples)
    pruned_accuracy = accuracy(pruned, dev_examples)
    quantization_loss = full_accuracy - quantized_accuracy
    pruning_loss = quantized_accuracy - pruned_accuracy
    if quantization_loss > 0.0025 or pruning_loss > 0.0025:
        raise SystemExit(
            "rejected: quantization or candidate pruning costs more than 0.25 percentage points"
        )
    digest = compile_model(
        args.output,
        biases=quantized_biases,
        weights=quantized_weights,
        buckets=args.buckets,
        direct=direct,
        candidates=candidates,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
    )
    token_count = sum(len(example.tokens) for example in dev_examples)
    report = {
        "schema": "remerge-pos-linear-report-v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "bucket_count": args.buckets,
        "feature_limit": FEATURE_LIMIT,
        "feature_schema": FEATURE_SCHEMA,
        "feature_hash": "fnv1a-64",
        "tag_order": TAGS,
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
        "artifact_sha256": digest,
        "dev_tokens": token_count,
        "dev_accuracy_full_precision_all_labels": full_accuracy,
        "dev_accuracy_quantized_all_labels": quantized_accuracy,
        "dev_accuracy_quantized_candidate_pruned": pruned_accuracy,
        "quantization_loss_percentage_points": quantization_loss * 100,
        "candidate_pruning_loss_percentage_points": pruning_loss * 100,
        "quantization_scale": quantization_scale,
        "direct_lexical_target_accuracy": DIRECT_TARGET_ACCURACY,
        "direct_lexical_support_threshold": direct_threshold,
        "direct_lexical_coverage": direct_coverage,
        "direct_lexical_accuracy": direct_accuracy,
        "direct_lexical_accepted_tokens": direct_total,
        "direct_lexical_correct_tokens": direct_correct,
        "direct_entries": len(direct),
        "candidate_entries": len(candidates),
        "removed_hash_collisions": collision_count,
        "command": " ".join(__import__("sys").argv),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.registration_template.parent.mkdir(parents=True, exist_ok=True)
    args.registration_template.write_text(
        json.dumps(
            registration_template(args, digest),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

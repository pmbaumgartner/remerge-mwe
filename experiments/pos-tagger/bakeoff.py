#!/usr/bin/env python3
"""Run a development-only, candidate-neutral POS bakeoff.

This is research infrastructure, not a product API.  It intentionally loads
only the frozen train/dev inputs; there is no final-split flag or fallback
path.  A conforming adapter is a zero-argument ``MODULE:FACTORY`` whose object
has immutable ``model_id``, ``tokenizer_id``, ``artifact_bytes`` and
``artifact_sha256`` attributes plus ``tag(boundaries)``.  ``tag`` must return
the identical document/sentence/token nesting, replacing each token with one
of the 17 UPOS strings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Protocol, Sequence

# The harness is deliberately executable as a checked-out research script; its
# project-owned loader and utility sensors are not installed package modules.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import remerge  # noqa: E402
from tests.pos.conftest import calculate_utility_metrics, exact_occurrences  # noqa: E402
from tests.pos.evaluation.loader import (  # noqa: E402
    GoldSplit,
    Sentence,
    load_gold_split,
    load_manifest,
    load_tagged_split,
)


UPOS = frozenset(
    {
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
    }
)
RETAINED_C2_ACCURACY = 0.937691
RETAINED_C2_TOLERANCE = 0.000_001
RETAINED_C2_TRAIN_SHA256 = (
    "e6a3784727e7726d4f1c2e10ff22dcd0ddeb8869ad6f85b73bb1ff5ef798e944"
)
RETAINED_C2_DEV_SHA256 = (
    "ef962ac05d844eaff46eeded125937129bfc0876d43963d66810cb73ffa8f5df"
)

Boundaries = tuple[tuple[tuple[str, ...], ...], ...]
Predictions = tuple[tuple[tuple[str, ...], ...], ...]


class Candidate(Protocol):
    model_id: str
    tokenizer_id: str
    artifact_bytes: int
    artifact_sha256: str
    artifact_path: Path

    def tag(self, documents: Boundaries) -> Predictions: ...


class RejectedEvaluation(ValueError):
    """A candidate or frozen-development invariant was violated."""


@dataclass(frozen=True)
class CandidateRegistration:
    candidate_id: str
    source_revision: str
    command: str
    seed: int
    data_hashes: dict[str, str]
    artifact_sha256: str
    environment: dict[str, str]
    model_id: str
    tokenizer_id: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(specification: str) -> Candidate:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise RejectedEvaluation("candidate must be MODULE:FACTORY")
    candidate = getattr(importlib.import_module(module_name), factory_name)()
    for name in (
        "model_id",
        "tokenizer_id",
        "artifact_bytes",
        "artifact_sha256",
        "artifact_path",
        "tag",
    ):
        if not hasattr(candidate, name):
            raise RejectedEvaluation(f"candidate lacks required {name!r}")
    if not isinstance(candidate.model_id, str) or not candidate.model_id:
        raise RejectedEvaluation("candidate model_id must be a non-empty string")
    if not isinstance(candidate.tokenizer_id, str) or not candidate.tokenizer_id:
        raise RejectedEvaluation("candidate tokenizer_id must be a non-empty string")
    if not isinstance(candidate.artifact_bytes, int) or candidate.artifact_bytes < 0:
        raise RejectedEvaluation(
            "candidate artifact_bytes must be a non-negative integer"
        )
    if (
        not isinstance(candidate.artifact_sha256, str)
        or len(candidate.artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate.artifact_sha256
        )
    ):
        raise RejectedEvaluation("candidate artifact_sha256 must be a SHA-256 digest")
    if (
        not isinstance(candidate.artifact_path, Path)
        or not candidate.artifact_path.is_file()
    ):
        raise RejectedEvaluation(
            "candidate artifact_path must name an existing artifact"
        )
    actual_digest = _sha(candidate.artifact_path)
    actual_bytes = candidate.artifact_path.stat().st_size
    if (
        candidate.artifact_sha256 != actual_digest
        or candidate.artifact_bytes != actual_bytes
    ):
        raise RejectedEvaluation(
            "candidate self-reported artifact metadata differs from artifact bytes"
        )
    return candidate


def load_registration(
    path: Path, candidate: Candidate, *, data_hashes: dict[str, str]
) -> CandidateRegistration:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RejectedEvaluation(
            f"cannot read candidate registration: {error}"
        ) from error
    required = {
        "candidate_id",
        "source_revision",
        "command",
        "seed",
        "data_hashes",
        "artifact_sha256",
        "environment",
        "model_id",
        "tokenizer_id",
    }
    if not isinstance(raw, dict) or required - raw.keys():
        raise RejectedEvaluation(
            "candidate registration lacks required provenance fields"
        )
    if (
        raw["model_id"] != candidate.model_id
        or raw["tokenizer_id"] != candidate.tokenizer_id
    ):
        raise RejectedEvaluation(
            "candidate registration model identity differs from adapter"
        )
    if raw["artifact_sha256"] != candidate.artifact_sha256:
        raise RejectedEvaluation("candidate registration digest differs from adapter")
    if raw["data_hashes"] != data_hashes:
        raise RejectedEvaluation(
            "candidate registration data hashes differ from frozen development inputs"
        )
    if (
        not isinstance(raw["candidate_id"], str)
        or not raw["candidate_id"]
        or not isinstance(raw["source_revision"], str)
        or len(raw["source_revision"]) != 40
        or any(character not in "0123456789abcdef" for character in raw["source_revision"])
        or not isinstance(raw["command"], str)
        or not raw["command"]
        or not isinstance(raw["seed"], int)
        or not isinstance(raw["environment"], dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw["environment"].items()
        )
        or not isinstance(raw["data_hashes"], dict)
        or set(raw["data_hashes"]) != {"train", "dev"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in raw["data_hashes"].values()
        )
    ):
        raise RejectedEvaluation(
            "candidate registration provenance fields are malformed"
        )
    return CandidateRegistration(**{name: raw[name] for name in required})


def boundaries(sentences: Sequence[Sentence]) -> Boundaries:
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for sentence in sentences:
        grouped.setdefault(sentence.document_id, []).append(
            tuple(token.form for token in sentence.tokens)
        )
    return tuple(tuple(value) for value in grouped.values())


def validate_predictions(expected: Boundaries, actual: object) -> Predictions:
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise RejectedEvaluation("candidate changed the document count")
    checked: list[tuple[tuple[str, ...], ...]] = []
    for document_index, (wanted_document, actual_document) in enumerate(
        zip(expected, actual)
    ):
        if not isinstance(actual_document, (list, tuple)) or len(
            actual_document
        ) != len(wanted_document):
            raise RejectedEvaluation(
                f"candidate changed sentence boundaries in document {document_index}"
            )
        document: list[tuple[str, ...]] = []
        for sentence_index, (wanted_sentence, actual_sentence) in enumerate(
            zip(wanted_document, actual_document)
        ):
            if not isinstance(actual_sentence, (list, tuple)) or len(
                actual_sentence
            ) != len(wanted_sentence):
                raise RejectedEvaluation(
                    f"candidate changed token count at {document_index}:{sentence_index}"
                )
            tags = tuple(actual_sentence)
            if any(tag not in UPOS for tag in tags):
                raise RejectedEvaluation(
                    f"candidate emitted invalid UPOS at {document_index}:{sentence_index}"
                )
            document.append(tags)
        checked.append(tuple(document))
    return tuple(checked)


def subset_predictions(
    source: Sequence[Sentence],
    predicted: Predictions,
    target: Sequence[Sentence],
) -> Predictions:
    """Select aligned sentence predictions without retagging or retokenizing."""

    source_documents: dict[str, list[Sentence]] = {}
    for sentence in source:
        source_documents.setdefault(sentence.document_id, []).append(sentence)
    by_sentence: dict[tuple[str, str], tuple[str, ...]] = {}
    for document_index, sentences in enumerate(source_documents.values()):
        for sentence_index, sentence in enumerate(sentences):
            by_sentence[(sentence.document_id, sentence.sentence_id)] = predicted[
                document_index
            ][sentence_index]

    target_documents: dict[str, list[tuple[str, ...]]] = {}
    for sentence in target:
        key = sentence.document_id, sentence.sentence_id
        try:
            tags = by_sentence[key]
        except KeyError as error:
            raise RejectedEvaluation(
                f"MWE development sentence {sentence.sentence_id!r} lacks a prediction"
            ) from error
        target_documents.setdefault(sentence.document_id, []).append(tags)
    return tuple(tuple(sentences) for sentences in target_documents.values())


def _flat_gold(sentences: Sequence[Sentence]) -> list[tuple[str, str, str]]:
    return [
        (token.form, token.upos, token.domain)
        for sentence in sentences
        for token in sentence.tokens
    ]


def _flat_predicted(predicted: Predictions) -> list[str]:
    return [tag for document in predicted for sentence in document for tag in sentence]


def quality_metrics(
    train: Sequence[Sentence], dev: Sequence[Sentence], predicted: Predictions
) -> dict[str, Any]:
    train_tags: dict[str, set[str]] = {}
    for form, tag, _domain in _flat_gold(train):
        train_tags.setdefault(form, set()).add(tag)
    gold = _flat_gold(dev)
    actual = _flat_predicted(predicted)
    if len(gold) != len(actual):
        raise RejectedEvaluation(
            "prediction stream length differs from development gold"
        )
    labels = sorted({tag for _form, tag, _domain in gold})

    def accuracy(items: list[tuple[str, str]]) -> float | None:
        return (
            None
            if not items
            else sum(left == right for left, right in items) / len(items)
        )

    pairs = [
        (tag, predicted_tag)
        for (_form, tag, _domain), predicted_tag in zip(gold, actual)
    ]

    def f1(label: str) -> float:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        return 0.0 if not (2 * tp + fp + fn) else 2 * tp / (2 * tp + fp + fn)

    domains: dict[str, list[tuple[str, str]]] = {}
    oov: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    for (form, tag, domain), guessed in zip(gold, actual):
        pair = (tag, guessed)
        domains.setdefault(domain, []).append(pair)
        if form not in train_tags:
            oov.append(pair)
        if len(train_tags.get(form, set())) > 1:
            ambiguous.append(pair)
    return {
        "overall_accuracy": accuracy(pairs),
        "macro_f1": statistics.fmean(f1(label) for label in labels),
        "oov_accuracy": accuracy(oov),
        "ambiguous_accuracy": accuracy(ambiguous),
        "per_domain_accuracy": {
            domain: accuracy(items) for domain, items in domains.items()
        },
        "supported_tag_f1": {
            label: {
                "count": sum(tag == label for _form, tag, _domain in gold),
                "f1": f1(label),
            }
            for label in labels
        },
    }


def _documents(
    split: GoldSplit, predicted: Predictions | None, model_id: str
) -> tuple[list[str], list[remerge.TaggedDocument], set[tuple[int, int, int, int]]]:
    by_doc: dict[str, list[Sentence]] = {}
    for sentence in split.sentences:
        by_doc.setdefault(sentence.document_id, []).append(sentence)
    raw: list[str] = []
    tagged: list[remerge.TaggedDocument] = []
    coordinates: dict[tuple[str, str], tuple[int, int]] = {}
    for document_index, sentences in enumerate(by_doc.values()):
        raw.append(
            "\n".join(
                " ".join(token.form for token in sentence.tokens)
                for sentence in sentences
            )
        )
        output_sentences = []
        for sentence_index, sentence in enumerate(sentences):
            coordinates[(sentence.document_id, sentence.sentence_id)] = (
                document_index,
                sentence_index,
            )
            tags = (
                predicted[document_index][sentence_index]
                if predicted
                else tuple(token.upos for token in sentence.tokens)
            )
            output_sentences.append(
                tuple(
                    remerge.TaggedToken(token.form, tag)
                    for token, tag in zip(sentence.tokens, tags)
                )
            )
        # ``TaggedDocument`` is an existing supplied-tag input type.  The
        # adapter identity stays in bakeoff evidence rather than becoming a
        # product-facing tagger/model field.
        tagged.append(remerge.TaggedDocument(tuple(output_sentences)))
    spans = {
        (
            coordinates[span.document_id, span.sentence_id][0],
            coordinates[span.document_id, span.sentence_id][1],
            span.start_token,
            span.end_token,
        )
        for span in split.mwe_spans
    }
    return raw, tagged, spans


def utility_metrics(
    split: GoldSplit,
    predicted: Predictions,
    model_id: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    raw, generated, gold = _documents(split, predicted, model_id)
    _raw, canonical, _gold = _documents(split, None, "gold-dev")
    arguments = {
        "method": configuration["score_method"],
        "min_count": configuration["min_count"],
        "min_score": configuration["min_score"],
        "on_exhausted": configuration["on_exhausted"],
    }
    iterations = configuration["requested_winners"]
    unfiltered = exact_occurrences(
        remerge.run_with_occurrences(raw, iterations, **arguments)
    )
    gold_filtered = exact_occurrences(
        remerge.run_tagged(
            canonical,
            iterations,
            patterns=[tuple(item) for item in configuration["patterns"]],
            **arguments,
        )
    )
    generated_filtered = exact_occurrences(
        remerge.run_tagged(
            generated,
            iterations,
            patterns=[tuple(item) for item in configuration["patterns"]],
            **arguments,
        )
    )
    result = calculate_utility_metrics(
        unfiltered,
        generated_filtered,
        gold_filtered,
        gold,
        document_count=len(generated),
        resamples=10_000,
    )
    return asdict(result) | {"bootstrap_resamples": 10_000}


def _measurement(values: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    lower = ordered[: len(ordered) // 2]
    upper = ordered[(len(ordered) + 1) // 2 :]
    iqr = statistics.median(upper) - statistics.median(lower)
    return {
        "seconds": values,
        "median_seconds": median,
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "iqr_over_median": iqr / median if median else 0.0,
    }


def warm_inference_measurements(
    candidate: Candidate,
    inputs: Boundaries,
    expected: Predictions,
) -> dict[str, float | list[float]]:
    """Time only inference while rejecting stateful or malformed later output."""

    for _ in range(3):
        actual = validate_predictions(inputs, candidate.tag(inputs))
        if actual != expected:
            raise RejectedEvaluation("candidate predictions changed during warmup")
    timings: list[float] = []
    for _ in range(15):
        started = perf_counter()
        raw = candidate.tag(inputs)
        timings.append(perf_counter() - started)
        actual = validate_predictions(inputs, raw)
        if actual != expected:
            raise RejectedEvaluation("candidate predictions changed during measurement")
    return _measurement(timings)


def isolated_load_measurements(
    specification: str, expected_artifact_sha256: str
) -> dict[str, Any]:
    """Measure factory load in new processes and normalize RSS to bytes."""
    module, separator, factory = specification.partition(":")
    if not separator:
        raise RejectedEvaluation("candidate must be MODULE:FACTORY")
    program = (
        "import hashlib,importlib,json,resource,sys,time; "
        "before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
        "started=time.perf_counter(); candidate=getattr(importlib.import_module(sys.argv[1]),sys.argv[2])(); "
        "elapsed=time.perf_counter()-started; "
        "after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
        "actual=hashlib.sha256(candidate.artifact_path.read_bytes()).hexdigest(); "
        "assert actual==candidate.artifact_sha256; "
        'unit=1 if sys.platform=="darwin" else 1024; '
        'print(json.dumps({"seconds":elapsed,"incremental_peak_rss_bytes":max(0,after-before)*unit,"artifact_sha256":actual}))'
    )
    samples = []
    for _ in range(15):
        result = subprocess.run(
            [sys.executable, "-c", program, module, factory],
            capture_output=True,
            check=True,
            text=True,
            env=os.environ
            | {"PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        samples.append(json.loads(result.stdout))
    digests = {sample["artifact_sha256"] for sample in samples}
    if digests != {expected_artifact_sha256}:
        raise RejectedEvaluation(
            "isolated candidate factory artifact differs from the registered adapter"
        )
    rss = [sample["incremental_peak_rss_bytes"] for sample in samples]
    return {
        "load": _measurement([sample["seconds"] for sample in samples]),
        "incremental_peak_rss": {
            "samples_bytes": rss,
            "median_bytes": statistics.median(rss),
            "maximum_bytes": max(rss),
        },
    }


def dev_analog_gates(quality: dict[str, Any], utility: dict[str, Any]) -> list[str]:
    """Use the frozen floors as transparent development rejection signals."""
    failures = []
    for name, floor in (
        ("overall_accuracy", 0.95),
        ("macro_f1", 0.80),
        ("oov_accuracy", 0.82),
        ("ambiguous_accuracy", 0.88),
    ):
        value = quality[name]
        if value is None:
            failures.append(f"{name} is unavailable")
        elif value < floor:
            failures.append(f"{name} is below the {floor:.0%} dev-analog floor")
    for domain, value in quality["per_domain_accuracy"].items():
        if value is None or value < 0.90:
            failures.append(f"domain {domain} is below the 90% dev-analog floor")
        elif quality["overall_accuracy"] - value > 0.05:
            failures.append(
                f"domain {domain} is more than 5 points below overall accuracy"
            )
    for tag, record in quality["supported_tag_f1"].items():
        if record["count"] >= 50 and record["f1"] < 0.50:
            failures.append(f"UPOS {tag} F1 is below the 50% dev-analog floor")
    if utility["precision_generated"] - utility["precision_unfiltered"] < 0.10:
        failures.append("MWE precision improvement is below 10 points")
    if utility["unfiltered_count"] <= 0:
        failures.append("MWE unfiltered candidate count is zero")
    elif (
        1
        - utility["generated_overlap_with_unfiltered"]
        / utility["unfiltered_count"]
        < 0.10
    ):
        failures.append("MWE generated filter removes less than 10% of candidates")
    if utility["recall_generated"] < 0.70:
        failures.append("MWE generated recall is below 70%")
    if utility["recall_unfiltered"] - utility["recall_generated"] > 0.05:
        failures.append("MWE generated recall loss exceeds 5 points")
    if utility["recall_gold_filtered"] - utility["recall_generated"] > 0.05:
        failures.append("MWE tagger recall loss exceeds 5 points")
    if utility["precision_improvement_interval"]["lower"] <= 0:
        failures.append("MWE precision improvement lower bound is not positive")
    if utility["recall_loss_unfiltered_interval"]["upper"] > 0.05:
        failures.append("MWE unfiltered recall-loss upper bound exceeds 5 points")
    if utility["recall_loss_gold_interval"]["upper"] > 0.05:
        failures.append("MWE gold recall-loss upper bound exceeds 5 points")
    return failures


def evaluate(
    candidate: Candidate,
    specification: str,
    registration_path: Path,
    acquisition_root: Path,
) -> dict[str, Any]:
    manifest = load_manifest()
    # The public command never calls a final-capable loader.  Keep the split names
    # literal here so a convenience parameter cannot accidentally expose final.
    train = load_tagged_split(manifest, acquisition_root, "train")
    dev_sentences = load_tagged_split(manifest, acquisition_root, "dev")
    dev_mwe = load_gold_split(manifest, acquisition_root, "dev")
    data_hashes = {
        "train": _sha(
            acquisition_root
            / manifest["sources"][manifest["splits"]["train"]["source"]][
                "relative_root"
            ]
            / manifest["splits"]["train"]["path"]
        ),
        "dev": _sha(
            acquisition_root
            / manifest["sources"][manifest["splits"]["dev"]["source"]]["relative_root"]
            / manifest["splits"]["dev"]["path"]
        ),
    }
    registration = load_registration(
        registration_path, candidate, data_hashes=data_hashes
    )
    development_boundaries = boundaries(dev_sentences)
    checked = validate_predictions(
        development_boundaries, candidate.tag(development_boundaries)
    )
    warm = warm_inference_measurements(
        candidate, development_boundaries, checked
    )
    metrics = quality_metrics(train, dev_sentences, checked)
    configuration = manifest["filter_configuration"]
    if not isinstance(configuration, dict):
        raise RejectedEvaluation("manifest filter configuration is malformed")
    mwe_predictions = subset_predictions(dev_sentences, checked, dev_mwe.sentences)
    utility = utility_metrics(
        dev_mwe, mwe_predictions, candidate.model_id, configuration
    )
    gates = dev_analog_gates(metrics, utility)
    resources = isolated_load_measurements(
        specification, candidate.artifact_sha256
    )
    resources["warm_inference"] = warm
    resources["artifact_bytes"] = candidate.artifact_path.stat().st_size
    resources["inference_dev_tokens_per_second"] = (
        sum(len(sentence.tokens) for sentence in dev_sentences)
        / warm["median_seconds"]
    )
    resources["retry_rule"] = (
        "IQR/median above 10% invalidates the timing result; stabilize the "
        "environment and rerun once, then reject a second high-variance run."
    )
    if warm["iqr_over_median"] > 0.10:
        gates.append("warm inference IQR/median exceeds 10%; retry is required")
    if resources["load"]["iqr_over_median"] > 0.10:
        gates.append("cold load IQR/median exceeds 10%; retry is required")
    if resources["artifact_bytes"] > 20 * 1024 * 1024:
        gates.append("artifact exceeds the 20 MiB dev-analog ceiling")
    if resources["load"]["median_seconds"] > 0.250:
        gates.append("cold load exceeds the 250 ms dev-analog ceiling")
    if resources["incremental_peak_rss"]["maximum_bytes"] > 128 * 1024 * 1024:
        gates.append("incremental peak RSS exceeds the 128 MiB dev-analog ceiling")
    if resources["inference_dev_tokens_per_second"] < 50_000:
        gates.append("inference throughput is below the 50k token/s dev-analog floor")
    return {
        "schema_version": 1,
        "status": "accepted" if not gates else "rejected",
        "protected_final_evaluated": False,
        "candidate": {
            "model_id": candidate.model_id,
            "tokenizer_id": candidate.tokenizer_id,
            "artifact_sha256": candidate.artifact_sha256,
            "artifact_bytes": candidate.artifact_path.stat().st_size,
            "artifact_path": str(candidate.artifact_path),
        },
        "registration": asdict(registration),
        "data_hashes": data_hashes,
        "quality": metrics,
        "mwe_utility": utility,
        "resources": resources,
        "hard_dev_analog_gates": {"passed": not gates, "failures": gates},
        "diagnostics": {
            "protected_final_thresholds": "not evaluated; development analog only"
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "development_only": True,
    }


def reproduce_retained_c2(
    train: Path, dev: Path, output: Path, report: Path
) -> dict[str, Any]:
    """Run the retained c2 trainer unchanged and reject a non-reproducing result."""
    train_digest = _sha(train)
    dev_digest = _sha(dev)
    if train_digest != RETAINED_C2_TRAIN_SHA256:
        raise RejectedEvaluation(
            "retained c2 reproduction requires the pinned training split"
        )
    if dev_digest != RETAINED_C2_DEV_SHA256:
        raise RejectedEvaluation(
            "retained c2 reproduction requires the pinned development split"
        )
    trainer = Path(__file__).parents[1] / "pos-linear" / "train.py"
    command = [
        sys.executable,
        str(trainer),
        "--train",
        str(train),
        "--dev",
        str(dev),
        "--output",
        str(output),
        "--report",
        str(report),
        "--registration-template",
        str(report.with_suffix(".registration.json")),
        "--seed",
        "20260719",
        "--epochs",
        "8",
        "--buckets",
        str(1 << 18),
        "--candidate-id",
        "c2",
        "--source-revision",
        "0" * 40,
    ]
    subprocess.run(command, check=True)
    result = json.loads(report.read_text(encoding="utf-8"))
    actual = result["dev_accuracy_quantized_candidate_pruned"]
    if abs(actual - RETAINED_C2_ACCURACY) > RETAINED_C2_TOLERANCE:
        raise RejectedEvaluation(
            f"retained c2 accuracy {actual:.6%} is outside deterministic tolerance of {RETAINED_C2_ACCURACY:.4%}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--acquisition-root", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduce-c2", action="store_true")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--artifact", type=Path)
    arguments = parser.parse_args()
    evidence: dict[str, Any] = {
        "status": "rejected",
        "protected_final_evaluated": False,
    }
    try:
        if arguments.reproduce_c2:
            if not all((arguments.train, arguments.dev, arguments.artifact)):
                raise RejectedEvaluation(
                    "--reproduce-c2 requires --train, --dev, and --artifact"
                )
            evidence = reproduce_retained_c2(
                arguments.train,
                arguments.dev,
                arguments.artifact,
                arguments.output.with_suffix(".c2-report.json"),
            )
        else:
            if not all(
                (
                    arguments.candidate,
                    arguments.acquisition_root,
                    arguments.registration,
                )
            ):
                raise RejectedEvaluation(
                    "--candidate, --acquisition-root, and --registration are required"
                )
            evidence = evaluate(
                load_candidate(arguments.candidate),
                arguments.candidate,
                arguments.registration,
                arguments.acquisition_root,
            )
    except Exception as error:
        evidence["failure"] = f"{type(error).__name__}: {error}"
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(2) from error
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if evidence.get("status") == "rejected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a reproducible, development-only POS data sufficiency study.

The study preserves the retained c2 training recipe while varying only the
whole-document training sample.  It intentionally has no protected-split
argument or fallback.  Outputs are research evidence, not a candidate
registration or an augmentation decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter
from types import ModuleType
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pos.evaluation.loader import (  # noqa: E402
    Sentence,
    load_gold_split,
    load_manifest,
    load_tagged_split,
)


DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_MWE_FRACTIONS = (0.05, 0.50, 1.00)
DEFAULT_EPOCHS = 8
DEFAULT_BUCKETS = 1 << 18
DEFAULT_SEED = 20_260_719
DIAGNOSIS_EQUIVALENCE_ACCURACY = 0.0025


class StudyError(ValueError):
    """A requested study condition cannot preserve its frozen invariants."""


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise StudyError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _harness() -> ModuleType:
    return _load_module(
        "pos_sufficiency_bakeoff", ROOT / "experiments/pos-tagger/bakeoff.py"
    )


def _trainer() -> ModuleType:
    return _load_module(
        "pos_sufficiency_trainer", ROOT / "experiments/pos-linear/train.py"
    )


def _adapter() -> ModuleType:
    return _load_module(
        "pos_sufficiency_c2_adapter", ROOT / "experiments/pos-tagger/c2_adapter.py"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_implementation() -> None:
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )
    if changed:
        raise StudyError(
            "normal evidence requires a committed clean implementation; use --calibration for an exploratory run"
        )


def _parse_fractions(value: str) -> tuple[float, ...]:
    try:
        fractions = tuple(float(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "fractions must be comma-separated floats"
        ) from error
    if len(fractions) < 1 or any(
        fraction <= 0 or fraction > 1 for fraction in fractions
    ):
        raise argparse.ArgumentTypeError("fractions must be in (0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise argparse.ArgumentTypeError("fractions must be unique and ascending")
    return fractions


def _document_groups(sentences: Sequence[Sentence]) -> dict[str, list[Sentence]]:
    groups: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        groups.setdefault(sentence.document_id, []).append(sentence)
    return groups


def _domain_documents(
    sentences: Sequence[Sentence],
) -> dict[str, tuple[tuple[str, tuple[Sentence, ...]], ...]]:
    grouped = _document_groups(sentences)
    output: dict[str, list[tuple[str, tuple[Sentence, ...]]]] = {}
    for document_id, document_sentences in grouped.items():
        domains = {
            token.domain for sentence in document_sentences for token in sentence.tokens
        }
        if len(domains) != 1:
            raise StudyError(f"document {document_id!r} has multiple domains")
        output.setdefault(next(iter(domains)), []).append(
            (document_id, tuple(document_sentences))
        )
    return {domain: tuple(documents) for domain, documents in sorted(output.items())}


def _token_count(documents: Iterable[tuple[str, Sequence[Sentence]]]) -> int:
    return sum(
        len(token_list.tokens)
        for _name, document in documents
        for token_list in document
    )


def _select_documents(
    documents: Sequence[tuple[str, tuple[Sentence, ...]]],
    *,
    target_tokens: int,
    seed: int,
) -> tuple[tuple[str, tuple[Sentence, ...]], ...]:
    if target_tokens <= 0:
        raise StudyError("document sample target must be positive")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    selected: list[tuple[str, tuple[Sentence, ...]]] = []
    total = 0
    for document in shuffled:
        proposed = total + _token_count((document,))
        if proposed < target_tokens:
            selected.append(document)
            total = proposed
            continue
        if not selected or abs(proposed - target_tokens) < abs(total - target_tokens):
            selected.append(document)
            total = proposed
            break
    if not selected:
        raise StudyError("stratified sample selected no documents")
    return tuple(selected)


def _sample_stratified(
    by_domain: dict[str, tuple[tuple[str, tuple[Sentence, ...]], ...]],
    *,
    fraction: float | None,
    target_tokens: int | None,
    seed: int,
) -> tuple[tuple[Sentence, ...], dict[str, Any]]:
    if (fraction is None) == (target_tokens is None):
        raise StudyError("provide exactly one of fraction or target_tokens")
    available_total = _token_count(
        document for documents in by_domain.values() for document in documents
    )
    selected: list[tuple[str, tuple[Sentence, ...]]] = []
    summary: dict[str, Any] = {}
    for domain, documents in by_domain.items():
        available = _token_count(documents)
        target = (
            math.ceil(available * fraction)
            if fraction is not None
            else max(1, round(target_tokens * available / available_total))
        )
        chosen = _select_documents(
            documents, target_tokens=target, seed=_seed(seed, domain)
        )
        selected.extend(chosen)
        summary[domain] = {
            "available_tokens": available,
            "target_tokens": target,
            "actual_tokens": _token_count(chosen),
            "documents": len(chosen),
        }
    selected_ids = {document_id for document_id, _sentences in selected}
    ordered = tuple(
        sentence
        for documents in by_domain.values()
        for document_id, document_sentences in documents
        if document_id in selected_ids
        for sentence in document_sentences
    )
    return ordered, summary


def _seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_hash(sentences: Sequence[Sentence]) -> str:
    payload = "\n".join(
        f"{sentence.document_id}\t{sentence.sentence_id}\t"
        + "\u001f".join(token.form + "\u001e" + token.upos for token in sentence.tokens)
        for sentence in sentences
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _sample_metadata(sentences: Sequence[Sentence]) -> dict[str, Any]:
    groups = _document_groups(sentences)
    domains = Counter(
        token.domain for sentence in sentences for token in sentence.tokens
    )
    return {
        "sample_sha256": _sample_hash(sentences),
        "documents": len(groups),
        "sentences": len(sentences),
        "tokens": sum(len(sentence.tokens) for sentence in sentences),
        "domains": dict(sorted(domains.items())),
        "document_ids_sha256": hashlib.sha256(
            "\n".join(sorted(groups)).encode()
        ).hexdigest(),
    }


def _plan_sample(
    condition: dict[str, Any],
    pool: dict[str, tuple[tuple[str, tuple[Sentence, ...]], ...]],
    *,
    target_tokens: int | None = None,
) -> dict[str, Any]:
    sample, strata = _sample_stratified(
        pool,
        fraction=None if target_tokens is not None else condition["sample_fraction"],
        target_tokens=target_tokens,
        seed=condition["sample_seed"],
    )
    metadata = _sample_metadata(sample)
    return condition | {
        "sample_document_ids": sorted(_document_groups(sample)),
        "sample_strata": strata,
        "requested_token_budget": (
            target_tokens
            if target_tokens is not None
            else condition["requested_token_budget"]
        ),
        "planned_actual_tokens": metadata["tokens"],
    }


def _plan_conditions(
    jobs: Sequence[dict[str, Any]],
    train: Sequence[Sentence],
    *,
    match_tolerance_fraction: float,
) -> list[dict[str, Any]]:
    by_domain = _domain_documents(train)
    planned: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if job["kind"] != "mixed_control":
            pool = (
                by_domain
                if job["kind"] == "learning_curve"
                else {
                    domain: documents
                    for domain, documents in by_domain.items()
                    if domain != job["held_out_domain"]
                }
            )
            planned[job["id"]] = _plan_sample(job, pool)
    for job in jobs:
        if job["kind"] != "mixed_control":
            continue
        leave_out = planned[
            f"loo-{job['held_out_domain']}-r{job['id'].rsplit('-r', 1)[1]}"
        ]
        target = leave_out["planned_actual_tokens"]
        mixed = _plan_sample(job, by_domain, target_tokens=target)
        delta = mixed["planned_actual_tokens"] - target
        tolerance = math.ceil(target * match_tolerance_fraction)
        planned[job["id"]] = mixed | {
            "matching": {
                "paired_condition": leave_out["id"],
                "target_tokens": target,
                "actual_delta_tokens": delta,
                "tolerance_tokens": tolerance,
                "status": "matched" if abs(delta) <= tolerance else "unmatched",
                "method": "closest document-prefix sample within proportional domain strata",
            }
        }
        planned[leave_out["id"]] = leave_out | {
            "matching": {
                "paired_condition": mixed["id"],
                "target_tokens": target,
                "actual_delta_tokens": 0,
                "tolerance_tokens": tolerance,
                "status": "matched" if abs(delta) <= tolerance else "unmatched",
                "method": "leave-one-domain-out reference budget",
            }
        }
    return [planned[job["id"]] for job in jobs]


def _examples(trainer: ModuleType, sentences: Sequence[Sentence]) -> tuple[Any, ...]:
    return tuple(
        trainer.Example(
            tuple(
                trainer.Token(token.form, trainer.TAG_INDEX[token.upos])
                for token in sentence.tokens
            )
        )
        for sentence in sentences
    )


def _train_c2(
    trainer: ModuleType,
    adapter: ModuleType,
    train: Sequence[Sentence],
    dev: Sequence[Sentence],
    *,
    seed: int,
    epochs: int,
    buckets: int,
    artifact: Path,
    model_id: str,
    tokenizer_id: str,
) -> tuple[Any, dict[str, Any]]:
    train_examples = _examples(trainer, train)
    dev_examples = _examples(trainer, dev)
    direct_candidates, candidates, collisions = trainer.lexicons(train_examples)
    model = trainer.train(train_examples, epochs, buckets, seed, candidates)
    biases, weights = model.averaged()
    direct, threshold, direct_accuracy, direct_coverage = trainer.select_direct_lexicon(
        direct_candidates, dev_examples
    )
    full, _, _ = trainer.predict(biases, weights, dev_examples, buckets)
    quantized_biases, quantized_weights, scale = trainer.quantize(biases, weights)
    quantized, _, _ = trainer.predict(
        quantized_biases, quantized_weights, dev_examples, buckets
    )
    pruned, direct_total, direct_correct = trainer.predict(
        quantized_biases,
        quantized_weights,
        dev_examples,
        buckets,
        direct,
        candidates,
    )
    full_accuracy = trainer.accuracy(full, dev_examples)
    quantized_accuracy = trainer.accuracy(quantized, dev_examples)
    pruned_accuracy = trainer.accuracy(pruned, dev_examples)
    if (
        full_accuracy - quantized_accuracy > 0.0025
        or quantized_accuracy - pruned_accuracy > 0.0025
    ):
        raise StudyError(
            "c2 quantization or candidate pruning differs from retained semantics"
        )
    digest = trainer.compile_model(
        artifact,
        biases=quantized_biases,
        weights=quantized_weights,
        buckets=buckets,
        direct=direct,
        candidates=candidates,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
    )
    return adapter.C2Adapter(artifact), {
        "artifact_sha256": digest,
        "full_precision_accuracy": full_accuracy,
        "quantized_accuracy": quantized_accuracy,
        "candidate_pruned_accuracy": pruned_accuracy,
        "quantization_scale": scale,
        "direct_lexical_support_threshold": threshold,
        "direct_lexical_coverage": direct_coverage,
        "direct_lexical_accuracy": direct_accuracy,
        "direct_lexical_accepted_tokens": direct_total,
        "direct_lexical_correct_tokens": direct_correct,
        "candidate_entries": len(candidates),
        "direct_entries": len(direct),
        "removed_hash_collisions": collisions,
    }


def _accuracy(pairs: Sequence[tuple[str, str]]) -> float | None:
    return (
        None if not pairs else sum(left == right for left, right in pairs) / len(pairs)
    )


def _bin(value: int, bounds: tuple[tuple[int, int | None, str], ...]) -> str:
    for lower, upper, label in bounds:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise AssertionError("support bin is incomplete")


def _error_slices(
    train: Sequence[Sentence], dev: Sequence[Sentence], predictions: Any
) -> dict[str, Any]:
    train_forms = Counter(token.form for sentence in train for token in sentence.tokens)
    train_tags = Counter(token.upos for sentence in train for token in sentence.tokens)
    gold = [token for sentence in dev for token in sentence.tokens]
    predicted = [
        tag for document in predictions for sentence in document for tag in sentence
    ]
    if len(gold) != len(predicted):
        raise StudyError("prediction count changed while deriving error slices")
    frequency: dict[str, list[tuple[str, str]]] = {}
    support: dict[str, list[tuple[str, str]]] = {}
    for token, guessed in zip(gold, predicted, strict=True):
        pair = token.upos, guessed
        frequency.setdefault(
            _bin(
                train_forms[token.form],
                (
                    (0, 0, "0"),
                    (1, 1, "1"),
                    (2, 5, "2-5"),
                    (6, 20, "6-20"),
                    (21, None, "21+"),
                ),
            ),
            [],
        ).append(pair)
        support.setdefault(
            _bin(
                train_tags[token.upos],
                ((0, 0, "0"), (1, 9, "1-9"), (10, 99, "10-99"), (100, None, "100+")),
            ),
            [],
        ).append(pair)
    return {
        "token_frequency": {
            name: {"count": len(pairs), "accuracy": _accuracy(pairs)}
            for name, pairs in sorted(frequency.items())
        },
        "tag_support": {
            name: {"count": len(pairs), "accuracy": _accuracy(pairs)}
            for name, pairs in sorted(support.items())
        },
    }


def _condition_key(condition: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: condition[key]
            for key in ("kind", "fraction", "held_out_domain")
            if key in condition
        },
        sort_keys=True,
    )


def _condition_job(
    condition: dict[str, Any],
    *,
    acquisition_root: str,
    work_dir: str,
    epochs: int,
    buckets: int,
    model_id: str,
    tokenizer_id: str,
    evaluate_mwe: bool,
) -> dict[str, Any]:
    harness = _harness()
    trainer = _trainer()
    adapter = _adapter()
    manifest = load_manifest()
    root = Path(acquisition_root)
    train_all = load_tagged_split(manifest, root, "train")
    dev = load_tagged_split(manifest, root, "dev")
    selected_ids = set(condition["sample_document_ids"])
    sample = tuple(
        sentence for sentence in train_all if sentence.document_id in selected_ids
    )
    if set(_document_groups(sample)) != selected_ids:
        raise StudyError(
            "planned document sample no longer matches pinned training data"
        )
    metadata = _sample_metadata(sample)
    artifact = Path(work_dir) / "artifacts" / f"{condition['id']}.rmpos"
    started = perf_counter()
    candidate, training = _train_c2(
        trainer,
        adapter,
        sample,
        dev,
        seed=condition["model_seed"],
        epochs=epochs,
        buckets=buckets,
        artifact=artifact,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
    )
    training_seconds = perf_counter() - started
    dev_boundaries = harness.boundaries(dev)
    started = perf_counter()
    predictions = harness.validate_predictions(
        dev_boundaries, candidate.tag(dev_boundaries)
    )
    inference_seconds = perf_counter() - started
    quality = harness.quality_metrics(sample, dev, predictions)
    result: dict[str, Any] = {
        "id": condition["id"],
        "kind": condition["kind"],
        "fraction": condition.get("fraction"),
        "held_out_domain": condition.get("held_out_domain"),
        "sample_seed": condition["sample_seed"],
        "model_seed": condition["model_seed"],
        "sample": metadata
        | {
            "strata": condition["sample_strata"],
            "requested_token_budget": condition["requested_token_budget"],
            "actual_budget_delta_tokens": metadata["tokens"]
            - condition["requested_token_budget"],
        },
        "artifact": {
            "path": str(artifact),
            "sha256": candidate.artifact_sha256,
            "bytes": candidate.artifact_bytes,
        },
        "training": training,
        "quality": quality,
        "held_out_domain_accuracy": (
            quality["per_domain_accuracy"].get(condition["held_out_domain"])
            if condition.get("held_out_domain") is not None
            else None
        ),
        "matching": condition.get("matching"),
        "error_slices": _error_slices(sample, dev, predictions),
        "runtime_seconds": {
            "train": training_seconds,
            "infer_full_dev": inference_seconds,
        },
        "mwe_utility": None,
        "mwe_status": "not selected for representative utility evaluation",
    }
    if evaluate_mwe:
        dev_mwe = load_gold_split(manifest, root, "dev")
        mwe_predictions = harness.subset_predictions(
            dev, predictions, dev_mwe.sentences
        )
        configuration = manifest["filter_configuration"]
        if not isinstance(configuration, dict):
            raise StudyError("manifest filter configuration is malformed")
        started = perf_counter()
        result["mwe_utility"] = harness.utility_metrics(
            dev_mwe, mwe_predictions, candidate.model_id, configuration
        )
        result["runtime_seconds"]["mwe_utility"] = perf_counter() - started
        result["mwe_status"] = "evaluated on aligned development subset"
    return result


def _t_interval(
    values: Sequence[float | int | None],
) -> dict[str, float | int | str | None]:
    """Two-sided 95% Student-t interval across independent repetitions."""

    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "lower": None,
            "upper": None,
            "method": "two-sided Student-t 95% interval across repetitions",
        }
    mean = statistics.fmean(numeric)
    if len(numeric) == 1:
        return {
            "count": 1,
            "mean": mean,
            "lower": None,
            "upper": None,
            "method": "unavailable: one repetition",
        }
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
    }.get(len(numeric) - 1, 1.960)
    margin = critical * statistics.stdev(numeric) / math.sqrt(len(numeric))
    return {
        "count": len(numeric),
        "mean": mean,
        "lower": mean - margin,
        "upper": mean + margin,
        "method": "two-sided Student-t 95% interval across repetitions",
    }


def _numeric_intervals(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int | str | None]]:
    """Aggregate every numeric leaf in a report section by its dotted path."""

    def flatten(value: Any, prefix: str = "") -> dict[str, float | int]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {prefix: value}
        if not isinstance(value, dict):
            return {}
        return {
            path: number
            for key, nested in value.items()
            for path, number in flatten(
                nested, f"{prefix}.{key}" if prefix else key
            ).items()
        }

    flattened = [flatten(record) for record in records]
    keys = sorted({key for item in flattened for key in item})
    return {key: _t_interval([item.get(key) for item in flattened]) for key in keys}


def _aggregate(conditions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        groups.setdefault(_condition_key(condition), []).append(condition)
    output = []
    for key, group in sorted(groups.items()):
        first = group[0]
        output.append(
            {
                "condition": json.loads(key),
                "replicates": len(group),
                "quality": _numeric_intervals([item["quality"] for item in group]),
                "error_slices": _numeric_intervals(
                    [item["error_slices"] for item in group]
                ),
                "train_tokens": _t_interval(
                    [item["sample"]["tokens"] for item in group]
                ),
                "mwe_evaluated": any(item["mwe_utility"] is not None for item in group),
                "mwe_utility": _numeric_intervals(
                    [item["mwe_utility"] for item in group if item["mwe_utility"]]
                ),
                "example_condition": first["id"],
            }
        )
    return output


def _paired_interval(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, float | int | str | None]:
    left_by_seed = {item["model_seed"]: item for item in left}
    differences = [
        _condition_metric(left_by_seed[seed], metric) - _condition_metric(item, metric)
        for seed, item in ((item["model_seed"], item) for item in right)
        if seed in left_by_seed
        and _condition_metric(left_by_seed[seed], metric) is not None
        and _condition_metric(item, metric) is not None
    ]
    return _t_interval(differences)


def _condition_metric(condition: dict[str, Any], metric: str) -> float | None:
    if metric == "overall_accuracy":
        return condition["quality"][metric]
    return condition[metric]


def _demonstrably_flat(interval: dict[str, Any]) -> bool:
    return (
        interval["lower"] is not None
        and interval["upper"] is not None
        and interval["lower"] >= 0
        and interval["upper"] <= DIAGNOSIS_EQUIVALENCE_ACCURACY
    )


def _error_concentration(
    conditions: Sequence[dict[str, Any]], fraction: float
) -> dict[str, dict[str, float | int | str | None]]:
    full = [
        item
        for item in conditions
        if item["kind"] == "learning_curve" and item["fraction"] == fraction
    ]

    def excess(item: dict[str, Any], group: str, key: str) -> float | None:
        overall = item["quality"]["overall_accuracy"]
        value = item["error_slices"][group].get(key, {}).get("accuracy")
        return None if value is None else (1 - value) - (1 - overall)

    return {
        "oov_error_excess": _t_interval(
            [
                None
                if item["quality"]["oov_accuracy"] is None
                else (1 - item["quality"]["oov_accuracy"])
                - (1 - item["quality"]["overall_accuracy"])
                for item in full
            ]
        ),
        "frequency_0_error_excess": _t_interval(
            [excess(item, "token_frequency", "0") for item in full]
        ),
        "frequency_1_error_excess": _t_interval(
            [excess(item, "token_frequency", "1") for item in full]
        ),
        "tag_support_0_error_excess": _t_interval(
            [excess(item, "tag_support", "0") for item in full]
        ),
        "tag_support_1_9_error_excess": _t_interval(
            [excess(item, "tag_support", "1-9") for item in full]
        ),
    }


def _diagnosis(
    conditions: Sequence[dict[str, Any]], fractions: Sequence[float]
) -> dict[str, Any]:
    curves = [item for item in conditions if item["kind"] == "learning_curve"]
    by_fraction: dict[float, list[dict[str, Any]]] = {}
    for condition in curves:
        by_fraction.setdefault(condition["fraction"], []).append(condition)
    marginal = _t_interval([])
    if len(fractions) >= 2:
        marginal = _paired_interval(
            by_fraction[fractions[-1]],
            by_fraction[fractions[-2]],
            metric="overall_accuracy",
        )
    diversity: dict[str, dict[str, float | int | str | None]] = {}
    for domain in sorted(
        {
            item["held_out_domain"]
            for item in conditions
            if item["kind"] == "mixed_control"
        }
    ):
        mixed = [
            item
            for item in conditions
            if item["kind"] == "mixed_control"
            and item["held_out_domain"] == domain
            and item.get("matching", {}).get("status") == "matched"
        ]
        leave_out = [
            item
            for item in conditions
            if item["kind"] == "leave_one_domain_out"
            and item["held_out_domain"] == domain
            and item.get("matching", {}).get("status") == "matched"
        ]
        diversity[str(domain)] = _paired_interval(
            mixed, leave_out, metric="held_out_domain_accuracy"
        )
    concentration = _error_concentration(conditions, fractions[-1])
    concentrated = any(
        value["lower"] is not None and value["lower"] > 0
        for value in concentration.values()
    )
    positive_volume = marginal["lower"] is not None and marginal["lower"] > 0
    positive_diversity = any(
        value["lower"] is not None and value["lower"] > 0
        for value in diversity.values()
    )
    flat = _demonstrably_flat(marginal)
    diversity_flat = bool(diversity) and all(
        _demonstrably_flat(value) for value in diversity.values()
    )
    diagnosis = (
        "volume_limited"
        if positive_volume and diversity_flat and concentrated
        else "diversity_limited"
        if positive_diversity and flat and concentrated
        else "architecture_limited"
        if flat and diversity_flat and not concentrated
        else "inconclusive"
    )
    return {
        "diagnosis": diagnosis,
        "full_data_marginal_accuracy": marginal,
        "matched_mixed_minus_leave_out_accuracy": diversity,
        "error_concentration": concentration,
        "rule": "A data-limited diagnosis requires a positive repeated-sample t interval, a demonstrably flat non-selected axis, and positive OOV or low-support error excess. Flat means an interval entirely within [0, 0.25 percentage points]; negative effects are never called flat.",
        "augmentation_gate_criteria": "Open a separate provenance decision only after reproducible positive full-data or matched-diversity intervals, low-support/OOV-concentrated errors, and representative development MWE utility; this study does not authorize augmentation.",
    }


def _jobs(
    fractions: Sequence[float],
    *,
    replicates: int,
    seed: int,
    diversity_fraction: float,
    domain_tokens: dict[str, int],
    include_diversity: bool,
    mwe_fractions: Sequence[float],
) -> list[dict[str, Any]]:
    jobs = []
    all_tokens = sum(domain_tokens.values())
    for replicate in range(replicates):
        for fraction in fractions:
            jobs.append(
                {
                    "id": f"curve-f{fraction:.2f}-r{replicate}",
                    "kind": "learning_curve",
                    "fraction": fraction,
                    "sample_fraction": fraction,
                    "sample_seed": _seed(seed + replicate, f"curve:{fraction}"),
                    "model_seed": seed + replicate,
                    "requested_token_budget": math.ceil(all_tokens * fraction),
                    "mwe": fraction in mwe_fractions,
                }
            )
        if include_diversity:
            for domain in sorted(domain_tokens):
                leave_out_tokens = all_tokens - domain_tokens[domain]
                requested_budget = math.ceil(leave_out_tokens * diversity_fraction)
                common = {
                    "fraction": diversity_fraction,
                    "held_out_domain": domain,
                    "sample_seed": _seed(seed + replicate, f"diversity:{domain}"),
                    "model_seed": seed + replicate,
                    "mwe": False,
                    "requested_token_budget": requested_budget,
                }
                jobs.append(
                    common
                    | {
                        "id": f"loo-{domain}-r{replicate}",
                        "kind": "leave_one_domain_out",
                        "sample_fraction": diversity_fraction,
                    }
                )
                jobs.append(
                    common
                    | {
                        "id": f"mixed-{domain}-r{replicate}",
                        "kind": "mixed_control",
                        "sample_fraction": requested_budget / all_tokens,
                    }
                )
    return jobs


def _data_hashes(manifest: dict[str, object], root: Path) -> dict[str, str]:
    sources = manifest["sources"]
    splits = manifest["splits"]
    if not isinstance(sources, dict) or not isinstance(splits, dict):
        raise StudyError("manifest sources or splits are malformed")
    result = {}
    for split_name in ("train", "dev"):
        split = splits[split_name]
        if not isinstance(split, dict):
            raise StudyError(f"manifest split {split_name!r} is malformed")
        source = sources[split["source"]]
        if not isinstance(source, dict):
            raise StudyError(f"manifest source for {split_name!r} is malformed")
        result[split_name] = _sha(root / source["relative_root"] / split["path"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=_parse_fractions, default=DEFAULT_FRACTIONS)
    parser.add_argument(
        "--mwe-fractions", type=_parse_fractions, default=DEFAULT_MWE_FRACTIONS
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--diversity-fraction", type=float, default=1.0)
    parser.add_argument("--match-tolerance-fraction", type=float, default=0.01)
    parser.add_argument("--no-diversity", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.replicates < 1:
        parser.error("--replicates must be positive")
    if arguments.workers < 1 or arguments.epochs < 1:
        parser.error("--workers and --epochs must be positive")
    if arguments.buckets < 2 or arguments.buckets & (arguments.buckets - 1):
        parser.error("--buckets must be a power of two")
    if not 0 < arguments.diversity_fraction <= 1:
        parser.error("--diversity-fraction must be in (0, 1]")
    if not 0 <= arguments.match_tolerance_fraction <= 0.05:
        parser.error("--match-tolerance-fraction must be in [0, 0.05]")
    if any(fraction not in arguments.fractions for fraction in arguments.mwe_fractions):
        parser.error("--mwe-fractions must be a subset of --fractions")
    if not arguments.calibration:
        if len(arguments.fractions) < 5 or arguments.fractions[-1] != 1.0:
            parser.error(
                "normal runs require at least five ascending fractions through 1.0"
            )
        if arguments.replicates < 5:
            parser.error("normal runs require at least five repetitions")
        if arguments.no_diversity:
            parser.error("normal runs require matched domain-diversity controls")

    evidence: dict[str, Any] = {
        "status": "rejected",
        "protected_final_evaluated": False,
    }
    try:
        if not arguments.calibration:
            _require_clean_implementation()
        manifest = load_manifest()
        train = load_tagged_split(manifest, arguments.acquisition_root, "train")
        domain_documents = _domain_documents(train)
        domain_tokens = {
            domain: _token_count(documents)
            for domain, documents in domain_documents.items()
        }
        jobs = _jobs(
            arguments.fractions,
            replicates=arguments.replicates,
            seed=arguments.seed,
            diversity_fraction=arguments.diversity_fraction,
            domain_tokens=domain_tokens,
            include_diversity=not arguments.no_diversity,
            mwe_fractions=arguments.mwe_fractions,
        )
        conditions = _plan_conditions(
            jobs,
            train,
            match_tolerance_fraction=arguments.match_tolerance_fraction,
        )
        provenance = {
            "code_revision": _code_revision(),
            "manifest_sha256": _sha(ROOT / "tests/pos/evaluation/manifest.json"),
            "data_hashes": _data_hashes(manifest, arguments.acquisition_root),
            "acquisition_root": str(arguments.acquisition_root),
            "candidate_recipe": {
                "name": "retained-c2-semantics",
                "epochs": arguments.epochs,
                "buckets": arguments.buckets,
                "model_id": "remerge-pos-linear-v1",
                "tokenizer_id": "unicode-whitespace-v1",
            },
            "command": " ".join(sys.argv),
            "argv": list(sys.argv),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "cwd": str(Path.cwd()),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
        }
        if arguments.dry_run:
            evidence = {
                "schema_version": 1,
                "status": "calibration_planned" if arguments.calibration else "planned",
                "protected_final_evaluated": False,
                "provenance": provenance,
                "conditions": conditions,
                "development_only": True,
            }
        else:
            arguments.work_dir.mkdir(parents=True, exist_ok=True)
            common = {
                "acquisition_root": str(arguments.acquisition_root),
                "work_dir": str(arguments.work_dir),
                "epochs": arguments.epochs,
                "buckets": arguments.buckets,
                "model_id": "remerge-pos-linear-v1",
                "tokenizer_id": "unicode-whitespace-v1",
            }
            if arguments.workers == 1:
                conditions = [
                    _condition_job(job, evaluate_mwe=job["mwe"], **common)
                    for job in conditions
                ]
            else:
                with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
                    futures = [
                        executor.submit(
                            _condition_job, job, evaluate_mwe=job["mwe"], **common
                        )
                        for job in conditions
                    ]
                    conditions = [future.result() for future in futures]
            conditions.sort(key=lambda item: item["id"])
            evidence = {
                "schema_version": 1,
                "status": "calibration" if arguments.calibration else "completed",
                "protected_final_evaluated": False,
                "development_only": True,
                "provenance": provenance,
                "design": {
                    "fractions": arguments.fractions,
                    "replicates": arguments.replicates,
                    "seed": arguments.seed,
                    "workers": arguments.workers,
                    "calibration": arguments.calibration,
                    "sampling": "whole-document, domain-stratified closest-prefix selection; diversity pairs record actual budget deltas and are excluded from diagnosis when outside tolerance",
                    "match_tolerance_fraction": arguments.match_tolerance_fraction,
                    "uncertainty": "two-sided Student-t 95% intervals across repeated deterministic sample/model seeds; MWE utility retains the common harness's 10,000 document-bootstrap resamples",
                },
                "conditions": conditions,
                "aggregates": _aggregate(conditions),
                "diagnosis": _diagnosis(conditions, arguments.fractions),
                "reference_error_overlap": {
                    "status": "unavailable",
                    "reason": "no approved offline reference prediction was supplied",
                },
            }
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


if __name__ == "__main__":
    main()

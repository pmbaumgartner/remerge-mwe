#!/usr/bin/env python3
"""Exercise an installed ``remerge-pos`` distribution in an isolated process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any

from remerge_pos import Tagger, UPOS_TAGS
from remerge_pos.training import (
    SELECTED_CONFIG,
    SELECTED_SEED,
    train,
    write_artifact,
)


def _measurement(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    lower = ordered[: len(ordered) // 2]
    upper = ordered[(len(ordered) + 1) // 2 :]
    iqr = statistics.median(upper) - statistics.median(lower)
    return {
        "seconds": values,
        "median_seconds": median,
        "iqr_over_median": iqr / median if median else 0.0,
    }


def _validate_predictions(
    expected: tuple[tuple[str, ...], ...], actual: object
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise ValueError("tagger changed the sentence count")
    checked = []
    for index, (forms, tags) in enumerate(zip(expected, actual, strict=True)):
        if not isinstance(tags, (list, tuple)) or len(tags) != len(forms):
            raise ValueError(f"tagger changed token count in sentence {index}")
        if any(tag not in UPOS_TAGS for tag in tags):
            raise ValueError(f"tagger emitted invalid UPOS in sentence {index}")
        checked.append(tuple(tags))
    return tuple(checked)


def _api_controls(tagger: Tagger) -> dict[str, str]:
    controls: dict[str, tuple[object, str]] = {
        "outer-string": ("not nested", "en"),
        "inner-string": (("not a sentence sequence",), "en"),
        "empty-sentence": (((),), "en"),
        "empty-form": (("",), "en"),
        "whitespace-form": (("two words",), "en"),
        "non-nfc-form": (("e\u0301",), "en"),
        "unsupported-language": (("word",), "fr"),
    }
    results = {}
    for name, (sentences, language) in controls.items():
        try:
            tagger.tag(sentences, language=language)  # type: ignore[arg-type]
        except ValueError as error:
            results[name] = f"rejected: {error}"
        else:
            raise AssertionError(f"API control {name!r} unexpectedly passed")
    valid = (("The", "watch"), ("I", "watch"))
    _validate_predictions(valid, tagger.tag(valid))
    results["valid-alignment"] = "passed"
    return results


def _train_selected(input_path: Path, artifact_path: Path) -> dict[str, Any]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    rows = tuple(
        tuple((str(form), str(tag)) for form, tag in sentence) for sentence in raw
    )
    started = perf_counter()
    digest = write_artifact(train(rows, SELECTED_CONFIG, SELECTED_SEED), artifact_path)
    return {
        "artifact_sha256": digest,
        "training_seconds": perf_counter() - started,
        "config": {
            "epochs": SELECTED_CONFIG.epochs,
            "feature_cutoff": SELECTED_CONFIG.feature_cutoff,
            "feature_buckets": SELECTED_CONFIG.feature_buckets,
        },
        "seed": SELECTED_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--api-controls", action="store_true")
    arguments = parser.parse_args()

    if arguments.train_input is not None:
        if arguments.train_output is None:
            parser.error("--train-input requires --train-output")
        if arguments.artifact is not None or arguments.input is not None:
            parser.error("training mode cannot also run inference")
        evidence = _train_selected(arguments.train_input, arguments.train_output)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    if arguments.artifact is None:
        parser.error("--artifact is required for inference and API controls")

    started = perf_counter()
    tagger = Tagger.load(arguments.artifact)
    load_seconds = perf_counter() - started
    evidence: dict[str, Any] = {
        "artifact_sha256": tagger.artifact_sha256,
        "model_id": tagger.model_id,
        "load_seconds": load_seconds,
    }
    if arguments.api_controls:
        evidence["controls"] = _api_controls(tagger)
    else:
        if arguments.input is None:
            parser.error("--input is required unless --api-controls is used")
        raw = json.loads(arguments.input.read_text(encoding="utf-8"))
        sentences = tuple(tuple(sentence) for sentence in raw)
        predictions = _validate_predictions(sentences, tagger.tag(sentences))
        for _ in range(arguments.warmups):
            if _validate_predictions(sentences, tagger.tag(sentences)) != predictions:
                raise RuntimeError("predictions changed during warmup")
        timings = []
        for _ in range(arguments.repetitions):
            started = perf_counter()
            actual = _validate_predictions(sentences, tagger.tag(sentences))
            timings.append(perf_counter() - started)
            if actual != predictions:
                raise RuntimeError("predictions changed during measurement")
        evidence.update(
            {
                "predictions": predictions,
                "timing": _measurement(timings),
                "token_count": sum(map(len, sentences)),
            }
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

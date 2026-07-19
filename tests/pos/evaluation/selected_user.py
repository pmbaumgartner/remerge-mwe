"""Evaluate the supplied-tag workflow on the approved public development corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import remerge
from tests.pos.evaluation.loader import (
    GoldSplit,
    MweSpan,
    load_gold_split,
    load_manifest,
)

Occurrence = tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateMetrics:
    count: int
    true_positives: int
    precision: float
    recall: float


def _occurrences(winners: list[remerge.WinnerWithOccurrences]) -> set[Occurrence]:
    return {
        (
            occurrence.document_index,
            occurrence.sentence_index,
            occurrence.start_token,
            occurrence.end_token,
        )
        for winner in winners
        for occurrence in winner.occurrences
    }


def _metrics(candidates: set[Occurrence], gold: set[Occurrence]) -> CandidateMetrics:
    true_positives = len(candidates & gold)
    return CandidateMetrics(
        count=len(candidates),
        true_positives=true_positives,
        precision=true_positives / len(candidates),
        recall=true_positives / len(gold),
    )


def _representative_corpus(
    gold_split: GoldSplit,
) -> tuple[list[str], list[remerge.TaggedDocument], set[Occurrence]]:
    documents: dict[str, list[tuple[str, tuple[remerge.TaggedToken, ...]]]] = {}
    document_order: list[str] = []
    document_indices: dict[str, int] = {}
    sentence_coordinates: dict[tuple[str, str], tuple[int, int]] = {}
    for sentence in gold_split.sentences:
        if sentence.document_id not in documents:
            documents[sentence.document_id] = []
            document_indices[sentence.document_id] = len(document_order)
            document_order.append(sentence.document_id)
        sentence_index = len(documents[sentence.document_id])
        sentence_coordinates[(sentence.document_id, sentence.sentence_id)] = (
            document_indices[sentence.document_id],
            sentence_index,
        )
        documents[sentence.document_id].append(
            (
                " ".join(token.form for token in sentence.tokens),
                tuple(
                    remerge.TaggedToken(token.form, token.upos)
                    for token in sentence.tokens
                ),
            )
        )
    raw_documents: list[str] = []
    tagged_documents: list[remerge.TaggedDocument] = []
    for document_id in document_order:
        sentences = documents[document_id]
        raw_documents.append("\n".join(raw for raw, _tagged in sentences))
        tagged_documents.append(
            remerge.TaggedDocument(tuple(tagged for _raw, tagged in sentences))
        )
    gold = {_occurrence(span, sentence_coordinates) for span in gold_split.mwe_spans}
    return raw_documents, tagged_documents, gold


def _occurrence(
    span: MweSpan,
    sentence_coordinates: dict[tuple[str, str], tuple[int, int]],
) -> Occurrence:
    document_index, sentence_index = sentence_coordinates[
        (span.document_id, span.sentence_id)
    ]
    return document_index, sentence_index, span.start_token, span.end_token


def evaluate(
    acquisition_root: Path,
    *,
    artifact_sha256: str,
    source_revision: str,
    selected_user: str,
) -> dict[str, Any]:
    manifest = load_manifest()
    gold_split = load_gold_split(manifest, acquisition_root, "dev")
    raw_documents, tagged_documents, gold = _representative_corpus(gold_split)
    frozen = cast(dict[str, Any], manifest["filter_configuration"])
    sources = cast(dict[str, dict[str, Any]], manifest["sources"])
    splits = cast(dict[str, dict[str, Any]], manifest["splits"])
    mwe = cast(dict[str, Any], manifest["mwe_gold"])
    mwe_dev = cast(dict[str, Any], mwe["splits"])["dev"]
    configuration: dict[str, Any] = {
        "iterations": frozen["requested_winners"],
        "patterns": frozen["patterns"],
        "method": frozen["score_method"],
        "min_count": frozen["min_count"],
        "min_score": frozen["min_score"],
        "on_exhausted": frozen["on_exhausted"],
    }
    discovery_arguments = {
        key: value for key, value in configuration.items() if key != "patterns"
    }

    started = perf_counter()
    unfiltered_winners = remerge.run_with_occurrences(
        raw_documents, **discovery_arguments
    )
    unfiltered_seconds = perf_counter() - started
    started = perf_counter()
    filtered_winners = remerge.run_tagged(
        tagged_documents,
        patterns=[tuple(pattern) for pattern in configuration["patterns"]],
        **discovery_arguments,
    )
    filtered_seconds = perf_counter() - started

    unfiltered = _occurrences(unfiltered_winners)
    filtered = _occurrences(filtered_winners)
    unfiltered_metrics = _metrics(unfiltered, gold)
    filtered_metrics = _metrics(filtered, gold)
    return {
        "schema_version": 1,
        "status": "completed",
        "selected_user": selected_user,
        "installed_version": version("remerge-mwe"),
        "artifact_sha256": artifact_sha256,
        "source_revision": source_revision,
        "protected_final_evaluated": False,
        "corpus": {
            "dataset": manifest["dataset_id"],
            "license": sources["ud_ewt"]["license"],
            "ud_ewt_revision": sources["ud_ewt"]["revision"],
            "ud_ewt_dev_sha256": splits["dev"]["sha256"],
            "streusle_revision": sources["streusle"]["revision"],
            "streusle_dev_sha256": mwe_dev["sha256"],
            "documents": len(
                {sentence.document_id for sentence in gold_split.sentences}
            ),
            "sentences": len(gold_split.sentences),
            "tokens": sum(len(sentence.tokens) for sentence in gold_split.sentences),
            "in_scope_gold_spans": len(gold_split.mwe_spans),
        },
        "configuration": configuration,
        "results": {
            "unfiltered": asdict(unfiltered_metrics),
            "supplied_tag": asdict(filtered_metrics),
            "candidate_count_reduction": (
                1 - filtered_metrics.count / unfiltered_metrics.count
            ),
            "precision_improvement_points": (
                filtered_metrics.precision - unfiltered_metrics.precision
            ),
            "filtered_overlap_with_unfiltered": len(filtered & unfiltered),
            "filtered_new_candidates": len(filtered - unfiltered),
            "filtered_excluded_unfiltered_gold": len((unfiltered & gold) - filtered),
            "filtered_new_gold": len((filtered & gold) - unfiltered),
            "filtered_missed_gold": len(gold - filtered),
            "unfiltered_winners": len(unfiltered_winners),
            "supplied_tag_winners": len(filtered_winners),
        },
        "runtime_seconds": {
            "unfiltered": unfiltered_seconds,
            "supplied_tag": filtered_seconds,
        },
    }


def _full_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(f"{name} must be a full lowercase SHA-256")
    return value


def _source_revision(value: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "source revision must be a full lowercase Git SHA"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition_root", type=Path)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--source-revision", required=True, type=_source_revision)
    parser.add_argument("--selected-user", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    artifact_sha256 = _full_sha256(arguments.artifact_sha256, "artifact SHA-256")
    evidence = evaluate(
        arguments.acquisition_root,
        artifact_sha256=artifact_sha256,
        source_revision=arguments.source_revision,
        selected_user=arguments.selected_user,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

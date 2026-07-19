#!/usr/bin/env python3
"""Evaluate the supplied-tag workflow on the approved public development corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import remerge  # noqa: E402
from tests.pos.evaluation.loader import (  # noqa: E402
    Sentence,
    _read_streusle,
    load_manifest,
    load_tagged_split,
)


STREUSLE_DEV_PATH = Path("streusle/dev/streusle.ud_dev.conllulex")
STREUSLE_DEV_SHA256 = (
    "37d940dbeb0f4d63ed8d0929f91980e491f79a907b7d28c48ec6e242f2db94d6"
)
EXPECTED_ALIGNED_SENTENCES = 546
EXPECTED_ALIGNED_TOKENS = 5_366
EXPECTED_DOCUMENTS = 192
EXPECTED_GOLD_SPANS = 23
PATTERNS: list[remerge.PosPattern] = [
    ("ADJ", "NOUN"),
    ("NOUN", "NOUN"),
    ("VERB", "NOUN"),
    ("VERB", "PART"),
]
Occurrence = tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateMetrics:
    count: int
    true_positives: int
    precision: float
    recall: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _occurrences(winners: list[remerge.TaggedWinnerInfo]) -> set[Occurrence]:
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


def _load_representative_corpus(
    acquisition_root: Path,
) -> tuple[list[str], list[remerge.TaggedDocument], set[Occurrence]]:
    manifest = load_manifest()
    development_sentences = load_tagged_split(manifest, acquisition_root, "dev")
    streusle_path = acquisition_root / STREUSLE_DEV_PATH
    observed_sha256 = _sha256(streusle_path)
    if observed_sha256 != STREUSLE_DEV_SHA256:
        raise ValueError(
            "STREUSLE development checksum mismatch: "
            f"{observed_sha256} != {STREUSLE_DEV_SHA256}"
        )

    ewt_by_sentence = {
        sentence.sentence_id: sentence for sentence in development_sentences
    }
    aligned: list[tuple[Sentence, dict[str, set[int]]]] = []
    for sentence_id, tokens, strong_mwes in _read_streusle(streusle_path):
        sentence = ewt_by_sentence.get(sentence_id)
        if sentence is None:
            continue
        observed_tokens = tuple(
            (token.form, token.upos) for token in sentence.tokens
        )
        if observed_tokens != tokens:
            raise ValueError(
                f"STREUSLE sentence {sentence_id!r} does not align with UD EWT dev"
            )
        aligned.append((sentence, dict(strong_mwes)))

    documents: dict[str, list[Sentence]] = defaultdict(list)
    gold_by_document: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    pattern_set = set(PATTERNS)
    for sentence, strong_mwes in aligned:
        sentence_index = len(documents[sentence.document_id])
        documents[sentence.document_id].append(sentence)
        for indices in strong_mwes.values():
            ordered = tuple(sorted(indices))
            if len(ordered) != 2 or ordered != (ordered[0], ordered[0] + 1):
                continue
            tags = tuple(sentence.tokens[index].upos for index in ordered)
            if tags in pattern_set:
                gold_by_document[sentence.document_id].add(
                    (sentence_index, ordered[0], ordered[-1] + 1)
                )

    raw_documents: list[str] = []
    tagged_documents: list[remerge.TaggedDocument] = []
    gold: set[Occurrence] = set()
    for document_index, (document_id, sentences) in enumerate(documents.items()):
        raw_documents.append(
            "\n".join(
                " ".join(token.form for token in sentence.tokens)
                for sentence in sentences
            )
        )
        tagged_documents.append(
            remerge.TaggedDocument(
                tuple(
                    tuple(
                        remerge.TaggedToken(token.form, token.upos)
                        for token in sentence.tokens
                    )
                    for sentence in sentences
                )
            )
        )
        gold.update(
            (document_index, sentence_index, start, end)
            for sentence_index, start, end in gold_by_document[document_id]
        )

    observed = (
        len(aligned),
        sum(len(sentence.tokens) for sentence, _mwes in aligned),
        len(documents),
        len(gold),
    )
    expected = (
        EXPECTED_ALIGNED_SENTENCES,
        EXPECTED_ALIGNED_TOKENS,
        EXPECTED_DOCUMENTS,
        EXPECTED_GOLD_SPANS,
    )
    if observed != expected:
        raise ValueError(f"representative corpus shape {observed} != frozen {expected}")
    return raw_documents, tagged_documents, gold


def evaluate(
    acquisition_root: Path,
    *,
    artifact_sha256: str,
    source_revision: str,
    selected_user: str,
) -> dict[str, Any]:
    raw_documents, tagged_documents, gold = _load_representative_corpus(
        acquisition_root
    )
    configuration: dict[str, Any] = {
        "iterations": 500,
        "patterns": [list(pattern) for pattern in PATTERNS],
        "method": "log_likelihood",
        "min_count": 2,
        "min_score": 0.0,
        "on_exhausted": "stop",
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
        tagged_documents, patterns=PATTERNS, **discovery_arguments
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
            "dataset": "UD English EWT r2.10 reviews dev + STREUSLE v4.5 dev",
            "license": "CC BY-SA 4.0",
            "ud_ewt_revision": (
                "b33472d3ce50a62056d6057f1b8a723e9d211176"
            ),
            "ud_ewt_dev_sha256": (
                "ef962ac05d844eaff46eeded125937129bfc0876d43963d66810cb73ffa8f5df"
            ),
            "streusle_revision": (
                "6c7855e717239e79321074765cc0f96dbcf72d1a"
            ),
            "streusle_dev_sha256": STREUSLE_DEV_SHA256,
            "documents": EXPECTED_DOCUMENTS,
            "sentences": EXPECTED_ALIGNED_SENTENCES,
            "tokens": EXPECTED_ALIGNED_TOKENS,
            "in_scope_gold_spans": EXPECTED_GOLD_SPANS,
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
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(f"{name} must be a full lowercase SHA-256")
    return value


def _source_revision(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
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

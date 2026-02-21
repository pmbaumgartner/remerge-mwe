from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PRFCounts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        denominator = precision + recall
        if denominator == 0.0:
            return 0.0
        return 2.0 * precision * recall / denominator

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def prf_from_sets(gold: set[T], predicted: set[T]) -> PRFCounts:
    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return PRFCounts(tp=tp, fp=fp, fn=fn)


def prf_from_counts(tp: int, fp: int, fn: int) -> PRFCounts:
    return PRFCounts(tp=tp, fp=fp, fn=fn)

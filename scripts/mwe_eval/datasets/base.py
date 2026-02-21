from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldMention:
    """A gold MWE mention represented as token indices in one sentence."""

    token_indices: tuple[int, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(self.token_indices))
        if not normalized:
            raise ValueError("token_indices must not be empty.")
        if any(index < 0 for index in normalized):
            raise ValueError("token_indices must be non-negative.")
        sorted_indices = tuple(sorted(normalized))
        if sorted_indices != normalized:
            normalized = sorted_indices
        object.__setattr__(self, "token_indices", normalized)

    @property
    def token_count(self) -> int:
        return len(self.token_indices)

    @property
    def is_contiguous(self) -> bool:
        if self.token_count < 2:
            return False
        first = self.token_indices[0]
        return self.token_indices == tuple(range(first, first + self.token_count))

    @property
    def contiguous_span(self) -> tuple[int, int] | None:
        if not self.is_contiguous:
            return None
        start = self.token_indices[0]
        return start, start + self.token_count


@dataclass(frozen=True, slots=True)
class Sentence:
    """Tokenized sentence plus associated gold MWE mentions."""

    sent_id: str
    tokens: tuple[str, ...]
    mentions: tuple[GoldMention, ...]

    def __post_init__(self) -> None:
        if not self.sent_id:
            raise ValueError("sent_id must be non-empty.")

    def non_singleton_mentions(self) -> tuple[GoldMention, ...]:
        return tuple(mention for mention in self.mentions if mention.token_count >= 2)

    def contiguous_mentions(self) -> tuple[GoldMention, ...]:
        return tuple(
            mention
            for mention in self.non_singleton_mentions()
            if mention.is_contiguous
        )

    def contiguous_spans(self) -> set[tuple[int, int]]:
        spans: set[tuple[int, int]] = set()
        for mention in self.contiguous_mentions():
            span = mention.contiguous_span
            if span is not None:
                spans.add(span)
        return spans

    def contiguous_types(self) -> set[tuple[str, ...]]:
        types: set[tuple[str, ...]] = set()
        for start, end in self.contiguous_spans():
            types.add(self.tokens[start:end])
        return types


@dataclass(frozen=True, slots=True)
class SplitData:
    """A normalized dataset split."""

    dataset: str
    split: str
    sentences: tuple[Sentence, ...]


def count_gold_mentions(split_data: SplitData) -> tuple[int, int, int]:
    """Return (gold_total, gold_contiguous, gold_discontinuous)."""

    gold_total = 0
    gold_contiguous = 0
    for sentence in split_data.sentences:
        non_singleton = sentence.non_singleton_mentions()
        contiguous = sentence.contiguous_mentions()
        gold_total += len(non_singleton)
        gold_contiguous += len(contiguous)
    return gold_total, gold_contiguous, gold_total - gold_contiguous

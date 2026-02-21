from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol


class _MergedLexemeLike(Protocol):
    @property
    def word(self) -> Any: ...


class _WinnerLike(Protocol):
    @property
    def merged_lexeme(self) -> _MergedLexemeLike: ...


DecodePolicy = Literal["all", "longest_per_start", "maximal_non_overlapping"]


@dataclass(slots=True)
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    end_lengths: list[int] = field(default_factory=list)


def build_trie(mwes: Iterable[tuple[str, ...]]) -> TrieNode:
    root = TrieNode()
    for mwe in mwes:
        if len(mwe) < 2:
            continue
        node = root
        for token in mwe:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.end_lengths.append(len(mwe))
    return root


def find_spans(
    tokens: tuple[str, ...] | list[str], trie: TrieNode
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    n_tokens = len(tokens)
    for start in range(n_tokens):
        node = trie
        end = start
        while end < n_tokens:
            token = tokens[end]
            next_node = node.children.get(token)
            if next_node is None:
                break
            node = next_node
            for length in node.end_lengths:
                out.add((start, start + length))
            end += 1
    return out


def _decode_longest_per_start(spans: set[tuple[int, int]]) -> set[tuple[int, int]]:
    best_by_start: dict[int, tuple[int, int]] = {}
    for start, end in spans:
        current = best_by_start.get(start)
        length = end - start
        if current is None:
            best_by_start[start] = (start, end)
            continue
        current_length = current[1] - current[0]
        if length > current_length or (length == current_length and end > current[1]):
            best_by_start[start] = (start, end)
    return set(best_by_start.values())


def _decode_maximal_non_overlapping(
    spans: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    candidates = sorted(
        _decode_longest_per_start(spans), key=lambda span: (span[0], -span[1])
    )
    selected: list[tuple[int, int]] = []

    current_end = -1
    for span in candidates:
        start, end = span
        if start < current_end:
            continue
        selected.append(span)
        current_end = end

    return set(selected)


def decode_spans(
    spans: set[tuple[int, int]],
    *,
    policy: DecodePolicy = "all",
) -> set[tuple[int, int]]:
    if policy == "all":
        return set(spans)
    if policy == "longest_per_start":
        return _decode_longest_per_start(spans)
    if policy == "maximal_non_overlapping":
        return _decode_maximal_non_overlapping(spans)
    raise ValueError(
        f"Unsupported decode policy {policy!r}. "
        "Expected one of: 'all', 'longest_per_start', 'maximal_non_overlapping'."
    )


def spans_to_types(
    tokens: tuple[str, ...] | list[str],
    spans: Iterable[tuple[int, int]],
) -> set[tuple[str, ...]]:
    return {tuple(tokens[start:end]) for start, end in spans}


def _tuple_is_subspan(candidate: tuple[str, ...], container: tuple[str, ...]) -> bool:
    if len(candidate) >= len(container):
        return False
    width = len(candidate)
    for start in range(0, len(container) - width + 1):
        if container[start : start + width] == candidate:
            return True
    return False


def suppress_subspan_types(types: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    if not types:
        return set()

    filtered: set[tuple[str, ...]] = set()
    for candidate in types:
        if any(
            _tuple_is_subspan(candidate, other) for other in types if other != candidate
        ):
            continue
        filtered.add(candidate)
    return filtered


def lexicon_from_winners(
    winners: Iterable[_WinnerLike],
    *,
    min_tokens: int = 2,
) -> set[tuple[str, ...]]:
    lexicon: set[tuple[str, ...]] = set()
    for winner in winners:
        word = winner.merged_lexeme.word
        if not isinstance(word, (list, tuple)):
            continue
        tokens = tuple(str(token) for token in word)
        if len(tokens) >= min_tokens:
            lexicon.add(tokens)
    return lexicon

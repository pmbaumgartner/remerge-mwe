import json
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import NewType, TypeVar

from ._core import (
    Engine,
    STATUS_BELOW_MIN_SCORE,
    STATUS_COMPLETED,
    STATUS_NO_CANDIDATE,
)


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"


class TieBreaker(str, Enum):
    deterministic = "deterministic"
    legacy_first_seen = "legacy_first_seen"


class Splitter(str, Enum):
    delimiter = "delimiter"
    sentencex = "sentencex"


class ExhaustionPolicy(str, Enum):
    stop = "stop"
    raise_ = "raise"


class NoCandidateBigramError(ValueError):
    """Raised when no candidate bigrams are available for selection."""


@dataclass(frozen=True, slots=True)
class Lexeme:
    word: tuple[str, ...]
    ix: int

    def __repr__(self) -> str:
        return f"({self.word}|{self.ix})"


LineIndex = NewType("LineIndex", int)
TokenIndex = NewType("TokenIndex", int)
Bigram = tuple[Lexeme, Lexeme]


@dataclass(frozen=True)
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    bigram_locations: list[tuple[LineIndex, TokenIndex]]

    @cached_property
    def cleaned_bigram_locations(self) -> tuple[tuple[LineIndex, TokenIndex], ...]:
        """Greedily select non-overlapping bigram starts per line."""
        clean_locations: list[tuple[LineIndex, TokenIndex]] = []
        for line, location_group in groupby(self.bigram_locations, key=lambda x: x[0]):
            exclude_tokens: set[TokenIndex] = set()
            token_ix = [i[1] for i in location_group]
            for token in token_ix:
                if token in exclude_tokens:
                    continue
                excludes = [i for i in token_ix if token <= i < token + self.n_lexemes]
                exclude_tokens.update(excludes)
                clean_locations.append((line, token))
        return tuple(clean_locations)

    def clean_bigram_locations(self) -> list[tuple[LineIndex, TokenIndex]]:
        return list(self.cleaned_bigram_locations)

    @property
    def n_lexemes(self) -> int:
        return len(self.merged_lexeme.word)

    @property
    def merge_token_count(self) -> int:
        return len(self.cleaned_bigram_locations)


StepPayload = tuple[
    float,
    list[str],
    int,
    list[str],
    int,
    list[str],
    int,
    list[tuple[int, int]],
]


EnumType = TypeVar("EnumType", bound=Enum)


def _coerce_enum(
    value: EnumType | str, enum_type: type[EnumType], argument_name: str
) -> EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        options = ", ".join(repr(option.value) for option in enum_type)
        raise ValueError(
            f"Invalid {argument_name} {value!r}. Expected one of: {options}."
        ) from exc


def _winner_from_payload(payload: StepPayload) -> tuple[WinnerInfo, float]:
    (
        selected_score,
        left_word,
        left_ix,
        right_word,
        right_ix,
        merged_word,
        merged_ix,
        bigram_locations,
    ) = payload

    winner = WinnerInfo(
        bigram=(
            Lexeme(tuple(left_word), left_ix),
            Lexeme(tuple(right_word), right_ix),
        ),
        merged_lexeme=Lexeme(tuple(merged_word), merged_ix),
        bigram_locations=[
            (LineIndex(line_ix), TokenIndex(token_ix))
            for (line_ix, token_ix) in bigram_locations
        ],
    )
    return winner, selected_score


def _check_engine_status(
    status: int,
    *,
    selected_score: float | None,
    min_score: float | None,
    on_exhausted: ExhaustionPolicy,
    method: SelectionMethod,
    min_count: int,
) -> None:
    if status == STATUS_NO_CANDIDATE and on_exhausted is ExhaustionPolicy.raise_:
        raise NoCandidateBigramError(
            f"No candidate bigrams available for method={method.value!r} "
            f"and min_count={min_count}."
        )

    if status == STATUS_BELOW_MIN_SCORE and on_exhausted is ExhaustionPolicy.raise_:
        raise NoCandidateBigramError(
            f"Best candidate score ({selected_score}) is below min_score ({min_score})."
        )

    if status not in {STATUS_COMPLETED, STATUS_NO_CANDIDATE, STATUS_BELOW_MIN_SCORE}:
        raise RuntimeError(f"Unexpected engine status code {status!r}.")


def _make_engine(
    corpus: list[str],
    method: SelectionMethod | str,
    min_count: int,
    tie_breaker: TieBreaker | str,
    splitter: Splitter | str,
    line_delimiter: str | None,
    sentencex_language: str,
) -> tuple[Engine, SelectionMethod, TieBreaker, Splitter]:
    method = _coerce_enum(method, SelectionMethod, "method")
    splitter = _coerce_enum(splitter, Splitter, "splitter")
    tie_breaker = _coerce_enum(tie_breaker, TieBreaker, "tie_breaker")

    engine = Engine(
        corpus,
        method.value,
        min_count,
        tie_breaker.value,
        splitter.value,
        line_delimiter,
        sentencex_language,
    )
    return engine, method, tie_breaker, splitter


def run(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    output: Path | None = None,
    output_debug_each_iteration: bool = False,
    tie_breaker: TieBreaker | str = TieBreaker.deterministic,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
) -> list[WinnerInfo]:
    """Run the remerge algorithm."""
    engine, method, _tie_breaker, _splitter = _make_engine(
        corpus,
        method,
        min_count,
        tie_breaker,
        splitter,
        line_delimiter,
        sentencex_language,
    )
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    if output is not None:
        print(f"Outputting winning merged lexemes to '{output}'")

    status, payloads, selected_score, _corpus_length = engine.run(iterations, min_score)
    _check_engine_status(
        status,
        selected_score=selected_score,
        min_score=min_score,
        on_exhausted=on_exhausted,
        method=method,
        min_count=min_count,
    )

    winners: list[WinnerInfo] = []
    for payload in payloads:
        winner, _ = _winner_from_payload(payload)
        winners.append(winner)

    if output is not None:
        if output_debug_each_iteration:
            for i in range(len(winners)):
                winner_lexemes = {
                    j: winners[j].merged_lexeme.word for j in range(i + 1)
                }
                output.write_text(json.dumps(winner_lexemes))
        else:
            winner_lexemes = {i: w.merged_lexeme.word for i, w in enumerate(winners)}
            output.write_text(json.dumps(winner_lexemes))

    return winners


def annotate(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    tie_breaker: TieBreaker | str = TieBreaker.deterministic,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    mwe_prefix: str = "<mwe:",
    mwe_suffix: str = ">",
    token_separator: str = "_",
) -> tuple[list[WinnerInfo], list[str], list[str]]:
    """Run the remerge algorithm and annotate the merged corpus."""
    engine, method, _tie_breaker, _splitter = _make_engine(
        corpus,
        method,
        min_count,
        tie_breaker,
        splitter,
        line_delimiter,
        sentencex_language,
    )
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    (
        status,
        payloads,
        selected_score,
        _corpus_length,
        annotated_docs,
        mwe_labels,
    ) = engine.run_and_annotate(
        iterations,
        min_score,
        mwe_prefix,
        mwe_suffix,
        token_separator,
    )
    _check_engine_status(
        status,
        selected_score=selected_score,
        min_score=min_score,
        on_exhausted=on_exhausted,
        method=method,
        min_count=min_count,
    )

    winners: list[WinnerInfo] = []
    for payload in payloads:
        winner, _ = _winner_from_payload(payload)
        winners.append(winner)

    return winners, annotated_docs, mwe_labels

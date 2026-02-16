from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TypeVar

from ._core import (
    Engine,
    STATUS_BELOW_MIN_SCORE,
    STATUS_COMPLETED,
    STATUS_NO_CANDIDATE,
    StepResult,
)


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"


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


Bigram = tuple[Lexeme, Lexeme]


@dataclass(frozen=True)
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    bigram_locations: list[tuple[int, int]]

    @cached_property
    def cleaned_bigram_locations(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.bigram_locations)

    def clean_bigram_locations(self) -> list[tuple[int, int]]:
        return list(self.bigram_locations)

    @property
    def n_lexemes(self) -> int:
        return len(self.merged_lexeme.word)

    @property
    def merge_token_count(self) -> int:
        return len(self.bigram_locations)


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


def _winner_from_step_result(step_result: StepResult) -> tuple[WinnerInfo, float]:
    winner = WinnerInfo(
        bigram=(
            Lexeme(tuple(step_result.left_word), step_result.left_ix),
            Lexeme(tuple(step_result.right_word), step_result.right_ix),
        ),
        merged_lexeme=Lexeme(tuple(step_result.merged_word), step_result.merged_ix),
        bigram_locations=[
            (line_ix, token_ix) for (line_ix, token_ix) in step_result.bigram_locations
        ],
    )
    return winner, step_result.score


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
    splitter: Splitter | str,
    line_delimiter: str | None,
    sentencex_language: str,
    rescore_interval: int,
) -> tuple[Engine, SelectionMethod, Splitter]:
    method = _coerce_enum(method, SelectionMethod, "method")
    splitter = _coerce_enum(splitter, Splitter, "splitter")
    if rescore_interval < 1:
        raise ValueError("rescore_interval must be greater than or equal to 1.")

    engine = Engine(
        corpus,
        method.value,
        min_count,
        splitter.value,
        line_delimiter,
        sentencex_language,
        rescore_interval,
    )
    return engine, method, splitter


def run(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
) -> list[WinnerInfo]:
    """Run the remerge algorithm."""
    engine, method, _splitter = _make_engine(
        corpus,
        method,
        min_count,
        splitter,
        line_delimiter,
        sentencex_language,
        rescore_interval,
    )
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    status, step_results, selected_score, _corpus_length = engine.run(
        iterations, min_score
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
    for step_result in step_results:
        winner, _ = _winner_from_step_result(step_result)
        winners.append(winner)

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
    rescore_interval: int = 25,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    mwe_prefix: str = "<mwe:",
    mwe_suffix: str = ">",
    token_separator: str = "_",
) -> tuple[list[WinnerInfo], list[str], list[str]]:
    """Run the remerge algorithm and annotate the merged corpus."""
    engine, method, _splitter = _make_engine(
        corpus,
        method,
        min_count,
        splitter,
        line_delimiter,
        sentencex_language,
        rescore_interval,
    )
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    (
        status,
        step_results,
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
    for step_result in step_results:
        winner, _ = _winner_from_step_result(step_result)
        winners.append(winner)

    return winners, annotated_docs, mwe_labels

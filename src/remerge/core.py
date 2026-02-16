from dataclasses import dataclass
from enum import Enum
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


@dataclass(frozen=True, slots=True)
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    score: float
    merge_token_count: int

    @property
    def n_lexemes(self) -> int:
        return len(self.merged_lexeme.word)


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


def _winner_from_step_result(step_result: StepResult) -> WinnerInfo:
    return WinnerInfo(
        bigram=(
            Lexeme(tuple(step_result.left_word), step_result.left_ix),
            Lexeme(tuple(step_result.right_word), step_result.right_ix),
        ),
        merged_lexeme=Lexeme(tuple(step_result.merged_word), step_result.merged_ix),
        score=step_result.score,
        merge_token_count=step_result.merge_token_count,
    )


def _collect_winners(step_results: list[StepResult]) -> list[WinnerInfo]:
    return [_winner_from_step_result(step_result) for step_result in step_results]


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
    if min_count < 0:
        raise ValueError("min_count must be greater than or equal to 0.")
    if rescore_interval < 1:
        raise ValueError("rescore_interval must be greater than or equal to 1.")
    if splitter is Splitter.sentencex and not sentencex_language.strip():
        raise ValueError("sentencex_language must be a non-empty language code.")

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


def _run_core(
    corpus: list[str],
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
) -> tuple[Engine, SelectionMethod, ExhaustionPolicy]:
    engine, method, _splitter = _make_engine(
        corpus,
        method,
        min_count,
        splitter,
        line_delimiter,
        sentencex_language,
        rescore_interval,
    )
    return engine, method, _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")


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
    """Run the remerge algorithm.

    The returned winners include:
    - ``score``: the candidate score used to select each winning bigram
      (frequency, log-likelihood, or NPMI depending on ``method``).
    - ``merge_token_count``: number of non-overlapping merge applications for
      that winner in the current iteration.
    """
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")

    engine, method, on_exhausted = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
        on_exhausted=on_exhausted,
    )

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
    return _collect_winners(step_results)


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
    """Run the remerge algorithm and annotate the merged corpus.

    ``annotate()`` uses the same winner payload as ``run()``.
    Output text is whitespace-normalized because tokenization is done with
    Rust ``split_whitespace()`` and reconstructed with single-space joins.
    """
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")

    engine, method, on_exhausted = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
        on_exhausted=on_exhausted,
    )

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
    return _collect_winners(step_results), annotated_docs, mwe_labels

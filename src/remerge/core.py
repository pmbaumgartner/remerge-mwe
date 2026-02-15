import json
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Literal, NewType, TypeVar

from tqdm import tqdm, trange

from ._core import Engine


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"


class TieBreaker(str, Enum):
    deterministic = "deterministic"
    legacy_first_seen = "legacy_first_seen"


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
ProgressBarOptions = Literal["all", "iterations", "none"]


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


def _winner_from_payload(payload: StepPayload) -> tuple[WinnerInfo, float, set[LineIndex]]:
    (
        selected_score,
        left_word,
        left_ix,
        right_word,
        right_ix,
        merged_word,
        merged_ix,
        bigram_locations,
        cleaned_locations,
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
    line_hits = {LineIndex(line_ix) for (line_ix, _) in cleaned_locations}
    return winner, selected_score, line_hits


def run(
    corpus: list[list[str]],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    output: Path | None = None,
    progress_bar: ProgressBarOptions = "iterations",
    tie_breaker: TieBreaker | str = TieBreaker.deterministic,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
) -> list[WinnerInfo]:
    """Run the remerge algorithm."""
    method = _coerce_enum(method, SelectionMethod, "method")
    tie_breaker = _coerce_enum(tie_breaker, TieBreaker, "tie_breaker")
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    winners: list[WinnerInfo] = []
    engine = Engine(corpus, method.value, min_count, tie_breaker.value)

    if output is not None:
        print(f"Outputting winning merged lexemes to '{output}'")

    iterations_iter = (
        trange(iterations)
        if progress_bar in {"all", "iterations"}
        else range(iterations)
    )

    for _ in iterations_iter:
        status, payload, selected_score = engine.step(min_score)

        if status == "no_candidate":
            if on_exhausted is ExhaustionPolicy.raise_:
                raise NoCandidateBigramError(
                    f"No candidate bigrams available for method={method.value!r} "
                    f"and min_count={min_count}."
                )
            break

        if status == "below_min_score":
            if on_exhausted is ExhaustionPolicy.raise_:
                raise NoCandidateBigramError(
                    f"Best candidate score ({selected_score}) is below "
                    f"min_score ({min_score})."
                )
            break

        if status != "winner" or payload is None:
            raise RuntimeError(f"Unexpected engine status {status!r}.")

        winner, score, line_hits = _winner_from_payload(payload)
        winners.append(winner)

        if output is not None:
            winner_lexemes = {i: w.merged_lexeme.word for i, w in enumerate(winners)}
            output.write_text(json.dumps(winner_lexemes))

        if isinstance(iterations_iter, tqdm):
            corpus_length = engine.corpus_length()
            pct_bgr = len(line_hits) / corpus_length if corpus_length else 0.0
            iterations_iter.set_postfix(
                {
                    "last_winner": winner.merged_lexeme.word,
                    "score": f"{score:.4g}",
                    "pct_bgr": f"{pct_bgr*100:.1f}%",
                }
            )

    return winners

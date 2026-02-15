import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import groupby, islice
from pathlib import Path
from collections.abc import Callable, Iterable, Sized
from typing import Literal, NamedTuple, NewType, TypeVar

import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange

_SMALL = 1e-10


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


class Lexeme(NamedTuple):
    word: tuple[str, ...]
    ix: int

    def __repr__(self) -> str:
        return f"({self.word}|{self.ix})"


LineIndex = NewType("LineIndex", int)
TokenIndex = NewType("TokenIndex", int)


@dataclass
class LexemeData:
    lexemes_to_locations: defaultdict[
        Lexeme, set[tuple[LineIndex, TokenIndex]]
    ] = field(default_factory=lambda: defaultdict(set))
    locations_to_lexemes: list[list[Lexeme]] = field(default_factory=list)
    lexemes_to_freqs: dict[Lexeme, int] = field(default_factory=dict)

    @classmethod
    def from_corpus(
        cls, corpus: Iterable[Iterable[str]], progress_bar: bool = False
    ) -> "LexemeData":
        lexeme_data = cls()
        total: int | None = len(corpus) if isinstance(corpus, Sized) else None
        corpus_iter = enumerate(corpus)
        if progress_bar:
            corpus_iter = tqdm(
                corpus_iter,
                desc="Creating LexemeData from Corpus",
                unit="line",
                total=total,
            )
        for (line_ix, tokens) in corpus_iter:
            line_lexemes = []
            for (word_ix, word) in enumerate(tokens):
                line_ix = LineIndex(line_ix)
                word_ix = TokenIndex(word_ix)
                lexeme = Lexeme(word=(word,), ix=0)
                loc = (line_ix, word_ix)
                lexeme_data.lexemes_to_locations[lexeme].add(loc)
                line_lexemes.append(lexeme)
            lexeme_data.locations_to_lexemes.append(line_lexemes)

        # Using this conditional prevents double counting merged lexemes.
        lexeme_data.lexemes_to_freqs = {
            k: len(v) for k, v in lexeme_data.lexemes_to_locations.items() if k.ix == 0
        }
        return lexeme_data

    @property
    def corpus_length(self) -> int:
        """Returns number of lines in corpus: max(line_ix) + 1."""
        return len(self.locations_to_lexemes)

    def render_corpus(self) -> list[list[Lexeme]]:
        return self.locations_to_lexemes

    def locations_to_root_lexemes(self, line: LineIndex) -> dict[TokenIndex, Lexeme]:
        lexeme_dicts = self.locations_to_lexemes[line]
        return {TokenIndex(k): v for k, v in enumerate(lexeme_dicts) if v.ix == 0}


Bigram = tuple[Lexeme, Lexeme]


def _count_bigram_line(*args):
    el1c = [b[0] for b in args]
    el2c = [b[1] for b in args]
    bc = [b for b in args]
    return (el1c, el2c, bc)


@dataclass
class BigramData:
    bigrams_to_freqs: Counter[Bigram] = field(default_factory=Counter)
    bigrams_to_locations: dict[Bigram, set[tuple[LineIndex, TokenIndex]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    left_lex_freqs: Counter[Lexeme] = field(default_factory=Counter)
    right_lex_freqs: Counter[Lexeme] = field(default_factory=Counter)

    @classmethod
    def from_lexemes(
        cls, lexeme_data: LexemeData, progress_bar: bool = False
    ) -> "BigramData":
        bigram_data = cls()
        corpus_iter = range(lexeme_data.corpus_length)
        if progress_bar:
            corpus_iter = tqdm(
                corpus_iter,
                desc="Creating BigramData from LexemeData",
                unit="line",
                total=lexeme_data.corpus_length - 1,
            )
        for line_ix in corpus_iter:
            line_lexeme_data = lexeme_data.locations_to_root_lexemes(LineIndex(line_ix))
            line_items = line_lexeme_data.items()
            line_bigrams = []
            for (left_ix, left), (_, right) in zip(
                line_items, islice(line_items, 1, None)
            ):
                bigram = (left, right)
                location = (LineIndex(line_ix), TokenIndex(left_ix))
                bigram_data.bigrams_to_locations[bigram].add(location)
                line_bigrams.append(bigram)
            bigram_data.batch_add_bigrams(line_bigrams)
        return bigram_data

    def batch_add_bigrams(self, bigram_locations: list[Bigram]):
        el1s, el2s, bigrams = _count_bigram_line(*bigram_locations)
        self.left_lex_freqs.update(el1s)
        self.right_lex_freqs.update(el2s)
        self.bigrams_to_freqs.update(bigrams)


@dataclass
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    bigram_locations: list[tuple[LineIndex, TokenIndex]]

    @classmethod
    def from_bigram_with_data(
        cls, bigram: Bigram, bigram_data: BigramData
    ) -> "WinnerInfo":
        el1_words = list(bigram[0].word)
        el2_words = list(bigram[1].word)
        all_words = el1_words + el2_words
        new_lexeme = Lexeme(word=tuple(all_words), ix=0)
        locations = sorted(bigram_data.bigrams_to_locations[bigram])
        return cls(bigram=bigram, merged_lexeme=new_lexeme, bigram_locations=locations)

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


def merge_winner(
    winner: WinnerInfo, lexeme_data: LexemeData, bigram_data: BigramData
) -> tuple[LexemeData, BigramData]:
    clean_locations = winner.cleaned_bigram_locations
    bigram_lines = {line_ix for line_ix, _ in clean_locations}
    old_bigrams_lookup = {
        line_ix: list(lexeme_data.locations_to_root_lexemes(LineIndex(line_ix)).items())
        for line_ix in bigram_lines
    }
    for (line_ix, word_ix) in clean_locations:
        for lexeme_index in range(winner.n_lexemes):
            pos = TokenIndex(word_ix + lexeme_index)
            old_lexeme = lexeme_data.locations_to_lexemes[line_ix][pos]
            lexeme = Lexeme(word=winner.merged_lexeme.word, ix=lexeme_index)
            lexeme_data.locations_to_lexemes[line_ix][pos] = lexeme
            lexeme_data.lexemes_to_locations[old_lexeme].remove((line_ix, pos))
            lexeme_data.lexemes_to_locations[lexeme].add((line_ix, pos))

    for line_ix, lexemes in old_bigrams_lookup.items():
        old_root_lexemes = [lexeme_item[1] for lexeme_item in lexemes]
        old_bigrams = list(zip(old_root_lexemes, islice(old_root_lexemes, 1, None)))

        new_root_lexemes_items = list(
            lexeme_data.locations_to_root_lexemes(LineIndex(line_ix)).items()
        )
        new_root_lexemes = [lex for _, lex in new_root_lexemes_items]
        new_bigrams = list(zip(new_root_lexemes, islice(new_root_lexemes, 1, None)))

        bigram_data.bigrams_to_freqs.update(new_bigrams)
        bigram_data.left_lex_freqs.update([b[0] for b in new_bigrams])
        bigram_data.right_lex_freqs.update([b[1] for b in new_bigrams])

        bigram_data.bigrams_to_freqs.subtract(old_bigrams)
        bigram_data.left_lex_freqs.subtract([b[0] for b in old_bigrams])
        bigram_data.right_lex_freqs.subtract([b[1] for b in old_bigrams])

        for (left_ix, left), (_, right) in zip(lexemes, islice(lexemes, 1, None)):
            bigram = (left, right)
            location = (LineIndex(line_ix), TokenIndex(left_ix))
            bigram_data.bigrams_to_locations[bigram].remove(location)

        for (left_ix, left), (_, right) in zip(
            new_root_lexemes_items, islice(new_root_lexemes_items, 1, None)
        ):
            bigram = (left, right)
            location = (LineIndex(line_ix), TokenIndex(left_ix))
            bigram_data.bigrams_to_locations[bigram].add(location)

    merge_token_count = winner.merge_token_count
    lexeme_data.lexemes_to_freqs[winner.merged_lexeme] = merge_token_count

    el1_freq = lexeme_data.lexemes_to_freqs[winner.bigram[0]]
    lexeme_data.lexemes_to_freqs[winner.bigram[0]] = el1_freq - merge_token_count

    el2_freq = lexeme_data.lexemes_to_freqs[winner.bigram[1]]
    lexeme_data.lexemes_to_freqs[winner.bigram[1]] = el2_freq - merge_token_count

    lexeme_data.lexemes_to_freqs = {
        k: v for k, v in lexeme_data.lexemes_to_freqs.items() if v != 0
    }
    lexeme_data.lexemes_to_locations = defaultdict(
        set, {k: v for k, v in lexeme_data.lexemes_to_locations.items() if v}
    )

    bigram_data.bigrams_to_freqs = Counter(
        {k: v for k, v in bigram_data.bigrams_to_freqs.items() if v > 0}
    )
    bigram_data.left_lex_freqs = Counter(
        {k: v for k, v in bigram_data.left_lex_freqs.items() if v > 0}
    )
    bigram_data.right_lex_freqs = Counter(
        {k: v for k, v in bigram_data.right_lex_freqs.items() if v > 0}
    )
    bigram_data.bigrams_to_locations = defaultdict(
        set, {k: v for k, v in bigram_data.bigrams_to_locations.items() if v}
    )
    assert winner.bigram not in bigram_data.bigrams_to_freqs
    return lexeme_data, bigram_data


@dataclass(frozen=True)
class BigramCandidateArrays:
    bigram_index: list[Bigram]
    bigram_freq_array: npt.NDArray[np.int64]
    el1_freq_array: npt.NDArray[np.int64]
    el2_freq_array: npt.NDArray[np.int64]
    total_bigram_count: int

    @classmethod
    def from_bigram_data(
        cls, bigram_data: BigramData, min_count: int = 0
    ) -> "BigramCandidateArrays":
        total_bigram_count = int(sum(bigram_data.bigrams_to_freqs.values()))
        candidate_items = [
            (bigram, freq)
            for (bigram, freq) in bigram_data.bigrams_to_freqs.items()
            if freq >= min_count
        ]

        if not candidate_items:
            empty = np.array([], dtype=np.int64)
            return cls([], empty, empty, empty, total_bigram_count)

        bigram_index = [bigram for bigram, _ in candidate_items]
        bigram_freq_array = np.array([freq for _, freq in candidate_items], dtype=np.int64)
        el1_freq_array = np.array(
            [bigram_data.left_lex_freqs[bigram[0]] for bigram, _ in candidate_items],
            dtype=np.int64,
        )
        el2_freq_array = np.array(
            [bigram_data.right_lex_freqs[bigram[1]] for bigram, _ in candidate_items],
            dtype=np.int64,
        )
        return cls(
            bigram_index=bigram_index,
            bigram_freq_array=bigram_freq_array,
            el1_freq_array=el1_freq_array,
            el2_freq_array=el2_freq_array,
            total_bigram_count=total_bigram_count,
        )


@dataclass(frozen=True)
class ScoredBigram:
    bigram: Bigram
    score: float
    frequency: int

    @property
    def merged_word(self) -> tuple[str, ...]:
        return self.bigram[0].word + self.bigram[1].word


def _safe_ll_term(
    observed: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.where(
        observed > 0,
        observed * np.log((observed / (expected + _SMALL)) + _SMALL),
        0.0,
    )


def _calculate_npmi(data: BigramCandidateArrays) -> npt.NDArray[np.float64]:
    if data.total_bigram_count == 0:
        return np.array([], dtype=np.float64)
    prob_ab = data.bigram_freq_array / data.total_bigram_count
    prob_a = data.el1_freq_array / data.total_bigram_count
    prob_b = data.el2_freq_array / data.total_bigram_count
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.log(prob_ab / (prob_a * prob_b))
        denominator = -(np.log(prob_ab))
        npmi = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
    perfect_association = np.isclose(denominator, 0.0) & np.isclose(numerator, 0.0)
    return np.where(perfect_association, 1.0, npmi)


def _calculate_log_likelihood(data: BigramCandidateArrays) -> npt.NDArray[np.float64]:
    if data.total_bigram_count == 0:
        return np.array([], dtype=np.float64)

    # For reference, see also: nltk.collocations.BigramAssocMeasures, specifically
    # _contingency in nltk.metrics.association.
    obs_a = data.bigram_freq_array.astype(np.float64)
    obs_b = data.el1_freq_array.astype(np.float64) - obs_a
    obs_c = data.el2_freq_array.astype(np.float64) - obs_a
    obs_d = data.total_bigram_count - obs_a - obs_b - obs_c
    obs_d = np.maximum(obs_d, 0.0)

    exp_a = ((obs_a + obs_b) * (obs_a + obs_c)) / data.total_bigram_count
    exp_b = ((obs_a + obs_b) * (obs_b + obs_d)) / data.total_bigram_count
    exp_c = ((obs_c + obs_d) * (obs_a + obs_c)) / data.total_bigram_count
    exp_d = ((obs_c + obs_d) * (obs_b + obs_d)) / data.total_bigram_count

    ll_a = _safe_ll_term(obs_a, exp_a)
    ll_b = _safe_ll_term(obs_b, exp_b)
    ll_c = _safe_ll_term(obs_c, exp_c)
    ll_d = _safe_ll_term(obs_d, exp_d)

    log_likelihood = 2.0 * (ll_a + ll_b + ll_c + ll_d)
    return np.where(obs_a > exp_a, log_likelihood, log_likelihood * -1.0)


def _coerce_scores(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if scores.size == 0:
        return scores
    return np.where(np.isfinite(scores), scores, -np.inf)


def _as_scored_bigrams(
    data: BigramCandidateArrays, scores: npt.NDArray[np.float64]
) -> list[ScoredBigram]:
    safe_scores = _coerce_scores(scores)
    if safe_scores.size == 0 or np.all(safe_scores == -np.inf):
        return []
    return [
        ScoredBigram(
            bigram=bigram,
            score=float(score),
            frequency=int(freq),
        )
        for bigram, score, freq in zip(
            data.bigram_index, safe_scores, data.bigram_freq_array
        )
    ]


def calculate_candidates_log_likelihood(
    bigram_data: BigramData, min_count: int = 0
) -> list[ScoredBigram]:
    data = BigramCandidateArrays.from_bigram_data(bigram_data, min_count=min_count)
    scores = _calculate_log_likelihood(data)
    return _as_scored_bigrams(data, scores)


def calculate_candidates_npmi(
    bigram_data: BigramData, min_count: int = 0
) -> list[ScoredBigram]:
    data = BigramCandidateArrays.from_bigram_data(bigram_data, min_count=min_count)
    scores = _calculate_npmi(data)
    return _as_scored_bigrams(data, scores)


def calculate_candidates_frequency(
    bigram_data: BigramData, min_count: int = 0
) -> list[ScoredBigram]:
    candidates = []
    for bigram, freq in bigram_data.bigrams_to_freqs.items():
        if freq < min_count:
            continue
        candidates.append(
            ScoredBigram(bigram=bigram, score=float(freq), frequency=int(freq))
        )
    return candidates


SelectionFunction = Callable[[BigramData, int], list[ScoredBigram]]
SELECTION_METHODS: dict[SelectionMethod, SelectionFunction] = {
    SelectionMethod.log_likelihood: calculate_candidates_log_likelihood,
    SelectionMethod.frequency: calculate_candidates_frequency,
    SelectionMethod.npmi: calculate_candidates_npmi,
}

ProgressBarOptions = Literal["all", "iterations", "none"]

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


def _select_candidate(
    candidates: list[ScoredBigram], tie_breaker: TieBreaker
) -> ScoredBigram | None:
    if not candidates:
        return None

    max_score = max(candidate.score for candidate in candidates)
    if tie_breaker is TieBreaker.legacy_first_seen:
        for candidate in candidates:
            if np.isclose(candidate.score, max_score, atol=1e-12, rtol=1e-12):
                return candidate
        return None

    top_score_candidates = [
        candidate
        for candidate in candidates
        if np.isclose(candidate.score, max_score, atol=1e-12, rtol=1e-12)
    ]
    top_score_candidates.sort(key=lambda c: (-c.frequency, c.merged_word))
    return top_score_candidates[0] if top_score_candidates else None


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
    """Run the remerge algorithm.

    Args:
        corpus (list[list[str]]): A corpus of already tokenized texts.
        iterations (int): The number of iterations to run the algorithm.
        method (SelectionMethod | str, optional): One of "frequency", "log_likelihood",
          or "npmi". Defaults to "log_likelihood".
        min_count (int, optional): The minimum count required for a bigram to be included
          in winner calculations. Defaults to 0.
        output (Path | None, optional): A file path to output winners as JSON.
          Defaults to None.
        progress_bar (ProgressBarOptions, optional): Verbosity of progress bar.
          Defaults to "iterations".
        tie_breaker (TieBreaker | str, optional): Tie-breaking strategy for equal-score
          candidates. Defaults to "deterministic".
        on_exhausted (ExhaustionPolicy | str, optional): Behavior when no candidate
          remains. Defaults to "stop".
        min_score (float | None, optional): Stop or raise when the best score is below
          this threshold. Defaults to None.

    Returns:
        list[WinnerInfo]: The winning bigram from each executed iteration.
    """
    method = _coerce_enum(method, SelectionMethod, "method")
    tie_breaker = _coerce_enum(tie_breaker, TieBreaker, "tie_breaker")
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    winners: list[WinnerInfo] = []
    all_progress = progress_bar == "all"
    lexemes = LexemeData.from_corpus(corpus, progress_bar=all_progress)
    bigrams = BigramData.from_lexemes(lexemes, progress_bar=all_progress)
    winner_selection_function = SELECTION_METHODS[method]

    if output is not None:
        print(f"Outputting winning merged lexemes to '{output}'")

    iterations_iter = (
        trange(iterations)
        if progress_bar in {"all", "iterations"}
        else range(iterations)
    )

    for _ in iterations_iter:
        candidates = winner_selection_function(bigrams, min_count)
        selected = _select_candidate(candidates, tie_breaker)

        if selected is None:
            if on_exhausted is ExhaustionPolicy.raise_:
                raise NoCandidateBigramError(
                    f"No candidate bigrams available for method={method.value!r} "
                    f"and min_count={min_count}."
                )
            break

        if min_score is not None and selected.score < min_score:
            if on_exhausted is ExhaustionPolicy.raise_:
                raise NoCandidateBigramError(
                    f"Best candidate score ({selected.score}) is below "
                    f"min_score ({min_score})."
                )
            break

        winner = WinnerInfo.from_bigram_with_data(
            bigram=selected.bigram, bigram_data=bigrams
        )
        winners.append(winner)

        if output:
            winner_lexemes = {i: w.merged_lexeme.word for i, w in enumerate(winners)}
            output.write_text(json.dumps(winner_lexemes))

        lexemes, bigrams = merge_winner(winner, lexemes, bigrams)

        if isinstance(iterations_iter, tqdm):
            lines = {line for line, _ in winner.cleaned_bigram_locations}
            pct_bgr = len(lines) / lexemes.corpus_length if lexemes.corpus_length else 0.0
            iterations_iter.set_postfix(
                {
                    "last_winner": winner.merged_lexeme.word,
                    "score": f"{selected.score:.4g}",
                    "pct_bgr": f"{pct_bgr*100:.1f}%",
                }
            )

    return winners

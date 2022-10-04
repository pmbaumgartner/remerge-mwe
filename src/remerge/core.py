import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import groupby, islice
from pathlib import Path
from typing import Callable
from typing import Counter as CounterType
from typing import (
    DefaultDict,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Set,
    Sized,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange

_SMALL: Final[float] = 1e-10


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"


class Lexeme(NamedTuple):
    word: Tuple[str, ...]
    ix: int

    def __repr__(self) -> str:
        return f"({self.word}|{self.ix})"


LineIndex = NewType("LineIndex", int)
TokenIndex = NewType("TokenIndex", int)


@dataclass
class LexemeData:
    lexemes_to_locations: DefaultDict[
        Lexeme, Set[Tuple[LineIndex, TokenIndex]]
    ] = field(default_factory=lambda: defaultdict(set))
    locations_to_lexemes: List[List[Lexeme]] = field(default_factory=list)
    lexemes_to_freqs: Dict[Lexeme, int] = field(default_factory=dict)

    @classmethod
    def from_corpus(
        cls, corpus: Iterable[Iterable[str]], progress_bar: bool = False
    ) -> "LexemeData":
        lexeme_data = cls()
        total: Optional[int] = len(corpus) if isinstance(corpus, Sized) else None
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

        # NOTE: Using this conditional prevents double counting merged lexemes.
        lexeme_data.lexemes_to_freqs = {
            k: len(v) for k, v in lexeme_data.lexemes_to_locations.items() if k.ix == 0
        }
        return lexeme_data

    @property
    def corpus_length(self) -> int:
        """Returns number of lines in corpus: max(line_ix) + 1."""
        return len(self.locations_to_lexemes)

    def render_corpus(self) -> List[List[Lexeme]]:
        return self.locations_to_lexemes

    def locations_to_root_lexemes(self, line: LineIndex) -> Dict[TokenIndex, Lexeme]:
        lexeme_dicts = self.locations_to_lexemes[line]
        return {TokenIndex(k): v for k, v in enumerate(lexeme_dicts) if v.ix == 0}


Bigram = Tuple[Lexeme, Lexeme]


def _count_bigram_line(*args):
    el1c = [b[0] for b in args]
    el2c = [b[1] for b in args]
    bc = [b for b in args]
    return (el1c, el2c, bc)


@dataclass
class BigramData:
    bigrams_to_freqs: CounterType[Bigram] = field(default_factory=Counter)
    bigrams_to_locations: Dict[Bigram, List[Tuple[LineIndex, TokenIndex]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    left_lex_freqs: CounterType[Lexeme] = field(default_factory=Counter)
    right_lex_freqs: CounterType[Lexeme] = field(default_factory=Counter)

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
                bigram_data.bigrams_to_locations[bigram].append(location)
                line_bigrams.append(bigram)
            bigram_data.batch_add_bigrams(line_bigrams)
        return bigram_data

    def batch_add_bigrams(self, bigram_locations: List[Bigram]):
        el1s, el2s, bigrams = _count_bigram_line(*bigram_locations)
        self.left_lex_freqs.update(el1s)
        self.right_lex_freqs.update(el2s)
        self.bigrams_to_freqs.update(bigrams)

    @property
    def bigram_count(self) -> int:
        """Counts the total number of bigrams."""
        return sum(self.bigrams_to_freqs.values())


@dataclass
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    bigram_locations: List[Tuple[LineIndex, TokenIndex]]

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

    def clean_bigram_locations(self) -> List[Tuple[LineIndex, TokenIndex]]:
        """This is greedily selecting correct bigrams from the candidate locations of bigrams.

        Why? Well, in the case of a sentence like (a, a, a), with winner = (a, a), we can only convert
        the first occurrence of this bigram and not the second, since the first occurence would be transformed into the bigram,
        the new bigram in the second position no longer exists - but could be a candidate for the next round if it is indeed that common
        of a pattern.

        A more complex example is with winner (a, b, a, b) in ((a, b), (a, b), (a, b)). Here is the same idea: once we
        merge the first occurence it is no longer available, even though it occurs later.
        """
        clean_locations: List[Tuple[LineIndex, TokenIndex]] = []
        for line, location in groupby(self.bigram_locations, key=lambda x: x[0]):
            exclude_token: Set[TokenIndex] = set()
            token_ix = [i[1] for i in location]
            for token in token_ix:
                if token in exclude_token:
                    continue
                excludes = [i for i in token_ix if i < token + self.n_lexemes]
                exclude_token.update(excludes)
                clean_locations.append((line, token))
        return clean_locations

    @property
    def n_lexemes(self) -> int:
        return len(self.merged_lexeme.word)

    @property
    def merge_token_count(self) -> int:
        # TODO: Optimize by putting in loop so we don't have to iterate here
        return len(self.clean_bigram_locations())


def merge_winner(
    winner: WinnerInfo, lexeme_data: LexemeData, bigram_data: BigramData
) -> Tuple[LexemeData, BigramData]:
    bigram_lines = set(i[0] for i in winner.clean_bigram_locations())
    old_bigrams_lookup = {
        line_ix: list(lexeme_data.locations_to_root_lexemes(LineIndex(line_ix)).items())
        for line_ix in bigram_lines
    }
    for (line_ix, word_ix) in winner.clean_bigram_locations():
        # Do Updates
        for lexeme_index in range(winner.n_lexemes):
            pos = TokenIndex(word_ix + lexeme_index)
            old_lexeme = lexeme_data.locations_to_lexemes[line_ix][pos]
            lexeme = Lexeme(word=winner.merged_lexeme.word, ix=lexeme_index)
            lexeme_data.locations_to_lexemes[line_ix][pos] = lexeme
            lexeme_data.lexemes_to_locations[old_lexeme].remove((line_ix, pos))
            lexeme_data.lexemes_to_locations[lexeme].add((line_ix, pos))
    for line_ix, lexemes in old_bigrams_lookup.items():

        old_bigrams = list(
            zip([l[1] for l in lexemes], islice([l[1] for l in lexemes], 1, None))
        )

        new_root_lexemes_items = list(
            lexeme_data.locations_to_root_lexemes(LineIndex(line_ix)).items()
        )
        new_root_lexemes = list(lex for _, lex in new_root_lexemes_items)
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
            bigram_data.bigrams_to_locations[bigram].append(location)

    lexeme_data.lexemes_to_freqs[winner.merged_lexeme] = winner.merge_token_count

    el1_freq = lexeme_data.lexemes_to_freqs[winner.bigram[0]]
    new_el1_freq = el1_freq - winner.merge_token_count
    lexeme_data.lexemes_to_freqs[winner.bigram[0]] = new_el1_freq

    el2_freq = lexeme_data.lexemes_to_freqs[winner.bigram[1]]
    new_el2_freq = el2_freq - winner.merge_token_count
    lexeme_data.lexemes_to_freqs[winner.bigram[1]] = new_el2_freq

    lexeme_data.lexemes_to_freqs = {
        k: v for k, v in lexeme_data.lexemes_to_freqs.items() if v != 0
    }
    lexeme_data.lexemes_to_locations = defaultdict(
        set, {k: v for k, v in lexeme_data.lexemes_to_locations.items() if v != set()}
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
    assert winner.bigram not in bigram_data.bigrams_to_freqs
    return lexeme_data, bigram_data


# NamedTuple doesn't support cached_property
@dataclass(frozen=True)
class BigramFreqArrays:
    bigram_index: List[Bigram]
    bigram_freq_array: npt.NDArray[np.int_]
    el1_freq_array: npt.NDArray[np.int_]
    el2_freq_array: npt.NDArray[np.int_]

    @cached_property
    def bigram_count(self) -> np.int_:
        return self.bigram_freq_array.sum()

    @classmethod
    def from_bigram_data(
        cls, bigram_data: BigramData, min_count: int = 0
    ) -> "BigramFreqArrays":
        length = len(
            [i for i in bigram_data.bigrams_to_freqs.values() if i >= min_count]
        )
        bigram_freq_array = np.empty(length, dtype=np.int_)
        el1_freq_array = np.empty(length, dtype=np.int_)
        el2_freq_array = np.empty(length, dtype=np.int_)
        bigram_index = []
        i = 0
        for (bigram, freq) in bigram_data.bigrams_to_freqs.items():
            if freq < min_count:
                continue
            bigram_freq_array[i] = freq
            l1 = bigram_data.left_lex_freqs[bigram[0]]
            el1_freq_array[i] = l1
            l2 = bigram_data.right_lex_freqs[bigram[1]]
            el2_freq_array[i] = l2
            bigram_index.append(bigram)
            i += 1
            # manually count instead of enumerate
        return cls(bigram_index, bigram_freq_array, el1_freq_array, el2_freq_array)


def calculate_winner_log_likelihood(
    bigram_data: BigramData, min_count: int = 0
) -> Bigram:
    data = BigramFreqArrays.from_bigram_data(bigram_data, min_count=min_count)
    log_likelihoods = _calculate_log_likelihood(data)
    winner_ix = np.argmax(log_likelihoods)
    winner: Bigram = data.bigram_index[winner_ix]
    return winner


def calculate_winner_npmi(bigram_data: BigramData, min_count: int = 0) -> Bigram:
    data = BigramFreqArrays.from_bigram_data(bigram_data, min_count=min_count)
    npmis = _calculate_npmi(data)
    winner_ix = np.argmax(npmis)
    winner: Bigram = data.bigram_index[winner_ix]
    return winner


def calculate_winner_frequency(bigrams: BigramData, min_count: int = 0) -> Bigram:
    return bigrams.bigrams_to_freqs.most_common(1)[0][0]


def _calculate_npmi(data: BigramFreqArrays) -> npt.NDArray[np.float_]:
    prob_ab = data.bigram_freq_array / data.bigram_count
    prob_a = data.el1_freq_array / data.bigram_count
    prob_b = data.el2_freq_array / data.bigram_count
    npmi = np.log(prob_ab / (prob_a * prob_b)) / -(np.log(prob_ab))
    return npmi


def _calculate_log_likelihood(data: BigramFreqArrays) -> npt.NDArray[np.float_]:
    # For reference, see also: nltk.collocations.BigramAssocMeasures, specifically _contingency
    # http://ecologyandevolution.org/statsdocs/online-stats-manual-chapter4.html
    obsA = data.bigram_freq_array
    obsB = data.el1_freq_array - obsA
    obsC = data.el2_freq_array - obsA
    obsD = data.bigram_count - obsA - obsB - obsC

    expA = ((obsA + obsB) * (obsA + obsC)) / data.bigram_count
    expB = ((obsA + obsB) * (obsB + obsD)) / data.bigram_count
    expC = ((obsC + obsD) * (obsA + obsC)) / data.bigram_count
    expD = ((obsC + obsD) * (obsB + obsD)) / data.bigram_count

    llA = obsA * np.log((obsA / (expA + _SMALL)) + _SMALL)
    llB = obsB * np.log((obsB / (expB + _SMALL)) + _SMALL)
    llC = obsC * np.log((obsC / (expC + _SMALL)) + _SMALL)
    llD = obsD * np.log((obsD / (expD + _SMALL)) + _SMALL)

    log_likelihood = 2.0 * (llA + llB + llC + llD)
    log_likelihood = np.where(llA > 0, log_likelihood, log_likelihood * -1.0)
    return log_likelihood


SELECTION_METHODS: Dict[SelectionMethod, Callable[[BigramData, int], Bigram]] = {
    SelectionMethod.log_likelihood: calculate_winner_log_likelihood,
    SelectionMethod.frequency: calculate_winner_frequency,
    SelectionMethod.npmi: calculate_winner_npmi,
}


def run(
    corpus: List[List[str]],
    iterations: int,
    *,
    method: SelectionMethod = SelectionMethod.log_likelihood,
    min_count: int = 0,
    output: Optional[Path] = None,
    progress_bar: Literal["all", "iterations", "none"] = "none",
) -> List[WinnerInfo]:
    """If choosing NPMI as the selection method, prefer using min_count because:

    'infrequent word pairs tend to dominate the top of bigramme lists that are ranked after PMI'
    """
    winners: List[WinnerInfo] = []
    all_progress = progress_bar == "all"
    lexemes = LexemeData.from_corpus(corpus, progress_bar=all_progress)
    bigrams = BigramData.from_lexemes(lexemes, progress_bar=all_progress)
    winner_selection_function = SELECTION_METHODS[method]
    iterations_iter = (
        trange(iterations)
        if progress_bar in {"all", "iterations"}
        else range(iterations)
    )
    for _ in iterations_iter:
        winning_bigram = winner_selection_function(bigrams, min_count)
        winner = WinnerInfo.from_bigram_with_data(
            bigram=winning_bigram, bigram_data=bigrams
        )
        winners.append(winner)
        if output:
            winner_lexemes = {i: w.merged_lexeme.word for i, w in enumerate(winners)}
            output.write_text(json.dumps(winner_lexemes))
        lexemes, bigrams = merge_winner(winner, lexemes, bigrams)
        if isinstance(iterations_iter, tqdm):
            lines = set(w[0] for w in winner.bigram_locations)
            pct_bgr = len(lines) / lexemes.corpus_length
            iterations_iter.set_postfix(
                {
                    "last_winner": winner.merged_lexeme.word,
                    "pct_bgr": f"{pct_bgr*100:.1f}%",
                }
            )
    return winners

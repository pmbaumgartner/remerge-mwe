import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Callable, Literal
from typing import Counter as CounterType
from typing import (
    DefaultDict,
    Dict,
    Iterable,
    List,
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

_SMALL = 1e-10


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"


class Lexeme(NamedTuple):
    word: Tuple[str, ...]
    ix: int

    def __repr__(self) -> str:
        return f"({self.word}, {self.ix})"


LineIndex = NewType("LineIndex", int)
TokenIndex = NewType("TokenIndex", int)


@dataclass
class LexemeData:
    lexemes_to_locations: DefaultDict[
        Lexeme, Set[Tuple[LineIndex, TokenIndex]]
    ] = field(default_factory=lambda: defaultdict(set))
    locations_to_lexemes: DefaultDict[LineIndex, Dict[TokenIndex, Lexeme]] = field(
        default_factory=lambda: defaultdict(dict)
    )
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
            for (word_ix, word) in enumerate(tokens):
                line_ix = LineIndex(line_ix)
                word_ix = TokenIndex(word_ix)
                lexeme = Lexeme(word=(word,), ix=0)
                loc = (line_ix, word_ix)
                lexeme_data.lexemes_to_locations[lexeme].add(loc)
                lexeme_data.locations_to_lexemes[line_ix][word_ix] = lexeme

        # NOTE: Using this conditional prevents double counting merged lexemes.
        lexeme_data.lexemes_to_freqs = {
            k: len(v) for k, v in lexeme_data.lexemes_to_locations.items() if k.ix == 0
        }
        return lexeme_data

    @property
    def corpus_length(self) -> int:
        """Returns number of lines in corpus: max(line_ix) + 1."""
        return max(self.locations_to_lexemes.keys()) + 1

    def render_corpus(self) -> List[List[Lexeme]]:
        rendered_corpus: List[List[Lexeme]] = []
        for line_ix in self.locations_to_lexemes:
            line_tokens = []
            for token_ix in self.locations_to_lexemes[line_ix]:
                line_tokens.append(self.locations_to_lexemes[line_ix][token_ix])
            rendered_corpus.append(line_tokens)
        return rendered_corpus

    def locations_to_root_lexemes(self, line: LineIndex) -> Dict[TokenIndex, Lexeme]:
        lexeme_dicts = self.locations_to_lexemes[line]
        return {k: v for k, v in lexeme_dicts.items() if v.ix == 0}


class Bigram(NamedTuple):
    el1: Lexeme
    el2: Lexeme


@dataclass
class BigramData:
    bigrams_to_freqs: CounterType[Bigram] = field(default_factory=Counter)
    bigrams_to_locations: Dict[Bigram, Set[Tuple[LineIndex, TokenIndex]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    left_lex_freqs: Dict[Lexeme, int] = field(default_factory=dict)
    right_lex_freqs: Dict[Lexeme, int] = field(default_factory=dict)

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
            line_items = list(line_lexeme_data.items())
            for (left_ix, left), (_, right) in zip(line_items, line_items[1:]):
                bigram = Bigram(el1=left, el2=right)
                location = (LineIndex(line_ix), TokenIndex(left_ix))
                bigram_data.add_bigram(bigram, location)
        return bigram_data

    def add_bigram(
        self, bigram: Bigram, location: Tuple[LineIndex, TokenIndex]
    ) -> None:
        if self.left_lex_freqs.get(bigram.el1, None):
            self.left_lex_freqs[bigram.el1] += 1
        else:
            self.left_lex_freqs[bigram.el1] = 1
        if self.right_lex_freqs.get(bigram.el2, None):
            self.right_lex_freqs[bigram.el2] += 1
        else:
            self.right_lex_freqs[bigram.el2] = 1
        self.bigrams_to_freqs[bigram] += 1
        self.bigrams_to_locations[bigram].add(location)

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
        el1_words = list(bigram.el1.word)
        el2_words = list(bigram.el2.word)
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


def merge_winner(winner: WinnerInfo, lexeme_data: LexemeData) -> LexemeData:
    for (line_ix, word_ix) in winner.clean_bigram_locations():
        for lexeme_index in range(winner.n_lexemes):
            pos = TokenIndex(word_ix + lexeme_index)
            old_lexeme = lexeme_data.locations_to_lexemes[line_ix][pos]
            lexeme = Lexeme(word=winner.merged_lexeme.word, ix=lexeme_index)
            lexeme_data.lexemes_to_locations[lexeme].add((LineIndex(line_ix), pos))
            lexeme_data.locations_to_lexemes[line_ix][pos] = lexeme
            lexeme_data.lexemes_to_locations[old_lexeme].remove((line_ix, pos))

    lexeme_data.lexemes_to_freqs[winner.merged_lexeme] = winner.merge_token_count

    el1_freq = lexeme_data.lexemes_to_freqs[winner.bigram.el1]
    new_el1_freq = el1_freq - winner.merge_token_count
    lexeme_data.lexemes_to_freqs[winner.bigram.el1] = new_el1_freq

    el2_freq = lexeme_data.lexemes_to_freqs[winner.bigram.el2]
    new_el2_freq = el2_freq - winner.merge_token_count
    lexeme_data.lexemes_to_freqs[winner.bigram.el2] = new_el2_freq

    lexeme_data.lexemes_to_freqs = {
        k: v for k, v in lexeme_data.lexemes_to_freqs.items() if v != 0
    }
    lexeme_data.lexemes_to_locations = defaultdict(
        set, {k: v for k, v in lexeme_data.lexemes_to_locations.items() if v != set()}
    )
    return lexeme_data


@dataclass(frozen=True)
class BigramFreqArrays:
    # This would be a NamedTuple, but those don't support
    # cached_property
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
            l1 = bigram_data.left_lex_freqs[bigram.el1]
            el1_freq_array[i] = l1
            l2 = bigram_data.right_lex_freqs[bigram.el2]
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
    # obsA = data.bigram_freq_array
    # obsB = data.el1_freq_array
    # obsC = data.el2_freq_array
    # obsD = data.bigram_count - (obsA + obsB + obsC)
    obsA = data.bigram_freq_array
    obsB = data.el1_freq_array - obsA
    obsC = data.el2_freq_array - obsA
    obsD = data.bigram_count - obsA - obsB - obsC
    # http://ecologyandevolution.org/statsdocs/online-stats-manual-chapter4.html
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
        lexemes = merge_winner(winner, lexemes)
        bigrams = BigramData.from_lexemes(lexemes)
        if isinstance(iterations_iter, tqdm):
            iterations_iter.set_postfix({"last_winner": winner.merged_lexeme.word})
    return winners

# flake8: noqa
import json
from pathlib import Path

import pytest
from src.remerge import __version__, run
from src.remerge.core import Lexeme, SelectionMethod

from .fixtures import sample_corpus


def test_version():
    assert __version__ == "0.2.1"


def test_single_iter(sample_corpus):
    winners = run(sample_corpus, 1)
    assert winners[0].merged_lexeme == Lexeme(("you", "know"), 0)


def test_consecutive_single():
    """The point of this test is to make sure that the greedy bigram merge is occuring correctly
    and the algorithm does not try to merge the middle bigram here."""
    corpus = ["a a a a".split()]
    winners = run(corpus, 2)
    assert winners[0].merge_token_count == 2  # count == 3 is incorrect
    assert winners[1].merged_lexeme == Lexeme(("a", "a", "a", "a"), 0)


def test_consecutive_remainder():
    """The point of this test is to make sure that the greedy bigram merge is occuring correctly
    and the algorithm does not try to merge the trailing bigram"""
    corpus = ["c a b a b a b d".split()]
    winners = run(corpus, 2, method="frequency")
    assert winners[0].merge_token_count == 3
    assert winners[1].merge_token_count == 1


@pytest.mark.parametrize("progress_bar", ["all", "iterations", "none"])
def test_progress_bar(progress_bar, sample_corpus):
    winners = run(sample_corpus, 2, progress_bar=progress_bar)


@pytest.mark.parametrize(
    "method,min_count", [("log_likelihood", 0), ("npmi", 25), ("frequency", 0)]
)
def test_methods(method, min_count, sample_corpus):
    # we're not following the 'test one thing' principle here, as we're also including
    # min_count, but that should be tested with `npmi` anyway
    winners = run(sample_corpus, 2, method=method, min_count=min_count)


def test_output(sample_corpus, tmp_path):
    output = tmp_path / "tmp.json"
    winners = run(sample_corpus, 1, output=output)
    results = json.loads(output.read_text())
    # json keys are strings
    assert tuple(results["0"]) == winners[0].merged_lexeme.word

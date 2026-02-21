import pytest

from scripts.mwe_eval.metrics import prf_from_counts, prf_from_sets


@pytest.mark.fast
def test_prf_from_sets_basic_math():
    gold = {(0, 2), (3, 5)}
    predicted = {(0, 2), (9, 10)}
    prf = prf_from_sets(gold, predicted)

    assert prf.tp == 1
    assert prf.fp == 1
    assert prf.fn == 1
    assert prf.precision == 0.5
    assert prf.recall == 0.5
    assert prf.f1 == 0.5


@pytest.mark.fast
def test_prf_from_counts_handles_zero_denominators():
    prf = prf_from_counts(tp=0, fp=0, fn=0)
    assert prf.precision == 0.0
    assert prf.recall == 0.0
    assert prf.f1 == 0.0

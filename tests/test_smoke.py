import pytest

from remerge import run


@pytest.mark.fast
def test_minimal_run_smoke() -> None:
    winners = run(["a b a b"], 1, method="frequency")
    assert winners
    assert winners[0].merged_lexeme.word == ("a", "b")

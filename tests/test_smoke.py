from remerge import run


def test_minimal_run_smoke() -> None:
    winners = run([["a", "b", "a", "b"]], 1, method="frequency", progress_bar="none")
    assert winners
    assert winners[0].merged_lexeme.word == ("a", "b")

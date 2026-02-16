# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "remerge-mwe>=0.3.1",
# ]
# ///

from importlib.metadata import version

from remerge import run


def main() -> None:
    winners = run(["a b a b"], 1, method="frequency")
    assert winners, "Expected at least one winner from simple corpus."
    assert winners[0].merged_lexeme.word == ("a", "b")
    print("remerge-mwe version:", version("remerge-mwe"))
    print("first winner:", winners[0].merged_lexeme.word)
    print("ok")


if __name__ == "__main__":
    main()

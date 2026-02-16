from importlib.metadata import version

from .core import (
    Bigram,
    annotate,
    ExhaustionPolicy,
    Lexeme,
    NoCandidateBigramError,
    SelectionMethod,
    Splitter,
    WinnerInfo,
    run,
)

__version__ = version("remerge-mwe")

__all__ = [
    "annotate",
    "run",
    "Bigram",
    "Lexeme",
    "WinnerInfo",
    "SelectionMethod",
    "Splitter",
    "ExhaustionPolicy",
    "NoCandidateBigramError",
]

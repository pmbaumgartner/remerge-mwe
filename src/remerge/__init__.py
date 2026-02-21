from importlib.metadata import version

from .core import (
    Bigram,
    annotate,
    ConsensusRunSpec,
    ExhaustionPolicy,
    Lexeme,
    NoCandidateBigramError,
    SearchStrategy,
    SelectionMethod,
    Splitter,
    StopwordPolicy,
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
    "ConsensusRunSpec",
    "SelectionMethod",
    "SearchStrategy",
    "Splitter",
    "StopwordPolicy",
    "ExhaustionPolicy",
    "NoCandidateBigramError",
]

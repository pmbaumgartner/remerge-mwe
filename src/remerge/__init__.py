from .core import (
    annotate,
    ExhaustionPolicy,
    NoCandidateBigramError,
    SelectionMethod,
    Splitter,
    TieBreaker,
    run,
)

__version__ = "0.2.1"

__all__ = [
    "annotate",
    "run",
    "SelectionMethod",
    "Splitter",
    "TieBreaker",
    "ExhaustionPolicy",
    "NoCandidateBigramError",
]

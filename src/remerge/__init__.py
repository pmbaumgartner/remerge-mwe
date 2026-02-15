from .core import (
    ExhaustionPolicy,
    NoCandidateBigramError,
    SelectionMethod,
    TieBreaker,
    run,
)

__version__ = "0.2.1"

__all__ = [
    "run",
    "SelectionMethod",
    "TieBreaker",
    "ExhaustionPolicy",
    "NoCandidateBigramError",
]

from importlib.metadata import version

from .core import (
    annotate,
    ExhaustionPolicy,
    NoCandidateBigramError,
    SelectionMethod,
    Splitter,
    run,
)

__version__ = version("remerge-mwe")

__all__ = [
    "annotate",
    "run",
    "SelectionMethod",
    "Splitter",
    "ExhaustionPolicy",
    "NoCandidateBigramError",
]

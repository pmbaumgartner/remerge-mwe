from importlib.metadata import version

from .core import (
    Bigram,
    annotate,
    annotate_tagged,
    ExhaustionPolicy,
    from_conllu,
    Lexeme,
    MweOccurrence,
    NoCandidateBigramError,
    PosPattern,
    SelectionMethod,
    Splitter,
    TaggedDocument,
    TaggedToken,
    TaggedWinnerInfo,
    WinnerInfo,
    run,
    run_tagged,
)

__version__ = version("remerge-mwe")

__all__ = [
    "annotate",
    "annotate_tagged",
    "run",
    "run_tagged",
    "Bigram",
    "Lexeme",
    "WinnerInfo",
    "SelectionMethod",
    "Splitter",
    "ExhaustionPolicy",
    "from_conllu",
    "NoCandidateBigramError",
    "PosPattern",
    "TaggedDocument",
    "TaggedToken",
    "TaggedWinnerInfo",
    "MweOccurrence",
]

from typing import Optional

StepPayload = tuple[
    float,
    list[str],
    int,
    list[str],
    int,
    list[str],
    int,
    list[tuple[int, int]],
]

RunOutcome = tuple[str, list[StepPayload], Optional[float], int]
AnnotateRunOutcome = tuple[
    str,
    list[StepPayload],
    Optional[float],
    int,
    list[str],
    list[str],
]

class Engine:
    def __init__(
        self,
        corpus: list[str],
        method: str,
        min_count: int,
        tie_breaker: str,
        splitter: str = "delimiter",
        line_delimiter: str | None = None,
        sentencex_language: str = "en",
    ) -> None: ...
    def corpus_length(self) -> int: ...
    def run(
        self,
        iterations: int,
        min_score: Optional[float] = None,
    ) -> RunOutcome: ...
    def run_and_annotate(
        self,
        iterations: int,
        min_score: Optional[float] = None,
        mwe_prefix: str = "<mwe:",
        mwe_suffix: str = ">",
        token_separator: str = "_",
    ) -> AnnotateRunOutcome: ...

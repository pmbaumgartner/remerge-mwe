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

ProgressPayload = tuple[int, float, list[str]]
RunOutcome = tuple[str, list[StepPayload], Optional[float], int, list[ProgressPayload]]

class Engine:
    def __init__(
        self,
        corpus: list[str],
        method: str,
        min_count: int,
        tie_breaker: str,
        line_delimiter: str | None = None,
    ) -> None: ...
    def corpus_length(self) -> int: ...
    def run(
        self,
        iterations: int,
        min_score: Optional[float] = None,
        return_progress: bool = False,
    ) -> RunOutcome: ...

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
    list[tuple[int, int]],
]

StepOutcome = tuple[str, Optional[StepPayload], Optional[float]]


class Engine:
    def __init__(
        self,
        corpus: list[list[str]],
        method: str,
        min_count: int,
        tie_breaker: str,
    ) -> None: ...
    def corpus_length(self) -> int: ...
    def step(self, min_score: Optional[float] = None) -> StepOutcome: ...

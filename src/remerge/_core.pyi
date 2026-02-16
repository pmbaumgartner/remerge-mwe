STATUS_COMPLETED: int
STATUS_NO_CANDIDATE: int
STATUS_BELOW_MIN_SCORE: int

class StepResult:
    @property
    def score(self) -> float: ...
    @property
    def left_word(self) -> list[str]: ...
    @property
    def left_ix(self) -> int: ...
    @property
    def right_word(self) -> list[str]: ...
    @property
    def right_ix(self) -> int: ...
    @property
    def merged_word(self) -> list[str]: ...
    @property
    def merged_ix(self) -> int: ...
    @property
    def merge_token_count(self) -> int: ...

RunOutcome = tuple[int, list[StepResult], float | None, int]
AnnotateRunOutcome = tuple[
    int,
    list[StepResult],
    float | None,
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
        splitter: str = "delimiter",
        line_delimiter: str | None = "\n",
        sentencex_language: str = "en",
        rescore_interval: int = 25,
    ) -> None: ...
    def corpus_length(self) -> int: ...
    def run(
        self,
        iterations: int,
        min_score: float | None = None,
    ) -> RunOutcome: ...
    def run_and_annotate(
        self,
        iterations: int,
        min_score: float | None = None,
        mwe_prefix: str = "<mwe:",
        mwe_suffix: str = ">",
        token_separator: str = "_",
    ) -> AnnotateRunOutcome: ...

from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
import math
import sys
from typing import TypeVar, cast

from ._core import (
    Engine,
    STATUS_BELOW_MIN_SCORE,
    STATUS_COMPLETED,
    STATUS_NO_CANDIDATE,
    StepResult,
)


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"
    logdice = "logdice"
    t_score = "t_score"
    delta_p = "delta_p"


class Splitter(str, Enum):
    delimiter = "delimiter"
    sentencex = "sentencex"


class ExhaustionPolicy(str, Enum):
    stop = "stop"
    raise_ = "raise"


class StopwordPolicy(str, Enum):
    none = "none"
    block_stopword_stopword = "block_stopword_stopword"
    block_any_stopword = "block_any_stopword"


class SearchStrategy(str, Enum):
    greedy = "greedy"
    beam = "beam"


class NoCandidateBigramError(ValueError):
    """Raised when no candidate bigrams are available for selection."""


@dataclass(frozen=True, slots=True)
class Lexeme:
    word: tuple[str, ...]
    ix: int

    def __repr__(self) -> str:
        return f"({self.word}|{self.ix})"

    def __str__(self) -> str:
        return self.text

    @property
    def text(self) -> str:
        return " ".join(self.word)

    @property
    def token_count(self) -> int:
        return len(self.word)


Bigram = tuple[Lexeme, Lexeme]


@dataclass(frozen=True, slots=True)
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    score: float
    merge_token_count: int
    merge_segment_range: int

    def __str__(self) -> str:
        return self.text

    @property
    def text(self) -> str:
        return str(self.merged_lexeme)

    @property
    def token_count(self) -> int:
        return self.n_lexemes

    @property
    def n_lexemes(self) -> int:
        return len(self.merged_lexeme.word)


@dataclass(frozen=True, slots=True)
class ConsensusRunSpec:
    method: SelectionMethod | str | None = None
    min_count: int | None = None
    min_score: float | None = None
    rescore_interval: int | None = None
    stopword_policy: StopwordPolicy | str | None = None
    stopwords: list[str] | tuple[str, ...] | None = None
    block_punct_only: bool | None = None
    min_range: int | None = None
    range_alpha: float | None = None
    min_p_ab: float | None = None
    min_p_ba: float | None = None
    min_merge_count: int | None = None
    search_strategy: SearchStrategy | str | None = None
    beam_width: int | None = None
    beam_top_m: int | None = None


EnumType = TypeVar("EnumType", bound=Enum)


class _PhraseTrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, _PhraseTrieNode] = {}
        self.terminal = False


_CONSENSUS_OVERRIDE_KEYS = {field.name for field in fields(ConsensusRunSpec)}


def _coerce_enum(
    value: EnumType | str, enum_type: type[EnumType], argument_name: str
) -> EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        options = ", ".join(repr(option.value) for option in enum_type)
        raise ValueError(
            f"Invalid {argument_name} {value!r}. Expected one of: {options}."
        ) from exc


def _validate_probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive.")
    return value


def _validate_output_filters(
    *,
    min_winner_score_output: float | None,
    min_winner_range_output: int,
) -> None:
    if min_winner_score_output is not None and not math.isfinite(
        min_winner_score_output
    ):
        raise ValueError("min_winner_score_output must be finite.")
    if min_winner_range_output < 1:
        raise ValueError("min_winner_range_output must be greater than or equal to 1.")


def _winner_from_step_result(step_result: StepResult) -> WinnerInfo:
    return WinnerInfo(
        bigram=(
            Lexeme(tuple(step_result.left_word), step_result.left_ix),
            Lexeme(tuple(step_result.right_word), step_result.right_ix),
        ),
        merged_lexeme=Lexeme(tuple(step_result.merged_word), step_result.merged_ix),
        score=step_result.score,
        merge_token_count=step_result.merge_token_count,
        merge_segment_range=step_result.merge_segment_range,
    )


def _collect_winners(step_results: list[StepResult]) -> list[WinnerInfo]:
    return [_winner_from_step_result(step_result) for step_result in step_results]


def _winner_label(
    winner: WinnerInfo,
    *,
    mwe_prefix: str,
    mwe_suffix: str,
    token_separator: str,
) -> str | None:
    phrase = winner.merged_lexeme.word
    if len(phrase) < 2:
        return None
    return f"{mwe_prefix}{token_separator.join(phrase)}{mwe_suffix}"


def _labels_for_winners(
    winners: list[WinnerInfo],
    *,
    mwe_prefix: str,
    mwe_suffix: str,
    token_separator: str,
) -> set[str]:
    labels: set[str] = set()
    for winner in winners:
        label = _winner_label(
            winner,
            mwe_prefix=mwe_prefix,
            mwe_suffix=mwe_suffix,
            token_separator=token_separator,
        )
        if label is not None:
            labels.add(label)
    return labels


def _filter_labels_for_winners(
    labels: list[str],
    winners: list[WinnerInfo],
    *,
    mwe_prefix: str,
    mwe_suffix: str,
    token_separator: str,
) -> list[str]:
    allowed = _labels_for_winners(
        winners,
        mwe_prefix=mwe_prefix,
        mwe_suffix=mwe_suffix,
        token_separator=token_separator,
    )
    if not allowed:
        return []
    return [label for label in labels if label in allowed]


def _apply_winner_output_filters(
    winners: list[WinnerInfo],
    *,
    min_winner_score_output: float | None,
    min_winner_range_output: int,
) -> list[WinnerInfo]:
    filtered: list[WinnerInfo] = []
    for winner in winners:
        if (
            min_winner_score_output is not None
            and winner.score < min_winner_score_output
        ):
            continue
        if winner.merge_segment_range < min_winner_range_output:
            continue
        filtered.append(winner)
    return filtered


def _check_engine_status(
    status: int,
    *,
    selected_score: float | None,
    min_score: float | None,
    on_exhausted: ExhaustionPolicy,
    method: SelectionMethod,
    min_count: int,
) -> None:
    if status == STATUS_NO_CANDIDATE and on_exhausted is ExhaustionPolicy.raise_:
        raise NoCandidateBigramError(
            f"No candidate bigrams available for method={method.value!r} "
            f"and min_count={min_count}."
        )

    if status == STATUS_BELOW_MIN_SCORE and on_exhausted is ExhaustionPolicy.raise_:
        raise NoCandidateBigramError(
            f"Best candidate score ({selected_score}) is below min_score ({min_score})."
        )

    if status not in {STATUS_COMPLETED, STATUS_NO_CANDIDATE, STATUS_BELOW_MIN_SCORE}:
        raise RuntimeError(f"Unexpected engine status code {status!r}.")


def _make_engine(
    corpus: list[str],
    method: SelectionMethod | str,
    min_count: int,
    splitter: Splitter | str,
    line_delimiter: str | None,
    sentencex_language: str,
    rescore_interval: int,
    stopwords: list[str] | None,
    stopword_policy: StopwordPolicy | str,
    block_punct_only: bool,
    min_range: int,
    range_alpha: float,
    min_p_ab: float | None,
    min_p_ba: float | None,
    min_merge_count: int,
    search_strategy: SearchStrategy | str,
    beam_width: int,
    beam_top_m: int,
) -> tuple[Engine, SelectionMethod, Splitter, SearchStrategy]:
    method = _coerce_enum(method, SelectionMethod, "method")
    splitter = _coerce_enum(splitter, Splitter, "splitter")
    stopword_policy = _coerce_enum(stopword_policy, StopwordPolicy, "stopword_policy")
    search_strategy = _coerce_enum(search_strategy, SearchStrategy, "search_strategy")
    if min_count < 0:
        raise ValueError("min_count must be greater than or equal to 0.")
    if rescore_interval < 1:
        raise ValueError("rescore_interval must be greater than or equal to 1.")
    if splitter is Splitter.sentencex and not sentencex_language.strip():
        raise ValueError("sentencex_language must be a non-empty language code.")
    if stopwords is not None and not isinstance(stopwords, list):
        raise TypeError("stopwords must be a list[str] or None.")
    if stopwords is not None and any(not isinstance(token, str) for token in stopwords):
        raise TypeError("stopwords must be a list[str] or None.")
    if not isinstance(block_punct_only, bool):
        raise TypeError("block_punct_only must be a bool.")
    if min_range < 1:
        raise ValueError("min_range must be greater than or equal to 1.")
    if not math.isfinite(range_alpha):
        raise ValueError("range_alpha must be finite.")
    if range_alpha < 0:
        raise ValueError("range_alpha must be greater than or equal to 0.")

    min_p_ab = _validate_probability("min_p_ab", min_p_ab)
    min_p_ba = _validate_probability("min_p_ba", min_p_ba)

    if min_merge_count < 1:
        raise ValueError("min_merge_count must be greater than or equal to 1.")
    if beam_width < 1:
        raise ValueError("beam_width must be greater than or equal to 1.")
    if beam_top_m < 1:
        raise ValueError("beam_top_m must be greater than or equal to 1.")

    engine = Engine(
        corpus,
        method.value,
        min_count,
        splitter.value,
        line_delimiter,
        sentencex_language,
        rescore_interval,
        stopwords,
        stopword_policy.value,
        block_punct_only,
        min_range,
        range_alpha,
        min_p_ab,
        min_p_ba,
        min_merge_count,
        search_strategy.value,
        beam_width,
        beam_top_m,
    )
    return engine, method, splitter, search_strategy


def _run_core(
    corpus: list[str],
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    stopwords: list[str] | None = None,
    stopword_policy: StopwordPolicy | str = StopwordPolicy.none,
    block_punct_only: bool = False,
    min_range: int = 1,
    range_alpha: float = 0.0,
    min_p_ab: float | None = None,
    min_p_ba: float | None = None,
    min_merge_count: int = 1,
    search_strategy: SearchStrategy | str = SearchStrategy.greedy,
    beam_width: int = 1,
    beam_top_m: int = 8,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
) -> tuple[Engine, SelectionMethod, Splitter, SearchStrategy, ExhaustionPolicy]:
    engine, method, splitter, search_strategy = _make_engine(
        corpus,
        method,
        min_count,
        splitter,
        line_delimiter,
        sentencex_language,
        rescore_interval,
        stopwords,
        stopword_policy,
        block_punct_only,
        min_range,
        range_alpha,
        min_p_ab,
        min_p_ba,
        min_merge_count,
        search_strategy,
        beam_width,
        beam_top_m,
    )
    return (
        engine,
        method,
        splitter,
        search_strategy,
        _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted"),
    )


def _validate_progress_arg(progress: bool) -> None:
    if not isinstance(progress, bool):
        raise TypeError("progress must be a bool.")


def _render_progress(completed: int, requested: int) -> None:
    sys.stderr.write(f"\rremerge progress: {completed}/{requested}")
    sys.stderr.flush()


def _run_with_optional_progress(
    engine: Engine,
    *,
    iterations: int,
    min_score: float | None,
    progress: bool,
) -> tuple[int, list[StepResult], float | None, int]:
    if not progress:
        return engine.run(iterations, min_score)

    remaining = iterations
    completed = 0
    status = STATUS_COMPLETED
    selected_score = None
    all_steps = []
    corpus_length = engine.corpus_length()

    while remaining > 0:
        batch_size = 1
        status, step_results, selected_score, _corpus_length = engine.run(
            batch_size,
            min_score,
        )
        if step_results:
            all_steps.extend(step_results)
            completed += len(step_results)
            _render_progress(completed, iterations)
        remaining -= batch_size

        if status != STATUS_COMPLETED:
            if completed > 0:
                sys.stderr.write("\n")
            return status, all_steps, selected_score, corpus_length

    if completed > 0:
        sys.stderr.write("\n")
    return status, all_steps, selected_score, corpus_length


def _run_single_winners(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str,
    min_count: int,
    splitter: Splitter | str,
    line_delimiter: str | None,
    sentencex_language: str,
    rescore_interval: int,
    stopwords: list[str] | None,
    stopword_policy: StopwordPolicy | str,
    block_punct_only: bool,
    min_range: int,
    range_alpha: float,
    min_p_ab: float | None,
    min_p_ba: float | None,
    min_merge_count: int,
    search_strategy: SearchStrategy | str,
    beam_width: int,
    beam_top_m: int,
    on_exhausted: ExhaustionPolicy | str,
    min_score: float | None,
    progress: bool,
) -> tuple[list[WinnerInfo], SelectionMethod]:
    (
        engine,
        method_enum,
        _splitter,
        _search_strategy,
        on_exhausted_enum,
    ) = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
        stopwords=stopwords,
        stopword_policy=stopword_policy,
        block_punct_only=block_punct_only,
        min_range=min_range,
        range_alpha=range_alpha,
        min_p_ab=min_p_ab,
        min_p_ba=min_p_ba,
        min_merge_count=min_merge_count,
        search_strategy=search_strategy,
        beam_width=beam_width,
        beam_top_m=beam_top_m,
        on_exhausted=on_exhausted,
    )

    status, step_results, selected_score, _corpus_length = _run_with_optional_progress(
        engine,
        iterations=iterations,
        min_score=min_score,
        progress=progress,
    )
    _check_engine_status(
        status,
        selected_score=selected_score,
        min_score=min_score,
        on_exhausted=on_exhausted_enum,
        method=method_enum,
        min_count=min_count,
    )
    return _collect_winners(step_results), method_enum


def _consensus_spec_to_overrides(
    spec: ConsensusRunSpec | dict[str, object],
) -> dict[str, object]:
    if isinstance(spec, ConsensusRunSpec):
        overrides: dict[str, object] = {}
        for field in fields(ConsensusRunSpec):
            value = getattr(spec, field.name)
            if value is not None:
                overrides[field.name] = value
        return overrides
    if isinstance(spec, dict):
        extra = set(spec) - _CONSENSUS_OVERRIDE_KEYS
        if extra:
            allowed = ", ".join(sorted(_CONSENSUS_OVERRIDE_KEYS))
            raise ValueError(
                "Unsupported keys in consensus run spec: "
                f"{', '.join(sorted(extra))}. Allowed keys: {allowed}."
            )
        return dict(spec)
    raise TypeError(
        "consensus_runs entries must be ConsensusRunSpec or dict[str, object]."
    )


def _normalize_consensus_runs(
    consensus_runs: list[ConsensusRunSpec | dict[str, object]] | None,
) -> list[dict[str, object]]:
    if consensus_runs is None:
        return []
    if len(consensus_runs) == 0:
        raise ValueError(
            "consensus_runs must contain at least one run spec when provided."
        )

    normalized: list[dict[str, object]] = []
    for entry in consensus_runs:
        overrides = _consensus_spec_to_overrides(entry)
        if "stopwords" in overrides:
            stopwords = overrides["stopwords"]
            if stopwords is None:
                pass
            elif isinstance(stopwords, (list, tuple)):
                overrides["stopwords"] = [str(token) for token in stopwords]
            else:
                raise TypeError("consensus run stopwords must be list[str] or None.")
        normalized.append(overrides)
    return normalized


def _merge_consensus_kwargs(
    base_kwargs: dict[str, object], overrides: dict[str, object]
) -> dict[str, object]:
    merged = dict(base_kwargs)
    for key, value in overrides.items():
        merged[key] = value
    return merged


def _winner_phrase(winner: WinnerInfo) -> tuple[str, ...]:
    return winner.merged_lexeme.word


def _winner_sort_key_for_consensus(
    winner: WinnerInfo,
) -> tuple[float, int, int, tuple[str, ...]]:
    return (
        winner.score,
        winner.merge_token_count,
        winner.merge_segment_range,
        winner.merged_lexeme.word,
    )


def _run_consensus(
    corpus: list[str],
    iterations: int,
    *,
    base_kwargs: dict[str, object],
    consensus_runs: list[ConsensusRunSpec | dict[str, object]],
    consensus_min_run_support: int,
    consensus_min_method_support: int,
    progress: bool,
) -> list[WinnerInfo]:
    if consensus_min_run_support < 1:
        raise ValueError(
            "consensus_min_run_support must be greater than or equal to 1."
        )
    if consensus_min_method_support < 1:
        raise ValueError(
            "consensus_min_method_support must be greater than or equal to 1."
        )

    normalized_specs = _normalize_consensus_runs(consensus_runs)

    phrase_support_by_run: dict[tuple[str, ...], set[int]] = defaultdict(set)
    phrase_support_by_method: dict[tuple[str, ...], set[str]] = defaultdict(set)
    phrase_winners: dict[tuple[str, ...], list[WinnerInfo]] = defaultdict(list)

    for run_ix, spec in enumerate(normalized_specs):
        run_kwargs = _merge_consensus_kwargs(base_kwargs, spec)
        winners, method_enum = _run_single_winners(
            corpus,
            iterations,
            method=cast(SelectionMethod | str, run_kwargs["method"]),
            min_count=cast(int, run_kwargs["min_count"]),
            splitter=cast(Splitter | str, run_kwargs["splitter"]),
            line_delimiter=cast(str | None, run_kwargs["line_delimiter"]),
            sentencex_language=cast(str, run_kwargs["sentencex_language"]),
            rescore_interval=cast(int, run_kwargs["rescore_interval"]),
            stopwords=cast(list[str] | None, run_kwargs["stopwords"]),
            stopword_policy=cast(StopwordPolicy | str, run_kwargs["stopword_policy"]),
            block_punct_only=cast(bool, run_kwargs["block_punct_only"]),
            min_range=cast(int, run_kwargs["min_range"]),
            range_alpha=cast(float, run_kwargs["range_alpha"]),
            min_p_ab=cast(float | None, run_kwargs["min_p_ab"]),
            min_p_ba=cast(float | None, run_kwargs["min_p_ba"]),
            min_merge_count=cast(int, run_kwargs["min_merge_count"]),
            search_strategy=cast(SearchStrategy | str, run_kwargs["search_strategy"]),
            beam_width=cast(int, run_kwargs["beam_width"]),
            beam_top_m=cast(int, run_kwargs["beam_top_m"]),
            on_exhausted=cast(ExhaustionPolicy | str, run_kwargs["on_exhausted"]),
            min_score=cast(float | None, run_kwargs["min_score"]),
            progress=progress,
        )

        seen_in_run: set[tuple[str, ...]] = set()
        for winner in winners:
            phrase = _winner_phrase(winner)
            if len(phrase) < 2:
                continue
            seen_in_run.add(phrase)
            phrase_winners[phrase].append(winner)

        for phrase in seen_in_run:
            phrase_support_by_run[phrase].add(run_ix)
            phrase_support_by_method[phrase].add(method_enum.value)

    kept: list[tuple[tuple[str, ...], int, int, WinnerInfo]] = []
    for phrase, winners in phrase_winners.items():
        run_support = len(phrase_support_by_run[phrase])
        method_support = len(phrase_support_by_method[phrase])
        if run_support < consensus_min_run_support:
            continue
        if method_support < consensus_min_method_support:
            continue
        representative = max(winners, key=_winner_sort_key_for_consensus)
        kept.append((phrase, run_support, method_support, representative))

    kept.sort(
        key=lambda item: (
            -item[1],
            -item[2],
            -item[3].score,
            -item[3].merge_token_count,
            -item[3].merge_segment_range,
            item[0],
        )
    )
    return [item[3] for item in kept]


def _build_phrase_trie(phrases: set[tuple[str, ...]]) -> _PhraseTrieNode:
    root = _PhraseTrieNode()
    for phrase in phrases:
        if len(phrase) < 2:
            continue
        node = root
        for token in phrase:
            if token not in node.children:
                node.children[token] = _PhraseTrieNode()
            node = node.children[token]
        node.terminal = True
    return root


def _decode_longest_non_overlapping_spans(
    tokens: list[str],
    trie: _PhraseTrieNode,
) -> list[tuple[int, int]]:
    best_by_start: dict[int, int] = {}
    token_count = len(tokens)
    for start in range(token_count):
        node = trie
        end = start
        best_end: int | None = None
        while end < token_count:
            next_node = node.children.get(tokens[end])
            if next_node is None:
                break
            node = next_node
            end += 1
            if node.terminal:
                best_end = end
        if best_end is not None:
            best_by_start[start] = best_end

    out: list[tuple[int, int]] = []
    ix = 0
    while ix < token_count:
        end = best_by_start.get(ix)
        if end is None:
            ix += 1
            continue
        out.append((ix, end))
        ix = end
    return out


def _annotate_tokens_with_lexicon(
    tokens: list[str],
    *,
    trie: _PhraseTrieNode,
    mwe_prefix: str,
    mwe_suffix: str,
    token_separator: str,
) -> tuple[str, set[str]]:
    spans = _decode_longest_non_overlapping_spans(tokens, trie)
    span_map = {start: end for start, end in spans}

    output: list[str] = []
    labels: set[str] = set()

    ix = 0
    while ix < len(tokens):
        end = span_map.get(ix)
        if end is None:
            output.append(tokens[ix])
            ix += 1
            continue

        phrase = tokens[ix:end]
        label = f"{mwe_prefix}{token_separator.join(phrase)}{mwe_suffix}"
        labels.add(label)
        output.append(label)
        ix = end

    return " ".join(output), labels


def _normalize_corpus_documents_for_consensus(
    corpus: list[str],
    *,
    method: SelectionMethod | str,
    min_count: int,
    splitter: Splitter | str,
    line_delimiter: str | None,
    sentencex_language: str,
    rescore_interval: int,
    stopwords: list[str] | None,
    stopword_policy: StopwordPolicy | str,
    block_punct_only: bool,
    min_range: int,
    range_alpha: float,
    min_p_ab: float | None,
    min_p_ba: float | None,
    min_merge_count: int,
    search_strategy: SearchStrategy | str,
    beam_width: int,
    beam_top_m: int,
) -> list[str]:
    (
        engine,
        _method,
        _splitter,
        _search_strategy,
        _on_exhausted,
    ) = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
        stopwords=stopwords,
        stopword_policy=stopword_policy,
        block_punct_only=block_punct_only,
        min_range=min_range,
        range_alpha=range_alpha,
        min_p_ab=min_p_ab,
        min_p_ba=min_p_ba,
        min_merge_count=min_merge_count,
        search_strategy=search_strategy,
        beam_width=beam_width,
        beam_top_m=beam_top_m,
        on_exhausted=ExhaustionPolicy.stop,
    )
    (
        _status,
        _step_results,
        _selected_score,
        _corpus_length,
        annotated_docs,
        _labels,
    ) = engine.run_and_annotate(0, None, "<mwe:", ">", "_")
    return annotated_docs


def _annotate_with_consensus_lexicon(
    normalized_docs: list[str],
    *,
    splitter: Splitter | str,
    line_delimiter: str | None,
    lexicon: set[tuple[str, ...]],
    mwe_prefix: str,
    mwe_suffix: str,
    token_separator: str,
) -> tuple[list[str], list[str]]:
    splitter_enum = _coerce_enum(splitter, Splitter, "splitter")
    trie = _build_phrase_trie(lexicon)

    annotated_docs: list[str] = []
    labels: set[str] = set()

    joiner: str
    if splitter_enum is Splitter.delimiter:
        joiner = "" if line_delimiter is None else line_delimiter
    else:
        joiner = " "

    for doc in normalized_docs:
        if (
            splitter_enum is Splitter.delimiter
            and line_delimiter is not None
            and line_delimiter != ""
        ):
            segments = doc.split(line_delimiter)
        else:
            segments = [doc]

        annotated_segments: list[str] = []
        for segment in segments:
            tokens = segment.split()
            rendered, segment_labels = _annotate_tokens_with_lexicon(
                tokens,
                trie=trie,
                mwe_prefix=mwe_prefix,
                mwe_suffix=mwe_suffix,
                token_separator=token_separator,
            )
            annotated_segments.append(rendered)
            labels.update(segment_labels)

        annotated_docs.append(joiner.join(annotated_segments))

    return annotated_docs, sorted(labels)


def run(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    stopwords: list[str] | None = None,
    stopword_policy: StopwordPolicy | str = StopwordPolicy.none,
    block_punct_only: bool = False,
    min_range: int = 1,
    range_alpha: float = 0.0,
    min_p_ab: float | None = None,
    min_p_ba: float | None = None,
    min_merge_count: int = 1,
    min_winner_score_output: float | None = None,
    min_winner_range_output: int = 1,
    search_strategy: SearchStrategy | str = SearchStrategy.greedy,
    beam_width: int = 1,
    beam_top_m: int = 8,
    consensus_runs: list[ConsensusRunSpec | dict[str, object]] | None = None,
    consensus_min_run_support: int = 2,
    consensus_min_method_support: int = 1,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
) -> list[WinnerInfo]:
    """Run the remerge algorithm.

    ``min_score`` controls selection-time stopping and does not filter returned winners.
    ``min_winner_score_output`` and ``min_winner_range_output`` are output-only filters.
    """
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)
    _validate_output_filters(
        min_winner_score_output=min_winner_score_output,
        min_winner_range_output=min_winner_range_output,
    )

    base_kwargs: dict[str, object] = {
        "method": method,
        "min_count": min_count,
        "splitter": splitter,
        "line_delimiter": line_delimiter,
        "sentencex_language": sentencex_language,
        "rescore_interval": rescore_interval,
        "stopwords": stopwords,
        "stopword_policy": stopword_policy,
        "block_punct_only": block_punct_only,
        "min_range": min_range,
        "range_alpha": range_alpha,
        "min_p_ab": min_p_ab,
        "min_p_ba": min_p_ba,
        "min_merge_count": min_merge_count,
        "search_strategy": search_strategy,
        "beam_width": beam_width,
        "beam_top_m": beam_top_m,
        "on_exhausted": on_exhausted,
        "min_score": min_score,
    }

    if consensus_runs is None:
        winners, _method = _run_single_winners(
            corpus,
            iterations,
            method=method,
            min_count=min_count,
            splitter=splitter,
            line_delimiter=line_delimiter,
            sentencex_language=sentencex_language,
            rescore_interval=rescore_interval,
            stopwords=stopwords,
            stopword_policy=stopword_policy,
            block_punct_only=block_punct_only,
            min_range=min_range,
            range_alpha=range_alpha,
            min_p_ab=min_p_ab,
            min_p_ba=min_p_ba,
            min_merge_count=min_merge_count,
            search_strategy=search_strategy,
            beam_width=beam_width,
            beam_top_m=beam_top_m,
            on_exhausted=on_exhausted,
            min_score=min_score,
            progress=progress,
        )
    else:
        winners = _run_consensus(
            corpus,
            iterations,
            base_kwargs=base_kwargs,
            consensus_runs=consensus_runs,
            consensus_min_run_support=consensus_min_run_support,
            consensus_min_method_support=consensus_min_method_support,
            progress=progress,
        )

    return _apply_winner_output_filters(
        winners,
        min_winner_score_output=min_winner_score_output,
        min_winner_range_output=min_winner_range_output,
    )


def annotate(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    stopwords: list[str] | None = None,
    stopword_policy: StopwordPolicy | str = StopwordPolicy.none,
    block_punct_only: bool = False,
    min_range: int = 1,
    range_alpha: float = 0.0,
    min_p_ab: float | None = None,
    min_p_ba: float | None = None,
    min_merge_count: int = 1,
    min_winner_score_output: float | None = None,
    min_winner_range_output: int = 1,
    search_strategy: SearchStrategy | str = SearchStrategy.greedy,
    beam_width: int = 1,
    beam_top_m: int = 8,
    consensus_runs: list[ConsensusRunSpec | dict[str, object]] | None = None,
    consensus_min_run_support: int = 2,
    consensus_min_method_support: int = 1,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
    mwe_prefix: str = "<mwe:",
    mwe_suffix: str = ">",
    token_separator: str = "_",
) -> tuple[list[WinnerInfo], list[str], list[str]]:
    """Run remerge and return winners, annotated documents, and labels.

    Without ``consensus_runs``, ``annotated_docs`` comes from the merge trajectory.
    With ``consensus_runs``, annotation uses consensus lexicon rematching over the
    normalized original tokenization.
    """
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)
    _validate_output_filters(
        min_winner_score_output=min_winner_score_output,
        min_winner_range_output=min_winner_range_output,
    )

    if consensus_runs is None:
        (
            engine,
            method_enum,
            _splitter,
            _search_strategy,
            on_exhausted_enum,
        ) = _run_core(
            corpus,
            method=method,
            min_count=min_count,
            splitter=splitter,
            line_delimiter=line_delimiter,
            sentencex_language=sentencex_language,
            rescore_interval=rescore_interval,
            stopwords=stopwords,
            stopword_policy=stopword_policy,
            block_punct_only=block_punct_only,
            min_range=min_range,
            range_alpha=range_alpha,
            min_p_ab=min_p_ab,
            min_p_ba=min_p_ba,
            min_merge_count=min_merge_count,
            search_strategy=search_strategy,
            beam_width=beam_width,
            beam_top_m=beam_top_m,
            on_exhausted=on_exhausted,
        )

        if not progress:
            (
                status,
                step_results,
                selected_score,
                _corpus_length,
                annotated_docs,
                mwe_labels,
            ) = engine.run_and_annotate(
                iterations,
                min_score,
                mwe_prefix,
                mwe_suffix,
                token_separator,
            )
            _check_engine_status(
                status,
                selected_score=selected_score,
                min_score=min_score,
                on_exhausted=on_exhausted_enum,
                method=method_enum,
                min_count=min_count,
            )

            winners = _collect_winners(step_results)
            filtered_winners = _apply_winner_output_filters(
                winners,
                min_winner_score_output=min_winner_score_output,
                min_winner_range_output=min_winner_range_output,
            )
            filtered_labels = _filter_labels_for_winners(
                mwe_labels,
                filtered_winners,
                mwe_prefix=mwe_prefix,
                mwe_suffix=mwe_suffix,
                token_separator=token_separator,
            )
            return filtered_winners, annotated_docs, filtered_labels

        status, step_results, selected_score, _corpus_length = (
            _run_with_optional_progress(
                engine,
                iterations=iterations,
                min_score=min_score,
                progress=progress,
            )
        )
        _check_engine_status(
            status,
            selected_score=selected_score,
            min_score=min_score,
            on_exhausted=on_exhausted_enum,
            method=method_enum,
            min_count=min_count,
        )
        (
            _status,
            _unused_step_results,
            _unused_selected_score,
            _unused_corpus_length,
            annotated_docs,
            mwe_labels,
        ) = engine.run_and_annotate(
            0,
            None,
            mwe_prefix,
            mwe_suffix,
            token_separator,
        )
        winners = _collect_winners(step_results)
        filtered_winners = _apply_winner_output_filters(
            winners,
            min_winner_score_output=min_winner_score_output,
            min_winner_range_output=min_winner_range_output,
        )
        filtered_labels = _filter_labels_for_winners(
            mwe_labels,
            filtered_winners,
            mwe_prefix=mwe_prefix,
            mwe_suffix=mwe_suffix,
            token_separator=token_separator,
        )
        return filtered_winners, annotated_docs, filtered_labels

    base_kwargs: dict[str, object] = {
        "method": method,
        "min_count": min_count,
        "splitter": splitter,
        "line_delimiter": line_delimiter,
        "sentencex_language": sentencex_language,
        "rescore_interval": rescore_interval,
        "stopwords": stopwords,
        "stopword_policy": stopword_policy,
        "block_punct_only": block_punct_only,
        "min_range": min_range,
        "range_alpha": range_alpha,
        "min_p_ab": min_p_ab,
        "min_p_ba": min_p_ba,
        "min_merge_count": min_merge_count,
        "search_strategy": search_strategy,
        "beam_width": beam_width,
        "beam_top_m": beam_top_m,
        "on_exhausted": on_exhausted,
        "min_score": min_score,
    }

    consensus_winners = _run_consensus(
        corpus,
        iterations,
        base_kwargs=base_kwargs,
        consensus_runs=consensus_runs,
        consensus_min_run_support=consensus_min_run_support,
        consensus_min_method_support=consensus_min_method_support,
        progress=progress,
    )
    filtered_winners = _apply_winner_output_filters(
        consensus_winners,
        min_winner_score_output=min_winner_score_output,
        min_winner_range_output=min_winner_range_output,
    )

    lexicon = {
        winner.merged_lexeme.word
        for winner in filtered_winners
        if len(winner.merged_lexeme.word) >= 2
    }
    normalized_docs = _normalize_corpus_documents_for_consensus(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
        stopwords=stopwords,
        stopword_policy=stopword_policy,
        block_punct_only=block_punct_only,
        min_range=min_range,
        range_alpha=range_alpha,
        min_p_ab=min_p_ab,
        min_p_ba=min_p_ba,
        min_merge_count=min_merge_count,
        search_strategy=search_strategy,
        beam_width=beam_width,
        beam_top_m=beam_top_m,
    )
    annotated_docs, labels = _annotate_with_consensus_lexicon(
        normalized_docs,
        splitter=splitter,
        line_delimiter=line_delimiter,
        lexicon=lexicon,
        mwe_prefix=mwe_prefix,
        mwe_suffix=mwe_suffix,
        token_separator=token_separator,
    )

    return filtered_winners, annotated_docs, labels

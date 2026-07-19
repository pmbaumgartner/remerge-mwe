from dataclasses import dataclass
from enum import Enum
import sys
from typing import Literal, TypeAlias, TypeVar
import unicodedata

from ._core import (
    Engine,
    PosEngine,
    STATUS_BELOW_MIN_SCORE,
    STATUS_COMPLETED,
    STATUS_NO_CANDIDATE,
    StepResult,
)


class SelectionMethod(str, Enum):
    frequency = "frequency"
    log_likelihood = "log_likelihood"
    npmi = "npmi"


class Splitter(str, Enum):
    delimiter = "delimiter"
    sentencex = "sentencex"


class ExhaustionPolicy(str, Enum):
    stop = "stop"
    raise_ = "raise"


class NoCandidateBigramError(ValueError):
    """Raised when no candidate bigrams are available for selection."""


UPOS_TAGS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)


@dataclass(frozen=True, slots=True)
class TaggedToken:
    form: str
    upos: str


@dataclass(frozen=True, slots=True)
class TaggedDocument:
    sentences: tuple[tuple[TaggedToken, ...], ...]
    language: str = "en"
    source: Literal["supplied", "builtin"] = "supplied"
    model_id: str | None = None


PosPatternPosition: TypeAlias = str | frozenset[str]
PosPattern: TypeAlias = tuple[PosPatternPosition, ...]


def from_conllu(text: str) -> list[TaggedDocument]:
    """Parse strict CoNLL-U into canonical tagged documents."""
    if not isinstance(text, str):
        raise TypeError("text must be a str containing decoded UTF-8 CoNLL-U.")
    if not unicodedata.is_normalized("NFC", text):
        raise ValueError("CoNLL-U input must be NFC-normalized.")

    documents: list[list[tuple[TaggedToken, ...]]] = [[]]
    sentence: list[TaggedToken] = []
    expected_word_id = 1
    saw_content_in_document = False

    def finish_sentence(line_number: int) -> None:
        nonlocal sentence, expected_word_id, saw_content_in_document
        if not sentence:
            return
        if expected_word_id == 1:
            raise ValueError(
                f"CoNLL-U sentence ending at line {line_number} has no word rows."
            )
        documents[-1].append(tuple(sentence))
        sentence = []
        expected_word_id = 1
        saw_content_in_document = True

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            finish_sentence(line_number)
            continue
        if line.startswith("#"):
            if line.startswith("# newdoc id"):
                if sentence:
                    raise ValueError(
                        f"CoNLL-U # newdoc marker at line {line_number} must occur between sentences."
                    )
                if saw_content_in_document or documents[-1]:
                    documents.append([])
                    saw_content_in_document = False
            continue

        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError(
                f"CoNLL-U line {line_number} must contain exactly ten tab-separated fields."
            )
        token_id, form, _lemma, upos, _xpos, _feats, _head, _deprel, _deps, _misc = (
            fields
        )
        if "-" in token_id:
            bounds = token_id.split("-", maxsplit=1)
            if len(bounds) != 2 or not all(part.isdigit() for part in bounds):
                raise ValueError(
                    f"Invalid CoNLL-U multiword token ID at line {line_number}."
                )
            continue
        if "." in token_id:
            parts = token_id.split(".", maxsplit=1)
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(
                    f"Invalid CoNLL-U empty-node ID at line {line_number}."
                )
            continue
        if not token_id.isdigit() or int(token_id) != expected_word_id:
            raise ValueError(
                f"CoNLL-U word ID at line {line_number} must be consecutive; "
                f"expected {expected_word_id}."
            )
        if not form or form == "_" or any(character.isspace() for character in form):
            raise ValueError(f"Invalid CoNLL-U FORM at line {line_number}.")
        if upos not in UPOS_TAGS:
            raise ValueError(f"Invalid CoNLL-U UPOS at line {line_number}: {upos!r}.")
        sentence.append(TaggedToken(form, upos))
        expected_word_id += 1

    finish_sentence(len(text.splitlines()) + 1)
    return [TaggedDocument(sentences=tuple(sentences)) for sentences in documents]


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
class MweOccurrence:
    document_index: int
    sentence_index: int
    start_token: int
    end_token: int


@dataclass(frozen=True, slots=True)
class TaggedWinnerInfo(WinnerInfo):
    occurrences: tuple[MweOccurrence, ...]


EnumType = TypeVar("EnumType", bound=Enum)


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


def _winner_from_step_result(step_result: StepResult) -> WinnerInfo:
    return WinnerInfo(
        bigram=(
            Lexeme(tuple(step_result.left_word), step_result.left_ix),
            Lexeme(tuple(step_result.right_word), step_result.right_ix),
        ),
        merged_lexeme=Lexeme(tuple(step_result.merged_word), step_result.merged_ix),
        score=step_result.score,
        merge_token_count=step_result.merge_token_count,
    )


def _collect_winners(step_results: list[StepResult]) -> list[WinnerInfo]:
    return [_winner_from_step_result(step_result) for step_result in step_results]


def _tagged_winner_from_step_result(step_result: StepResult) -> TaggedWinnerInfo:
    winner = _winner_from_step_result(step_result)
    occurrences = tuple(
        MweOccurrence(*coordinate)
        for coordinate in zip(
            step_result.occurrence_documents,
            step_result.occurrence_sentences,
            step_result.occurrence_starts,
            step_result.occurrence_ends,
            strict=True,
        )
    )
    return TaggedWinnerInfo(
        bigram=winner.bigram,
        merged_lexeme=winner.merged_lexeme,
        score=winner.score,
        merge_token_count=winner.merge_token_count,
        occurrences=occurrences,
    )


def _collect_tagged_winners(
    step_results: list[StepResult],
) -> list[TaggedWinnerInfo]:
    return [
        _tagged_winner_from_step_result(step_result) for step_result in step_results
    ]


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
) -> tuple[Engine, SelectionMethod, Splitter]:
    method = _coerce_enum(method, SelectionMethod, "method")
    splitter = _coerce_enum(splitter, Splitter, "splitter")
    if min_count < 0:
        raise ValueError("min_count must be greater than or equal to 0.")
    if rescore_interval < 1:
        raise ValueError("rescore_interval must be greater than or equal to 1.")
    if splitter is Splitter.sentencex and not sentencex_language.strip():
        raise ValueError("sentencex_language must be a non-empty language code.")

    engine = Engine(
        corpus,
        method.value,
        min_count,
        splitter.value,
        line_delimiter,
        sentencex_language,
        rescore_interval,
    )
    return engine, method, splitter


def _run_core(
    corpus: list[str],
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
) -> tuple[Engine, SelectionMethod, ExhaustionPolicy]:
    engine, method, _splitter = _make_engine(
        corpus,
        method,
        min_count,
        splitter,
        line_delimiter,
        sentencex_language,
        rescore_interval,
    )
    return engine, method, _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")


def _validate_progress_arg(progress: bool) -> None:
    if not isinstance(progress, bool):
        raise TypeError("progress must be a bool.")


def _validate_tagged_corpus(
    corpus: list[TaggedDocument],
) -> list[list[list[tuple[str, str]]]]:
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a list of TaggedDocument values.")

    validated: list[list[list[tuple[str, str]]]] = []
    for document_index, document in enumerate(corpus):
        if not isinstance(document, TaggedDocument):
            raise TypeError(f"corpus[{document_index}] must be a TaggedDocument.")
        if document.language != "en":
            raise ValueError(f"corpus[{document_index}].language must be 'en'.")
        if document.source not in {"supplied", "builtin"}:
            raise ValueError(
                f"corpus[{document_index}].source must be 'supplied' or 'builtin'."
            )
        if document.model_id is not None and (
            not isinstance(document.model_id, str) or not document.model_id.strip()
        ):
            raise ValueError(
                f"corpus[{document_index}].model_id must be None or a non-empty string."
            )
        if document.source == "builtin" and document.model_id is None:
            raise ValueError(
                f"corpus[{document_index}].model_id is required for source='builtin'."
            )
        if not isinstance(document.sentences, tuple):
            raise TypeError(f"corpus[{document_index}].sentences must be a tuple.")

        validated_document: list[list[tuple[str, str]]] = []
        for sentence_index, sentence in enumerate(document.sentences):
            coordinate = f"corpus[{document_index}].sentences[{sentence_index}]"
            if not isinstance(sentence, tuple):
                raise TypeError(f"{coordinate} must be a tuple.")
            if not sentence:
                raise ValueError(f"{coordinate} must not be empty.")

            validated_sentence: list[tuple[str, str]] = []
            for token_index, token in enumerate(sentence):
                token_coordinate = f"{coordinate}[{token_index}]"
                if not isinstance(token, TaggedToken):
                    raise TypeError(f"{token_coordinate} must be a TaggedToken.")
                if not isinstance(token.form, str) or not token.form:
                    raise ValueError(
                        f"{token_coordinate}.form must be a non-empty string."
                    )
                if any(character.isspace() for character in token.form):
                    raise ValueError(
                        f"{token_coordinate}.form must not contain Unicode whitespace."
                    )
                if not unicodedata.is_normalized("NFC", token.form):
                    raise ValueError(f"{token_coordinate}.form must be NFC-normalized.")
                if token.upos not in UPOS_TAGS:
                    raise ValueError(
                        f"{token_coordinate}.upos must be one of the 17 Universal POS tags."
                    )
                validated_sentence.append((token.form, token.upos))
            validated_document.append(validated_sentence)
        validated.append(validated_document)
    return validated


def _normalize_pos_patterns(patterns: list[PosPattern]) -> list[list[list[str]]]:
    if not isinstance(patterns, list):
        raise TypeError("patterns must be a list of POS patterns.")
    if not patterns:
        raise ValueError("patterns must contain at least one pattern.")

    canonical_patterns: set[tuple[tuple[str, ...], ...]] = set()
    for pattern_index, pattern in enumerate(patterns):
        if not isinstance(pattern, tuple):
            raise TypeError(f"patterns[{pattern_index}] must be a tuple.")
        if len(pattern) < 2:
            raise ValueError(
                f"patterns[{pattern_index}] must contain at least two positions."
            )
        canonical_positions: list[tuple[str, ...]] = []
        for position_index, position in enumerate(pattern):
            coordinate = f"patterns[{pattern_index}][{position_index}]"
            if isinstance(position, str):
                if position != "*" and position not in UPOS_TAGS:
                    raise ValueError(f"{coordinate} is not a Universal POS tag or '*'.")
                alternatives = (position,)
            elif isinstance(position, frozenset):
                if not position:
                    raise ValueError(
                        f"{coordinate} must not be an empty alternative set."
                    )
                if not all(
                    isinstance(tag, str) and tag in UPOS_TAGS for tag in position
                ):
                    raise ValueError(
                        f"{coordinate} alternatives must be exact Universal POS tags."
                    )
                alternatives = tuple(sorted(position))
            else:
                raise TypeError(
                    f"{coordinate} must be a POS tag, '*', or frozenset of POS tags."
                )
            canonical_positions.append(alternatives)
        canonical_patterns.add(tuple(canonical_positions))

    return [
        [list(alternatives) for alternatives in pattern]
        for pattern in sorted(canonical_patterns)
    ]


def _make_pos_engine(
    corpus: list[TaggedDocument],
    patterns: list[PosPattern],
    method: SelectionMethod | str,
    min_count: int,
) -> tuple[PosEngine, SelectionMethod]:
    method = _coerce_enum(method, SelectionMethod, "method")
    if min_count < 0:
        raise ValueError("min_count must be greater than or equal to 0.")
    engine = PosEngine(
        _validate_tagged_corpus(corpus),
        _normalize_pos_patterns(patterns),
        method.value,
        min_count,
    )
    return engine, method


def _render_progress(completed: int, requested: int) -> None:
    sys.stderr.write(f"\rremerge progress: {completed}/{requested}")
    sys.stderr.flush()


def _run_with_optional_progress(
    engine: Engine | PosEngine,
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
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
) -> list[WinnerInfo]:
    """Run the remerge algorithm.

    The returned winners include:
    - ``score``: the candidate score used to select each winning bigram
      (frequency, log-likelihood, or NPMI depending on ``method``).
    - ``merge_token_count``: number of non-overlapping merge applications for
      that winner in the current iteration.
    """
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)

    engine, method, on_exhausted = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
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
        on_exhausted=on_exhausted,
        method=method,
        min_count=min_count,
    )
    return _collect_winners(step_results)


def run_with_occurrences(
    corpus: list[str],
    iterations: int,
    *,
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    splitter: Splitter | str = Splitter.delimiter,
    line_delimiter: str | None = "\n",
    sentencex_language: str = "en",
    rescore_interval: int = 25,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
) -> list[TaggedWinnerInfo]:
    """Run unfiltered discovery with original-token occurrence coordinates."""
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)
    engine, method, on_exhausted = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
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
        on_exhausted=on_exhausted,
        method=method,
        min_count=min_count,
    )
    return _collect_tagged_winners(step_results)


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
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
    mwe_prefix: str = "<mwe:",
    mwe_suffix: str = ">",
    token_separator: str = "_",
) -> tuple[list[WinnerInfo], list[str], list[str]]:
    """Run the remerge algorithm and annotate the merged corpus.

    ``annotate()`` uses the same winner payload as ``run()``.
    Output text is whitespace-normalized because tokenization is done with
    Rust ``split_whitespace()`` and reconstructed with single-space joins.
    """
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)

    engine, method, on_exhausted = _run_core(
        corpus,
        method=method,
        min_count=min_count,
        splitter=splitter,
        line_delimiter=line_delimiter,
        sentencex_language=sentencex_language,
        rescore_interval=rescore_interval,
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
            on_exhausted=on_exhausted,
            method=method,
            min_count=min_count,
        )
        return _collect_winners(step_results), annotated_docs, mwe_labels

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
        on_exhausted=on_exhausted,
        method=method,
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
    return _collect_winners(step_results), annotated_docs, mwe_labels


def run_tagged(
    corpus: list[TaggedDocument],
    iterations: int,
    *,
    patterns: list[PosPattern],
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
) -> list[TaggedWinnerInfo]:
    """Discover POS-constrained MWEs from occurrence-aligned supplied tags."""
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)
    engine, method = _make_pos_engine(corpus, patterns, method, min_count)
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")
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
        on_exhausted=on_exhausted,
        method=method,
        min_count=min_count,
    )
    return _collect_tagged_winners(step_results)


def annotate_tagged(
    corpus: list[TaggedDocument],
    iterations: int,
    *,
    patterns: list[PosPattern],
    method: SelectionMethod | str = SelectionMethod.log_likelihood,
    min_count: int = 0,
    on_exhausted: ExhaustionPolicy | str = ExhaustionPolicy.stop,
    min_score: float | None = None,
    progress: bool = False,
    mwe_prefix: str = "<mwe:",
    mwe_suffix: str = ">",
    token_separator: str = "_",
) -> tuple[list[TaggedWinnerInfo], list[str], list[str]]:
    """Discover and annotate POS-constrained MWEs from supplied tags."""
    if iterations < 0:
        raise ValueError("iterations must be greater than or equal to 0.")
    _validate_progress_arg(progress)
    engine, method = _make_pos_engine(corpus, patterns, method, min_count)
    on_exhausted = _coerce_enum(on_exhausted, ExhaustionPolicy, "on_exhausted")

    if not progress:
        status, steps, score, _length, documents, labels = engine.run_and_annotate(
            iterations,
            min_score,
            mwe_prefix,
            mwe_suffix,
            token_separator,
        )
    else:
        status, steps, score, _length = _run_with_optional_progress(
            engine,
            iterations=iterations,
            min_score=min_score,
            progress=True,
        )
        _status, _steps, _score, _length, documents, labels = engine.run_and_annotate(
            0,
            None,
            mwe_prefix,
            mwe_suffix,
            token_separator,
        )

    _check_engine_status(
        status,
        selected_score=score,
        min_score=min_score,
        on_exhausted=on_exhausted,
        method=method,
        min_count=min_count,
    )
    return _collect_tagged_winners(steps), documents, labels

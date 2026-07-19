"""Checksum-verified, offline-only loader for the frozen POS/MWE manifest.

This module intentionally has no network code.  Callers acquire the two upstream
repositories outside this checkout, at their pinned revisions, then pass the
containing directory to :func:`validate_acquired_dataset`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import cast
import unicodedata


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

MANIFEST_PATH = Path(__file__).with_name("manifest.json")


class ManifestError(ValueError):
    """The manifest or acquired data violates a frozen evaluation invariant."""


@dataclass(frozen=True)
class Token:
    document_id: str
    sentence_id: str
    token_index: int
    form: str
    upos: str
    domain: str


@dataclass(frozen=True)
class Sentence:
    document_id: str
    sentence_id: str
    tokens: tuple[Token, ...]

    @property
    def normalized_sequence(self) -> tuple[str, ...]:
        return tuple(token.form for token in self.tokens)


@dataclass(frozen=True)
class DatasetValidation:
    split_tokens: Mapping[str, int]
    final_oov_tokens: int
    final_ambiguous_tokens: int
    final_domain_tokens: Mapping[str, int]
    contiguous_strong_mwe_spans: int


@dataclass(frozen=True)
class MweSpan:
    """An adjudicated contiguous strong MWE, with an end-exclusive token span."""

    document_id: str
    sentence_id: str
    start_token: int
    end_token: int


@dataclass(frozen=True)
class FinalGold:
    """Protected final canonical tokens and adjudicated contiguous MWE spans."""

    sentences: tuple[Sentence, ...]
    mwe_spans: frozenset[MweSpan]


def load_manifest(path: Path | None = None) -> dict[str, object]:
    """Load and validate the project-owned, non-corpus manifest."""

    with (path or MANIFEST_PATH).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Validate manifest structure without reading any protected corpus file."""

    required = {
        "schema_version",
        "dataset_id",
        "language",
        "release_adequate",
        "sources",
        "splits",
        "named_domains",
        "mwe_gold",
        "filter_configuration",
        "protected_final",
    }
    missing = required - set(manifest)
    if missing:
        raise ManifestError(f"manifest missing required keys: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ManifestError("unsupported manifest schema version")
    if manifest["language"] != "en":
        raise ManifestError("v1 evaluation manifest must be English")
    if manifest["release_adequate"] is not True:
        raise ManifestError("manifest is not authorized for release evaluation")

    sources = _mapping(manifest["sources"], "sources")
    splits = _mapping(manifest["splits"], "splits")
    if set(splits) != {"train", "dev", "final"}:
        raise ManifestError("manifest must define exactly train, dev, and final splits")
    for name, split_value in splits.items():
        split = _mapping(split_value, f"splits.{name}")
        source = split.get("source")
        if source not in sources:
            raise ManifestError(f"splits.{name}.source is not a declared source")
        _sha256(split.get("sha256"), f"splits.{name}.sha256")
        if not isinstance(split.get("path"), str) or not split["path"]:
            raise ManifestError(f"splits.{name}.path must be a non-empty string")
        _int(split.get("expected_source_tokens"), f"splits.{name}.expected_source_tokens")
        if _int(split.get("expected_tokens"), f"splits.{name}.expected_tokens") <= 0:
            raise ManifestError(f"splits.{name}.expected_tokens must be positive")
    final_split = _mapping(splits["final"], "splits.final")
    for field in (
        "minimum_tokens",
        "minimum_oov_tokens",
        "minimum_ambiguous_tokens",
        "minimum_tokens_per_named_domain",
    ):
        if _int(final_split.get(field), f"splits.final.{field}") <= 0:
            raise ManifestError(f"splits.final.{field} must be positive")
    for name, source_value in sources.items():
        source = _mapping(source_value, f"sources.{name}")
        _sha256(source.get("license_sha256"), f"sources.{name}.license_sha256")
        if not isinstance(source.get("relative_root"), str):
            raise ManifestError(f"sources.{name}.relative_root must be a string")

    mwe = _mapping(manifest["mwe_gold"], "mwe_gold")
    if mwe.get("source") not in sources:
        raise ManifestError("mwe_gold.source is not a declared source")
    _sha256(mwe.get("sha256"), "mwe_gold.sha256")
    for field in ("minimum_in_scope_spans", "expected_contiguous_strong_spans", "expected_unretained_sentences"):
        if _int(mwe.get(field), f"mwe_gold.{field}") < 0:
            raise ManifestError(f"mwe_gold.{field} must not be negative")


def validate_acquired_dataset(manifest: Mapping[str, object], acquisition_root: Path) -> DatasetValidation:
    """Verify frozen hashes, split isolation, POS slices, and STREUSLE alignment.

    This is the only function that reads final gold data.  Call it from the
    protected evaluation harness, never from training or model-selection code.
    """

    validate_manifest(manifest)
    sources = _mapping(manifest["sources"], "sources")
    for name, source_value in sources.items():
        source = _mapping(source_value, f"sources.{name}")
        root = acquisition_root / _string(source["relative_root"], f"sources.{name}.relative_root")
        _verify_sha256(root / "LICENSE.txt", _string(source["license_sha256"], "license_sha256"))

    splits = _mapping(manifest["splits"], "splits")
    parsed_splits: dict[str, tuple[Sentence, ...]] = {}
    for split_name, split_value in splits.items():
        split = _mapping(split_value, f"splits.{split_name}")
        source = _mapping(sources[_string(split["source"], "split source")], "source")
        path = acquisition_root / _string(source["relative_root"], "source root") / _string(split["path"], "split path")
        _verify_sha256(path, _string(split["sha256"], "split hash"))
        sentences = tuple(read_conllu(path))
        source_count = sum(len(sentence.tokens) for sentence in sentences)
        expected_source_tokens = _int(
            split["expected_source_tokens"], f"splits.{split_name}.expected_source_tokens"
        )
        if source_count != expected_source_tokens:
            raise ManifestError(f"{split_name} source token count {source_count} != frozen {expected_source_tokens}")
        parsed_splits[split_name] = sentences

    parsed_splits = _deduplicate_splits(parsed_splits)
    for split_name, sentences in parsed_splits.items():
        expected_tokens = _int(
            _mapping(splits[split_name], f"splits.{split_name}")["expected_tokens"],
            f"splits.{split_name}.expected_tokens",
        )
        count = sum(len(sentence.tokens) for sentence in sentences)
        if count != expected_tokens:
            raise ManifestError(f"{split_name} retained token count {count} != frozen {expected_tokens}")
    _assert_split_isolation(parsed_splits)
    final_tokens = tuple(token for sentence in parsed_splits["final"] for token in sentence.tokens)
    train_tokens = tuple(token for sentence in parsed_splits["train"] for token in sentence.tokens)
    training_forms = {token.form for token in train_tokens}
    training_tags = defaultdict(set)
    for token in train_tokens:
        training_tags[token.form].add(token.upos)
    oov = sum(token.form not in training_forms for token in final_tokens)
    ambiguous = sum(len(training_tags[token.form]) >= 2 for token in final_tokens)
    domains = Counter(token.domain for token in final_tokens)

    final_spec = _mapping(splits["final"], "splits.final")
    if len(final_tokens) < _int(final_spec["minimum_tokens"], "splits.final.minimum_tokens"):
        raise ManifestError("final split does not meet its frozen token floor")
    if oov < _int(final_spec["minimum_oov_tokens"], "splits.final.minimum_oov_tokens"):
        raise ManifestError("final split does not meet its frozen OOV floor")
    if ambiguous < _int(final_spec["minimum_ambiguous_tokens"], "splits.final.minimum_ambiguous_tokens"):
        raise ManifestError("final split does not meet its frozen ambiguity floor")
    for domain, expected_count_value in _mapping(manifest["named_domains"], "named_domains").items():
        expected_count = _int(expected_count_value, f"named_domains.{domain}")
        if domains[domain] != expected_count:
            raise ManifestError(f"final domain {domain!r} has {domains[domain]} tokens, expected {expected_count}")
        if domains[domain] < _int(
            final_spec["minimum_tokens_per_named_domain"], "splits.final.minimum_tokens_per_named_domain"
        ):
            raise ManifestError(f"final domain {domain!r} falls below the frozen adequacy floor")

    mwe_spans = _load_final_mwe_spans(manifest, acquisition_root, parsed_splits["final"])
    return DatasetValidation(
        split_tokens={name: sum(len(sentence.tokens) for sentence in sentences) for name, sentences in parsed_splits.items()},
        final_oov_tokens=oov,
        final_ambiguous_tokens=ambiguous,
        final_domain_tokens=dict(domains),
        contiguous_strong_mwe_spans=len(mwe_spans),
    )


def load_tagged_split(
    manifest: Mapping[str, object], acquisition_root: Path, split_name: str, *, allow_final: bool = False
) -> tuple[Sentence, ...]:
    """Load one checksum-verified canonical split without retokenizing it."""

    validate_manifest(manifest)
    if split_name not in {"train", "dev", "final"}:
        raise ManifestError(f"unknown split {split_name!r}")
    if split_name == "final" and not allow_final:
        raise ManifestError("final split requires explicit protected-harness authorization")
    splits = _mapping(manifest["splits"], "splits")
    sources = _mapping(manifest["sources"], "sources")
    split_order = ("train", "dev", "final")
    raw_splits: dict[str, tuple[Sentence, ...]] = {}
    for name in split_order[: split_order.index(split_name) + 1]:
        split = _mapping(splits[name], f"splits.{name}")
        source = _mapping(sources[_string(split["source"], "split source")], "split source")
        path = acquisition_root / _string(source["relative_root"], "source root") / _string(split["path"], "split path")
        _verify_sha256(path, _string(split["sha256"], "split hash"))
        raw_splits[name] = tuple(read_conllu(path))
    sentences = _deduplicate_splits(raw_splits)[split_name]
    expected = _int(_mapping(splits[split_name], f"splits.{split_name}")["expected_tokens"], "expected_tokens")
    if sum(len(sentence.tokens) for sentence in sentences) != expected:
        raise ManifestError(f"{split_name} retained token count does not match the frozen manifest")
    return sentences


def load_final_gold(manifest: Mapping[str, object], acquisition_root: Path, *, allow_final: bool = False) -> FinalGold:
    """Load protected final units after every frozen adequacy check has passed."""

    if not allow_final:
        raise ManifestError("final gold requires explicit protected-harness authorization")
    validate_acquired_dataset(manifest, acquisition_root)
    sentences = load_tagged_split(manifest, acquisition_root, "final", allow_final=True)
    return FinalGold(sentences=sentences, mwe_spans=_load_final_mwe_spans(manifest, acquisition_root, sentences))


def read_conllu(path: Path) -> Iterable[Sentence]:
    """Yield canonical integer-word sentences from a strict CoNLL-U source."""

    document_id: str | None = None
    sentence_id: str | None = None
    rows: list[Token] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines() + [""], start=1):
        if raw_line.startswith("# newdoc id = "):
            document_id = raw_line.removeprefix("# newdoc id = ").strip()
            if not document_id:
                raise ManifestError(f"{path}:{line_number}: empty document ID")
            continue
        if raw_line.startswith("# sent_id = "):
            sentence_id = raw_line.removeprefix("# sent_id = ").strip()
            if not sentence_id:
                raise ManifestError(f"{path}:{line_number}: empty sentence ID")
            continue
        if raw_line:
            if raw_line.startswith("#"):
                continue
            fields = raw_line.split("\t")
            if len(fields) != 10:
                raise ManifestError(f"{path}:{line_number}: expected 10 CoNLL-U fields")
            if not fields[0].isdigit():
                continue
            if document_id is None or sentence_id is None:
                raise ManifestError(f"{path}:{line_number}: word row lacks document or sentence ID")
            form, upos = fields[1], fields[3]
            if not form or any(character.isspace() for character in form):
                raise ManifestError(f"{path}:{line_number}: invalid canonical FORM")
            if unicodedata.normalize("NFC", form) != form:
                raise ManifestError(f"{path}:{line_number}: FORM is not NFC")
            if upos not in UPOS_TAGS:
                raise ManifestError(f"{path}:{line_number}: invalid UPOS {upos!r}")
            rows.append(Token(document_id, sentence_id, len(rows), form, upos, _domain(document_id)))
            continue
        if rows:
            yield Sentence(rows[0].document_id, rows[0].sentence_id, tuple(rows))
            rows = []
        sentence_id = None
    if rows:
        raise AssertionError("sentinel blank line should have emitted the last sentence")


def _load_final_mwe_spans(
    manifest: Mapping[str, object], acquisition_root: Path, final_sentences: tuple[Sentence, ...]
) -> frozenset[MweSpan]:
    sources = _mapping(manifest["sources"], "sources")
    mwe = _mapping(manifest["mwe_gold"], "mwe_gold")
    source = _mapping(sources[_string(mwe["source"], "mwe source")], "mwe source")
    path = acquisition_root / _string(source["relative_root"], "mwe source root") / _string(mwe["path"], "mwe path")
    _verify_sha256(path, _string(mwe["sha256"], "mwe hash"))
    final_by_sentence = {sentence.sentence_id: sentence for sentence in final_sentences}
    spans: set[MweSpan] = set()
    unretained_sentences = 0
    for sentence_id, tokens, strong_mwes in _read_streusle(path):
        expected = final_by_sentence.get(sentence_id)
        if expected is None:
            unretained_sentences += 1
            continue
        observed = tuple((token.form, token.upos) for token in expected.tokens)
        if observed != tokens:
            raise ManifestError(f"STREUSLE sentence {sentence_id!r} does not canonically align to EWT")
        for indices in strong_mwes.values():
            ordered = tuple(sorted(indices))
            if len(ordered) < 2 or ordered != tuple(range(ordered[0], ordered[-1] + 1)):
                continue
            spans.add(MweSpan(expected.document_id, sentence_id, ordered[0], ordered[-1] + 1))
    expected_spans = _int(mwe["expected_contiguous_strong_spans"], "mwe_gold.expected_contiguous_strong_spans")
    if len(spans) != expected_spans:
        raise ManifestError(f"contiguous strong MWE span count {len(spans)} != frozen {expected_spans}")
    if len(spans) < _int(mwe["minimum_in_scope_spans"], "mwe_gold.minimum_in_scope_spans"):
        raise ManifestError("MWE gold span count is below the frozen adequacy floor")
    if unretained_sentences != _int(mwe["expected_unretained_sentences"], "mwe_gold.expected_unretained_sentences"):
        raise ManifestError(
            f"STREUSLE unretained sentence count {unretained_sentences} != frozen {mwe['expected_unretained_sentences']}"
        )
    return frozenset(spans)


def _read_streusle(path: Path) -> Iterable[tuple[str, tuple[tuple[str, str], ...], Mapping[str, set[int]]]]:
    sentence_id: str | None = None
    tokens: list[tuple[str, str]] = []
    strong_mwes: dict[str, set[int]] = defaultdict(set)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines() + [""], start=1):
        if raw_line.startswith("# sent_id = "):
            sentence_id = raw_line.removeprefix("# sent_id = ").strip()
            continue
        if raw_line:
            if raw_line.startswith("#"):
                continue
            fields = raw_line.split("\t")
            if len(fields) < 11:
                raise ManifestError(f"{path}:{line_number}: malformed CONLLULEX row")
            if fields[0].isdigit():
                tokens.append((fields[1], fields[3]))
                for member in fields[10].split(";"):
                    if member == "_":
                        continue
                    try:
                        group, index = member.split(":", 1)
                        strong_mwes[group].add(int(index) - 1)
                    except ValueError as exc:
                        raise ManifestError(f"{path}:{line_number}: malformed strong MWE membership") from exc
            continue
        if sentence_id is not None:
            yield sentence_id, tuple(tokens), strong_mwes
        sentence_id = None
        tokens = []
        strong_mwes = defaultdict(set)


def _assert_split_isolation(parsed_splits: Mapping[str, tuple[Sentence, ...]]) -> None:
    document_owner: dict[str, str] = {}
    sequence_owner: dict[tuple[str, ...], str] = {}
    for split, sentences in parsed_splits.items():
        for sentence in sentences:
            existing_document = document_owner.setdefault(sentence.document_id, split)
            if existing_document != split:
                raise ManifestError(f"document {sentence.document_id!r} crosses {existing_document}/{split}")
            sequence = sentence.normalized_sequence
            existing_sequence = sequence_owner.setdefault(sequence, split)
            if existing_sequence != split:
                raise ManifestError(f"normalized token sequence crosses {existing_sequence}/{split}: {sequence!r}")


def _deduplicate_splits(parsed_splits: Mapping[str, tuple[Sentence, ...]]) -> dict[str, tuple[Sentence, ...]]:
    """Retain each normalized sentence once, with frozen train/dev/final precedence."""

    retained: dict[str, tuple[Sentence, ...]] = {}
    seen: set[tuple[str, ...]] = set()
    for split in ("train", "dev", "final"):
        if split not in parsed_splits:
            continue
        output: list[Sentence] = []
        for sentence in parsed_splits[split]:
            if sentence.normalized_sequence not in seen:
                seen.add(sentence.normalized_sequence)
                output.append(sentence)
        retained[split] = tuple(output)
    return retained


def _verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ManifestError(f"required offline acquisition file is missing: {path}")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ManifestError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _domain(document_id: str) -> str:
    return document_id.split("-", 1)[0]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{name} must be a string")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ManifestError(f"{name} must be an integer")
    return value


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ManifestError(f"{name} must be a lowercase SHA-256 digest")

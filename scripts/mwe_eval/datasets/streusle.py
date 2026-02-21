from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.request import urlretrieve
import zipfile

from .base import GoldMention, Sentence, SplitData

DATASET_NAME = "streusle"
_STREUSLE_TAG = "v5.0"
_STREUSLE_URL = (
    f"https://github.com/nert-nlp/streusle/archive/refs/tags/{_STREUSLE_TAG}.zip"
)


def _streusle_root(data_root: Path) -> Path:
    return data_root / "streusle"


def _coerce_token(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("word", "text", "token", "form"):
            token = value.get(key)
            if isinstance(token, str):
                return token
    return str(value)


def _extract_tokens(record: dict[str, Any]) -> tuple[str, ...]:
    if "tokens" in record and isinstance(record["tokens"], list):
        return tuple(_coerce_token(token) for token in record["tokens"])

    if "toks" in record and isinstance(record["toks"], list):
        return tuple(_coerce_token(token) for token in record["toks"])

    if isinstance(record.get("text"), str):
        return tuple(record["text"].split())

    raise ValueError(
        f"Unable to extract tokens from record keys={sorted(record.keys())}."
    )


def _normalize_indices(
    raw_indices: Any,
    *,
    token_count: int,
    assume_one_based: bool,
) -> tuple[int, ...]:
    if not isinstance(raw_indices, (list, tuple)):
        return ()
    indices: list[int] = []
    for item in raw_indices:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        indices.append(index)

    if assume_one_based:
        indices = [index - 1 for index in indices]

    normalized = tuple(
        sorted(set(index for index in indices if 0 <= index < token_count))
    )
    return normalized


def _mentions_from_bio(tags: list[str]) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    start: int | None = None
    label: str | None = None

    def flush(end: int) -> None:
        nonlocal start, label
        if start is None:
            return
        if end - start >= 2:
            mentions.append(GoldMention(tuple(range(start, end)), label))
        start = None
        label = None

    for idx, tag in enumerate(tags):
        raw = tag.strip()
        if not raw or raw == "O":
            flush(idx)
            continue

        if "-" in raw:
            prefix, raw_label = raw.split("-", 1)
        else:
            prefix, raw_label = raw, None
        prefix = prefix.upper()

        if prefix == "B":
            flush(idx)
            start = idx
            label = raw_label
        elif prefix == "I":
            if start is None:
                start = idx
                label = raw_label
        else:
            flush(idx)
            start = idx
            label = raw_label

    flush(len(tags))
    return mentions


def _mentions_from_container(container: Any, token_count: int) -> list[GoldMention]:
    mentions: list[GoldMention] = []

    iterable: Iterable[Any]
    if isinstance(container, dict):
        iterable = container.values()
    elif isinstance(container, list):
        iterable = container
    else:
        return mentions

    for item in iterable:
        label: str | None = None
        indices: tuple[int, ...] = ()
        if isinstance(item, dict):
            label_raw = (
                item.get("label")
                or item.get("type")
                or item.get("lexcat")
                or item.get("mwe_type")
            )
            if isinstance(label_raw, str):
                label = label_raw

            if "toknums" in item:
                indices = _normalize_indices(
                    item.get("toknums"),
                    token_count=token_count,
                    assume_one_based=True,
                )
            elif "token_indices" in item:
                indices = _normalize_indices(
                    item.get("token_indices"),
                    token_count=token_count,
                    assume_one_based=False,
                )
            elif "indices" in item:
                indices = _normalize_indices(
                    item.get("indices"),
                    token_count=token_count,
                    assume_one_based=False,
                )
            elif "tokens" in item:
                indices = _normalize_indices(
                    item.get("tokens"),
                    token_count=token_count,
                    assume_one_based=False,
                )
        elif isinstance(item, (list, tuple)):
            indices = _normalize_indices(
                item,
                token_count=token_count,
                assume_one_based=False,
            )

        if len(indices) >= 2:
            mentions.append(GoldMention(indices, label))

    return mentions


def _extract_mentions(
    record: dict[str, Any], token_count: int
) -> tuple[GoldMention, ...]:
    mentions: list[GoldMention] = []
    for key in ("mwes", "smwes", "wmwes"):
        if key in record:
            mentions.extend(_mentions_from_container(record[key], token_count))

    if mentions:
        deduped = dict.fromkeys(mentions)
        return tuple(deduped)

    for key in ("bio_tags", "mwe_tags", "tags"):
        raw = record.get(key)
        if isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
            return tuple(_mentions_from_bio(raw))

    return ()


def _record_to_sentence(record: dict[str, Any], index: int, source: str) -> Sentence:
    tokens = _extract_tokens(record)
    mentions = _extract_mentions(record, token_count=len(tokens))
    sent_id_raw = record.get("sent_id") or record.get("id") or f"{source}:{index}"
    return Sentence(sent_id=str(sent_id_raw), tokens=tokens, mentions=mentions)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        return records

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("sentences", "data", "examples"):
            maybe = payload.get(key)
            if isinstance(maybe, list):
                return [item for item in maybe if isinstance(item, dict)]

        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [item for item in value if isinstance(item, dict)]

    raise ValueError(f"Unrecognized JSON shape in {path}.")


def _discover_split_file(root: Path, split: str) -> Path:
    split_lower = split.lower()
    candidates = [
        path
        for path in root.glob("**/*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and split_lower in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No STREUSLE file found for split={split!r} under {root}."
        )

    def score(path: Path) -> tuple[int, int, int, str]:
        name = path.name.lower()
        exact = int(path.stem.lower() == split_lower)
        contains = int(split_lower in name)
        depth = len(path.parts)
        return (-exact, -contains, depth, str(path))

    return sorted(candidates, key=score)[0]


def _find_extracted_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("streusle-")
    )


def download(data_root: Path, *, force: bool = False) -> Path:
    root = _streusle_root(data_root)
    archive = root / f"streusle-{_STREUSLE_TAG}.zip"
    extracted = _find_extracted_dirs(root)

    if extracted and not force:
        return extracted[0]

    root.mkdir(parents=True, exist_ok=True)

    if force:
        for directory in extracted:
            shutil.rmtree(directory)
        if archive.exists():
            archive.unlink()

    if not archive.exists():
        urlretrieve(_STREUSLE_URL, archive)

    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(root)

    extracted = _find_extracted_dirs(root)
    if not extracted:
        raise RuntimeError(
            f"Download succeeded but no extracted STREUSLE directory found in {root}."
        )
    return extracted[0]


def load_split(data_root: Path, split: str, **_: object) -> SplitData:
    root = _streusle_root(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"STREUSLE data not found at {root}. Run download first."
        )

    split_file = _discover_split_file(root, split)
    records = _load_json_records(split_file)
    sentences = tuple(
        _record_to_sentence(record, idx, split_file.stem)
        for idx, record in enumerate(records)
    )
    return SplitData(dataset=DATASET_NAME, split=split, sentences=sentences)

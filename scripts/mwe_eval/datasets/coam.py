from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
from typing import Any

from .base import GoldMention, Sentence, SplitData

DATASET_NAME = "coam"
_COAM_DATASET_ID = "yusuke196/CoAM"


def _coam_root(data_root: Path) -> Path:
    return data_root / "coam"


def _saved_dataset_dir(data_root: Path) -> Path:
    return _coam_root(data_root) / "hf_saved"


def _load_dataset_module() -> Any:
    try:
        return importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CoAM support requires the optional 'datasets' dependency. "
            "Install eval dependencies first."
        ) from exc


def download(data_root: Path, *, force: bool = False) -> Path:
    root = _coam_root(data_root)
    saved = _saved_dataset_dir(data_root)

    if saved.exists() and not force:
        return saved

    if force and saved.exists():
        shutil.rmtree(saved)

    root.mkdir(parents=True, exist_ok=True)

    datasets = _load_dataset_module()
    try:
        dataset_dict = datasets.load_dataset(_COAM_DATASET_ID)
        dataset_dict.save_to_disk(str(saved))
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "401" in message or "403" in message or "gated" in message.lower():
            raise RuntimeError(
                "Unable to access CoAM. Run 'huggingface-cli login' and accept "
                "dataset access terms on the CoAM dataset page, then retry."
            ) from exc
        raise

    return saved


def _extract_tokens(row: dict[str, Any]) -> tuple[str, ...]:
    for key in ("tokens", "words", "text_tokens"):
        value = row.get(key)
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
        ):
            return tuple(value)

    if isinstance(row.get("text"), str):
        return tuple(row["text"].split())

    raise ValueError(
        f"Unable to extract tokens from CoAM row keys={sorted(row.keys())}."
    )


def _mentions_from_spans(
    spans: Any,
    labels: Any,
    *,
    token_count: int,
) -> list[GoldMention]:
    if not isinstance(spans, list):
        return []

    out: list[GoldMention] = []
    for index, span in enumerate(spans):
        label: str | None = None
        if (
            isinstance(labels, list)
            and index < len(labels)
            and isinstance(labels[index], str)
        ):
            label = labels[index]

        if isinstance(span, list) and len(span) == 2:
            start, end = span
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= token_count
            ):
                indices = tuple(range(start, end))
                if len(indices) >= 2:
                    out.append(GoldMention(indices, label))

    return out


def _mentions_from_generic_mwes(raw: Any, token_count: int) -> list[GoldMention]:
    if not isinstance(raw, list):
        return []

    out: list[GoldMention] = []
    for item in raw:
        if isinstance(item, dict):
            indices_raw = (
                item.get("token_indices") or item.get("indices") or item.get("tokens")
            )
            label_raw = item.get("label") or item.get("type") or item.get("mwe_type")
            if not isinstance(indices_raw, list):
                continue
            indices: list[int] = []
            for token_index in indices_raw:
                if isinstance(token_index, int) and 0 <= token_index < token_count:
                    indices.append(token_index)
            normalized = tuple(sorted(set(indices)))
            if len(normalized) < 2:
                continue
            label = label_raw if isinstance(label_raw, str) else None
            out.append(GoldMention(normalized, label))
    return out


def _mentions_from_bio(tags: list[str]) -> list[GoldMention]:
    out: list[GoldMention] = []
    start: int | None = None
    label: str | None = None

    def flush(end: int) -> None:
        nonlocal start, label
        if start is None:
            return
        if end - start >= 2:
            out.append(GoldMention(tuple(range(start, end)), label))
        start = None
        label = None

    for idx, tag in enumerate(tags):
        value = tag.strip()
        if value == "" or value == "O":
            flush(idx)
            continue

        if "-" in value:
            prefix, cur_label = value.split("-", 1)
        else:
            prefix, cur_label = value, None
        prefix = prefix.upper()

        if prefix == "B":
            flush(idx)
            start = idx
            label = cur_label
        elif prefix == "I":
            if start is None:
                start = idx
                label = cur_label
        else:
            flush(idx)
            start = idx
            label = cur_label

    flush(len(tags))
    return out


def _extract_mentions(row: dict[str, Any], token_count: int) -> tuple[GoldMention, ...]:
    mentions: list[GoldMention] = []

    mentions.extend(
        _mentions_from_spans(
            row.get("mwe_spans") or row.get("spans"),
            row.get("mwe_labels") or row.get("labels"),
            token_count=token_count,
        )
    )
    mentions.extend(_mentions_from_generic_mwes(row.get("mwes"), token_count))

    for key in ("bio_tags", "mwe_tags", "tags"):
        raw = row.get(key)
        if isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
            mentions.extend(_mentions_from_bio(raw))
            break

    return tuple(dict.fromkeys(mentions))


def _rows_to_split_data(rows: list[dict[str, Any]], split: str) -> SplitData:
    sentences: list[Sentence] = []
    for idx, row in enumerate(rows):
        tokens = _extract_tokens(row)
        mentions = _extract_mentions(row, token_count=len(tokens))
        sent_id = str(row.get("sent_id") or row.get("id") or f"{split}:{idx}")
        sentences.append(Sentence(sent_id=sent_id, tokens=tokens, mentions=mentions))
    return SplitData(dataset=DATASET_NAME, split=split, sentences=tuple(sentences))


def _load_from_json_fallback(root: Path, split: str) -> SplitData | None:
    json_candidates = sorted(path for path in root.glob("*.json") if path.is_file())
    for path in json_candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            maybe_rows = payload.get(split)
            if isinstance(maybe_rows, list):
                rows = [row for row in maybe_rows if isinstance(row, dict)]
                return _rows_to_split_data(rows, split)
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
            return _rows_to_split_data(rows, split)
    return None


def load_split(data_root: Path, split: str, **_: object) -> SplitData:
    saved = _saved_dataset_dir(data_root)
    if saved.exists():
        datasets = _load_dataset_module()
        dataset_dict = datasets.load_from_disk(str(saved))
        if split not in dataset_dict:
            raise KeyError(f"Split {split!r} not found in saved CoAM dataset.")
        rows = [dict(row) for row in dataset_dict[split]]
        return _rows_to_split_data(rows, split)

    fallback = _load_from_json_fallback(_coam_root(data_root), split)
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        f"CoAM data not found at {saved}. Run download first or provide JSON fallback."
    )

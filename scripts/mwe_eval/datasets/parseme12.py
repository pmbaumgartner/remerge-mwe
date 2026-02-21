from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import subprocess

from .base import GoldMention, Sentence, SplitData

DATASET_NAME = "parseme12"
_PARSEME_GIT_URL = "https://gitlab.com/parseme/sharedtask-data.git"
_PARSEME_TAG = "st1.2-release-training-v1"


def _parseme_repo_dir(data_root: Path) -> Path:
    return data_root / "parseme" / "sharedtask-data"


def _run_checked(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}): {stderr or 'no stderr output'}"
        ) from exc


def download(data_root: Path, *, force: bool = False) -> Path:
    repo_dir = _parseme_repo_dir(data_root)
    if repo_dir.exists() and not force:
        return repo_dir

    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(["git", "clone", _PARSEME_GIT_URL, repo_dir.name], cwd=repo_dir.parent)
    _run_checked(["git", "checkout", _PARSEME_TAG], cwd=repo_dir)
    return repo_dir


def _matches_split(path: Path, split: str) -> bool:
    target = split.lower()
    stem = path.stem.lower()
    name = path.name.lower()
    parent = path.parent.name.lower()
    if stem == target or parent == target:
        return True
    return target in name


def _path_has_language(path: Path, language: str) -> bool:
    upper = language.upper()
    return any(part.upper() == upper for part in path.parts)


def _discover_cupt_files(
    repo_dir: Path, split: str, parseme_langs: list[str]
) -> list[Path]:
    all_cupt = sorted(repo_dir.glob("**/*.cupt"))
    selected: list[Path] = []
    for language in parseme_langs:
        lang_paths = [
            path
            for path in all_cupt
            if _path_has_language(path, language) and _matches_split(path, split)
        ]
        selected.extend(lang_paths)

    # Keep deterministic order and remove duplicates.
    return sorted(dict.fromkeys(selected))


def _extract_sent_id(comments: list[str], fallback: str) -> str:
    for comment in comments:
        if comment.startswith("# sent_id ="):
            return comment.split("=", 1)[1].strip()
        if comment.startswith("# source_sent_id ="):
            return comment.split("=", 1)[1].strip()
    return fallback


def _parse_mwe_cell(cell: str) -> list[tuple[str, str | None]]:
    value = cell.strip()
    if value in {"", "*", "_"}:
        return []

    out: list[tuple[str, str | None]] = []
    for part in value.split(";"):
        chunk = part.strip()
        if not chunk or chunk in {"*", "_"}:
            continue
        if ":" in chunk:
            mention_id, label = chunk.split(":", 1)
            out.append((mention_id.strip(), label.strip() or None))
        else:
            out.append((chunk, None))
    return out


def _parse_sentence(
    rows: list[str],
    comments: list[str],
    *,
    fallback_sent_id: str,
) -> Sentence:
    tokens: list[str] = []
    mention_to_indices: dict[str, list[int]] = defaultdict(list)
    mention_to_label: dict[str, str | None] = {}

    for row in rows:
        cols = row.split("\t")
        if len(cols) < 2:
            continue

        token_id = cols[0]
        if "-" in token_id or "." in token_id:
            # Multiword token or empty node.
            continue
        try:
            int(token_id)
        except ValueError:
            continue

        token_index = len(tokens)
        tokens.append(cols[1])

        mwe_cell = cols[-1] if cols else "*"
        for mention_id, label in _parse_mwe_cell(mwe_cell):
            if not mention_id:
                continue
            mention_to_indices[mention_id].append(token_index)
            if label is not None and mention_id not in mention_to_label:
                mention_to_label[mention_id] = label

    mentions: list[GoldMention] = []
    for mention_id in sorted(
        mention_to_indices.keys(), key=lambda value: (not value.isdigit(), value)
    ):
        indices = tuple(sorted(set(mention_to_indices[mention_id])))
        if len(indices) < 2:
            continue
        mentions.append(GoldMention(indices, mention_to_label.get(mention_id)))

    sent_id = _extract_sent_id(comments, fallback_sent_id)
    return Sentence(sent_id=sent_id, tokens=tuple(tokens), mentions=tuple(mentions))


def _parse_cupt_file(path: Path, *, language: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    rows: list[str] = []
    comments: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            if rows:
                fallback_sent_id = f"{language}:{path.stem}:{len(sentences)}"
                sentences.append(
                    _parse_sentence(rows, comments, fallback_sent_id=fallback_sent_id)
                )
            rows = []
            comments = []
            continue

        if line.startswith("#"):
            comments.append(line)
        else:
            rows.append(line)

    if rows:
        fallback_sent_id = f"{language}:{path.stem}:{len(sentences)}"
        sentences.append(
            _parse_sentence(rows, comments, fallback_sent_id=fallback_sent_id)
        )

    return sentences


def load_split(
    data_root: Path,
    split: str,
    **kwargs: object,
) -> SplitData:
    parseme_langs_raw = kwargs.get("parseme_langs", ["EN"])
    if isinstance(parseme_langs_raw, (list, tuple)):
        parseme_langs = [str(language).upper() for language in parseme_langs_raw]
    else:
        parseme_langs = ["EN"]

    repo_dir = _parseme_repo_dir(data_root)
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"PARSEME data not found at {repo_dir}. Run download first."
        )

    split_files = _discover_cupt_files(repo_dir, split, parseme_langs)
    if not split_files:
        raise FileNotFoundError(
            f"No PARSEME .cupt files found for split={split!r}, languages={parseme_langs}."
        )

    sentences: list[Sentence] = []
    for path in split_files:
        language = next(
            (part for part in path.parts if part.upper() in set(parseme_langs)),
            parseme_langs[0],
        )
        sentences.extend(_parse_cupt_file(path, language=language))

    return SplitData(dataset=DATASET_NAME, split=split, sentences=tuple(sentences))

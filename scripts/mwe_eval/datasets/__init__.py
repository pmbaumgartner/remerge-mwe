from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from . import coam, parseme12, streusle

_DATASET_MODULES: dict[str, ModuleType] = {
    "parseme12": parseme12,
    "streusle": streusle,
    "coam": coam,
}


def dataset_names() -> tuple[str, ...]:
    return tuple(_DATASET_MODULES.keys())


def expand_dataset(dataset: str) -> tuple[str, ...]:
    if dataset == "all":
        return dataset_names()
    if dataset not in _DATASET_MODULES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Expected one of: {dataset_names()}."
        )
    return (dataset,)


def load_split(dataset: str, data_root: Path, split: str, **kwargs: Any):
    module = _DATASET_MODULES[dataset]
    return module.load_split(data_root, split, **kwargs)


def download(dataset: str, data_root: Path, *, force: bool = False) -> Path:
    module = _DATASET_MODULES[dataset]
    return module.download(data_root, force=force)

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.mwe_eval import datasets

_DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "mwe_eval"
).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download datasets for MWE evaluation."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[*datasets.dataset_names(), "all"],
        help="Dataset to download.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help=f"Root directory for downloaded datasets (default: {_DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when existing files are present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_root: Path = args.data_root.resolve()
    names = datasets.expand_dataset(args.dataset)

    for name in names:
        print(f"[download] dataset={name} root={data_root}")
        downloaded_path = datasets.download(name, data_root, force=args.force)
        print(f"[done] {name}: {downloaded_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

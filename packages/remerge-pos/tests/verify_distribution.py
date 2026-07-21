"""Verify that remerge-pos distributions contain code and MIT metadata only."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile


PACKAGE = Path(__file__).parents[1]
VERSION = "0.1.0"
WHEEL_PREFIX = f"remerge_pos-{VERSION}.dist-info"
SDIST_PREFIX = f"remerge_pos-{VERSION}"
MODULE_FILES = {
    "remerge_pos/__init__.py",
    "remerge_pos/_api.py",
    "remerge_pos/_perceptron.py",
    "remerge_pos/training.py",
}


def _files(names: list[str]) -> set[str]:
    return {name for name in names if not name.endswith("/")}


def verify_wheel(path: Path) -> None:
    expected = MODULE_FILES | {
        f"{WHEEL_PREFIX}/METADATA",
        f"{WHEEL_PREFIX}/RECORD",
        f"{WHEEL_PREFIX}/WHEEL",
        f"{WHEEL_PREFIX}/licenses/LICENSE",
    }
    with zipfile.ZipFile(path) as archive:
        actual = _files(archive.namelist())
        if actual != expected:
            raise RuntimeError(
                f"unexpected wheel contents: {sorted(actual ^ expected)}"
            )
        metadata = BytesParser().parsebytes(archive.read(f"{WHEEL_PREFIX}/METADATA"))
        if metadata["Name"] != "remerge-pos":
            raise RuntimeError("wheel has the wrong project name")
        if metadata["License-Expression"] != "MIT":
            raise RuntimeError("wheel has the wrong license expression")
        if metadata.get_all("Requires-Dist"):
            raise RuntimeError("wheel unexpectedly declares runtime dependencies")
        if (
            archive.read(f"{WHEEL_PREFIX}/licenses/LICENSE")
            != (PACKAGE / "LICENSE").read_bytes()
        ):
            raise RuntimeError("wheel license does not match the package license")


def verify_sdist(path: Path) -> None:
    expected = {
        f"{SDIST_PREFIX}/LICENSE",
        f"{SDIST_PREFIX}/PKG-INFO",
        f"{SDIST_PREFIX}/README.md",
        f"{SDIST_PREFIX}/pyproject.toml",
        *(f"{SDIST_PREFIX}/src/{name}" for name in MODULE_FILES),
    }
    with tarfile.open(path, "r:gz") as archive:
        actual = {member.name for member in archive.getmembers() if member.isfile()}
        if actual != expected:
            raise RuntimeError(
                f"unexpected sdist contents: {sorted(actual ^ expected)}"
            )
        license_member = archive.extractfile(f"{SDIST_PREFIX}/LICENSE")
        if (
            license_member is None
            or license_member.read() != (PACKAGE / "LICENSE").read_bytes()
        ):
            raise RuntimeError("sdist license does not match the package license")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    arguments = parser.parse_args()
    verify_wheel(arguments.wheel)
    verify_sdist(arguments.sdist)
    print("distribution contents and metadata passed")


if __name__ == "__main__":
    main()

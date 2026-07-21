"""Verify that remerge-pos distributions contain code and MIT metadata only."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import tarfile
import tomllib
import zipfile


PACKAGE = Path(__file__).parents[1]
VERSION = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]["version"]
WHEEL_PREFIX = f"remerge_pos-{VERSION}.dist-info"
SDIST_PREFIX = f"remerge_pos-{VERSION}"
MODULE_FILES = {
    "remerge_pos/__init__.py",
    "remerge_pos/_api.py",
    "remerge_pos/_perceptron.py",
    "remerge_pos/training.py",
}
QUALIFIED_MODULE_SHA256 = {
    "remerge_pos/__init__.py": "41c0f80ebd1edc3ce7383194cc808e556bd50625dc7d2a5ab6bb440e7651ea54",
    "remerge_pos/_api.py": "4ed92b85f64d03420b235bce674100a124181f10017b38623d25cf67a6887d50",
    "remerge_pos/_perceptron.py": "6edfbf3e806dda4d3717dfb2f84a6c973552e416f91048c5f099402d0bd9f17a",
    "remerge_pos/training.py": "b627623bc118e6396e2d21364e1eb4a892a4a003f95d32d1802f2d5184c525b8",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        for name, expected_sha256 in QUALIFIED_MODULE_SHA256.items():
            if _sha256(archive.read(name)) != expected_sha256:
                raise RuntimeError(f"wheel module differs from Q1: {name}")


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
        for name, expected_sha256 in QUALIFIED_MODULE_SHA256.items():
            member = archive.extractfile(f"{SDIST_PREFIX}/src/{name}")
            if member is None or _sha256(member.read()) != expected_sha256:
                raise RuntimeError(f"sdist module differs from Q1: {name}")


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

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Reject release artifacts outside the pretagged-only package boundary."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


FORBIDDEN_PARTS = {"bin", "docs", "tests", "reference_corpus"}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".conllu",
    ".conllulex",
    ".onnx",
    ".pt",
    ".safetensors",
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def forbidden_member(name: str) -> bool:
    path = PurePosixPath(name)
    lowered_parts = {part.lower() for part in path.parts}
    return (
        bool(lowered_parts & FORBIDDEN_PARTS)
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
    )


def audit_wheel(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if any(forbidden_member(name) for name in names):
            raise AssertionError("wheel contains forbidden test, corpus, or model data")
        if any(
            not (name.startswith("remerge/") or ".dist-info/" in name) for name in names
        ):
            raise AssertionError("wheel contains a file outside remerge or dist-info")
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise AssertionError("wheel must contain exactly one METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        if any(line.startswith("Requires-Dist:") for line in metadata.splitlines()):
            raise AssertionError("wheel unexpectedly declares a runtime dependency")
        if "License-Expression: MIT" not in metadata and "License: MIT" not in metadata:
            raise AssertionError("wheel metadata does not declare the MIT license")
        stub_names = [name for name in names if name.endswith("remerge/_core.pyi")]
        if len(stub_names) != 1:
            raise AssertionError("wheel must contain the native extension stub")
        if b"LinearPosModel" in archive.read(stub_names[0]):
            raise AssertionError("wheel exposes the rejected experimental model loader")
    return {"path": str(path), "sha256": digest(path), "members": names}


def audit_sdist(path: Path) -> dict[str, object]:
    with tarfile.open(path, "r:*") as archive:
        names = sorted(
            member.name for member in archive.getmembers() if member.isfile()
        )
    if any(forbidden_member(name) for name in names):
        raise AssertionError("sdist contains forbidden test, corpus, or model data")
    allowed_top_level = {
        ".cargo_vcs_info.json",
        "Cargo.lock",
        "Cargo.toml",
        "Cargo.toml.orig",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
    }
    for name in names:
        relative = PurePosixPath(*PurePosixPath(name).parts[1:])
        if str(relative) in allowed_top_level:
            continue
        if relative.parts[:2] == ("rust", "src"):
            if relative.name == "linear.rs":
                raise AssertionError("sdist contains the rejected experimental loader")
            continue
        if relative.parts[:2] == ("src", "remerge"):
            continue
        raise AssertionError(f"sdist member is outside the allowlist: {relative}")
    return {"path": str(path), "sha256": digest(path), "members": names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evidence = {
        "schema_version": 1,
        "status": "accepted",
        "wheel": audit_wheel(args.wheel),
        "sdist": audit_sdist(args.sdist),
    }
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()

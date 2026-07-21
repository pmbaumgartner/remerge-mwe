#!/usr/bin/env python3
"""Build and verify the E1-authorized ``remerge-pos`` prerelease assets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/remerge-pos"
VERIFY_DISTRIBUTION = PACKAGE / "tests/verify_distribution.py"
PACKAGING_SMOKE = PACKAGE / "tests/packaging_smoke.py"
PROBE = ROOT / "tools/pos_package_probe.py"
REPOSITORY = "pmbaumgartner/remerge-mwe"
REMOTE = "origin"

VERSION = "0.1.0a1"
TAG = f"remerge-pos-v{VERSION}"
MODEL_DISTRIBUTION = f"remerge-pos-en-ewt-{VERSION}"
MODEL_ARCHIVE_NAME = f"{MODEL_DISTRIBUTION}.tar.gz"
MODEL_ID = "remerge-pos-structured-perceptron-v1"
QUALIFIED_SOURCE_REVISION = "47ab7cb3060d0319d71e02ed4be0afcca6f2d95a"
SOURCE_DATE_EPOCH = 1_784_597_194
MODEL_SHA256 = "a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d"
MODEL_BYTES = 1_515_159
LICENSE_SHA256 = "b3d1b0f4c6ae151f7eb78738f46ebd5ee140f8a7f76501ba26af123140d35ae7"
TRAIN_SHA256 = "e6a3784727e7726d4f1c2e10ff22dcd0ddeb8869ad6f85b73bb1ff5ef798e944"
DEV_SHA256 = "ef962ac05d844eaff46eeded125937129bfc0876d43963d66810cb73ffa8f5df"
FINAL_SHA256 = "465100c5b6fb606725068dc70680d65cc047fbb4ae25195375618bc6353a4e58"
EWT_REVISION = "b33472d3ce50a62056d6057f1b8a723e9d211176"
EWT_REPOSITORY = "https://github.com/UniversalDependencies/UD_English-EWT.git"
EWT_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
EWT_ACQUISITION_DATE = "2026-07-19"
EWT_ACQUISITION_COMMANDS = (
    f"git clone {EWT_REPOSITORY} <acquisition-root>/ud-ewt",
    f"git -C <acquisition-root>/ud-ewt checkout --detach {EWT_REVISION}",
)
EVALUATION_MANIFEST = "tests/pos/evaluation/manifest.json"
EVALUATION_MANIFEST_SHA256 = (
    "6bcf551e4834352bef22465f938c17317bfa4d614d87799bb8804754d10182db"
)
TRAINER_REVISION = "548a22df4f90da5a658f4eac90cf6f2207de4b16"
TRAINER_COMMAND = (
    "uv run --no-sync python experiments/pos-structured-perceptron/evaluate.py "
    "--acquisition-root <acquisition-root> --c2-artifact <c2-artifact> "
    "--tnt-artifact <tnt-artifact> --tnt-report <tnt-report> "
    "--output-dir <output-dir> --output <report.json>"
)
DEPENDENCY_LOCK_SHA256 = (
    "0d8f680c8a0a5804c1140ac9bd68f7cc0dc0bc8771ee3ed6ea0e0a849b3b94fd"
)

MAX_MODEL_BYTES = 20 * 1024 * 1024
MAX_TEXT_BYTES = 128 * 1024
MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MODULE_PATHS = (
    Path("packages/remerge-pos/src/remerge_pos/__init__.py"),
    Path("packages/remerge-pos/src/remerge_pos/_api.py"),
    Path("packages/remerge-pos/src/remerge_pos/_perceptron.py"),
    Path("packages/remerge-pos/src/remerge_pos/training.py"),
)


class ReleaseError(RuntimeError):
    """A prerelease construction or verification invariant failed."""


class PublicationOutcomeUnknown(ReleaseError):
    """A remote mutation started but its final GitHub state needs reconciliation."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    evidence = {
        "argv": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode:
        raise ReleaseError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-2000:]}"
        )
    return evidence


def _run_binary(command: Sequence[str], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=destination,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
        )
    stderr = completed.stderr.decode(errors="replace")
    if completed.returncode:
        output.unlink(missing_ok=True)
        raise ReleaseError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{stderr[-2000:]}"
        )
    return {
        "argv": list(command),
        "returncode": completed.returncode,
        "stderr": stderr[-4000:],
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
    }


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _clean_source_revision() -> str:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise ReleaseError("release construction requires a clean committed checkout")
    return _git("rev-parse", "HEAD")


def _assert_committed(path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(ROOT.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise ReleaseError(
            f"release control is not in this checkout: {path}"
        ) from error
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode or completed.stdout != path.read_bytes():
        raise ReleaseError(f"release control is not committed at HEAD: {relative}")
    return relative.as_posix()


def _assert_remote_head(source_revision: str) -> str:
    branch = _git("branch", "--show-current")
    if not branch:
        raise ReleaseError("release staging requires a named branch")
    result = _run(["git", "ls-remote", "--heads", REMOTE, f"refs/heads/{branch}"])[
        "stdout"
    ].strip()
    if not result or result.split()[0] != source_revision:
        raise ReleaseError(f"{REMOTE}/{branch} does not contain the release controls")
    return branch


def _assert_target_in_head(target_commit: str) -> None:
    _run(["git", "merge-base", "--is-ancestor", target_commit, "HEAD"])


def _remote_tag_target() -> str | None:
    result = _run(["git", "ls-remote", "--tags", REMOTE, f"refs/tags/{TAG}"])[
        "stdout"
    ].strip()
    if not result:
        return None
    lines = [line.split() for line in result.splitlines()]
    if len(lines) != 1 or len(lines[0]) != 2:
        raise ReleaseError(f"remote tag {TAG} is ambiguous")
    return lines[0][0]


def _ensure_remote_tag(target_commit: str) -> dict[str, Any]:
    remote_target = _remote_tag_target()
    if remote_target is not None:
        if remote_target != target_commit:
            raise ReleaseError(f"remote tag {TAG} points to the wrong commit")
        return {"created": False, "target_commit": remote_target}
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        _run(["git", "tag", TAG, target_commit])
    elif completed.stdout.strip() != target_commit:
        raise ReleaseError(f"local tag {TAG} points to the wrong commit")
    push = _run(["git", "push", REMOTE, f"refs/tags/{TAG}"])
    if _remote_tag_target() != target_commit:
        raise ReleaseError(f"remote tag {TAG} did not resolve to the release target")
    return {"created": True, "target_commit": target_commit, "push": push}


def _release_view_optional() -> dict[str, Any] | None:
    command = [
        "gh",
        "release",
        "view",
        TAG,
        "--repo",
        REPOSITORY,
        "--json",
        "apiUrl,assets,body,isDraft,isPrerelease,name,tagName,targetCommitish,url",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        if "release not found" in completed.stderr.lower():
            return None
        raise ReleaseError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-2000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError("GitHub release response is not JSON") from error
    if not isinstance(value, dict):
        raise ReleaseError("GitHub release response is not an object")
    return value


def _release_view() -> dict[str, Any]:
    value = _release_view_optional()
    if value is None:
        raise ReleaseError(f"GitHub release {TAG} does not exist")
    return value


def _release_assets(
    release: Mapping[str, Any], plan: Mapping[str, Any], *, draft: bool
) -> dict[str, Mapping[str, Any]]:
    if (
        release.get("tagName") != TAG
        or release.get("targetCommitish") != plan["target_commit"]
        or release.get("name") != f"remerge-pos {VERSION}"
        or release.get("isDraft") is not draft
        or release.get("isPrerelease") is not True
    ):
        raise ReleaseError("GitHub release identity or state differs")
    body = release.get("body")
    if (
        not isinstance(body, str)
        or {
            "filename": plan["notes"]["filename"],
            "sha256": _sha_bytes(body.encode()),
            "bytes": len(body.encode()),
        }
        != plan["notes"]
    ):
        raise ReleaseError("GitHub release notes differ from the manifest")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ReleaseError("GitHub release assets are missing")
    by_name = {asset.get("name"): asset for asset in assets if isinstance(asset, dict)}
    expected_names = {details["filename"] for details in plan["assets"].values()}
    if set(by_name) != expected_names or len(by_name) != len(assets):
        raise ReleaseError("GitHub release asset set differs from the manifest")
    for details in plan["assets"].values():
        asset = by_name[details["filename"]]
        if (
            asset.get("size") != details["bytes"]
            or not isinstance(asset.get("apiUrl"), str)
            or not asset.get("apiUrl")
            or not isinstance(asset.get("id"), str)
            or not asset.get("id")
        ):
            raise ReleaseError(f"GitHub release asset metadata differs: {asset}")
    return by_name


def _asset_snapshot(assets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("apiUrl", "digest", "id", "name", "size", "state", "updatedAt")
    return {
        name: {field: asset.get(field) for field in fields}
        for name, asset in sorted(assets.items())
    }


def _download_release_assets(
    assets: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    destination: Path,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    if destination.exists():
        raise ReleaseError("release download directory must not already exist")
    destination.mkdir(parents=True)
    paths = {}
    commands = []
    for logical_name, details in sorted(plan["assets"].items()):
        asset = assets[details["filename"]]
        path = destination / details["filename"]
        commands.append(
            _run_binary(
                [
                    "gh",
                    "api",
                    asset["apiUrl"],
                    "--header",
                    "Accept: application/octet-stream",
                ],
                path,
            )
        )
        paths[logical_name] = path
    return paths, commands


def _qualified_module_parity() -> dict[str, str]:
    hashes = {}
    for path in MODULE_PATHS:
        actual = (ROOT / path).read_bytes()
        qualified = subprocess.run(
            ["git", "show", f"{QUALIFIED_SOURCE_REVISION}:{path}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        if actual != qualified:
            raise ReleaseError(f"distributed module differs from Q1: {path}")
        hashes[str(path)] = _sha_bytes(actual)
    return hashes


def _model_card() -> bytes:
    return (
        "# remerge-pos English EWT model 0.1.0a1\n\n"
        "Experimental averaged structured perceptron for caller-tokenized English "
        "UPOS tagging. It was trained only on UD English EWT r2.10 and is "
        "distributed separately from the MIT code package under the conservative "
        "CC BY-SA 4.0 posture.\n\n"
        "Qualified EWT final diagnostics: 94.2625% overall accuracy, 0.9236 "
        "macro-F1, 78.7947% OOV accuracy, and 94.6775% ambiguous-token "
        "accuracy. These are English web-text diagnostics, not a broad-English "
        "claim. MWE behavior was not a package gate.\n\n"
        "The artifact contains numeric model parameters and provenance hashes, "
        "not corpus text, gold labels, Python pickle, or executable payload. "
        "Load it only from an explicit local path with remerge-pos 0.1.0a1.\n"
    ).encode()


def _attribution() -> bytes:
    return (
        "# Attribution and modification notice\n\n"
        "Model name: remerge-pos English EWT averaged structured perceptron\n\n"
        "Source: UD English EWT r2.10\n\n"
        f"Source repository: {EWT_REPOSITORY}\n\n"
        f"Source revision: {EWT_REVISION}\n\n"
        f"Source license URL: {EWT_LICENSE_URL}\n\n"
        f"Acquired: {EWT_ACQUISITION_DATE}\n\n"
        "Acquisition commands:\n\n"
        f"- `{EWT_ACQUISITION_COMMANDS[0]}`\n"
        f"- `{EWT_ACQUISITION_COMMANDS[1]}`\n\n"
        f"Evaluation manifest: {EVALUATION_MANIFEST}\n\n"
        f"Evaluation manifest SHA-256: {EVALUATION_MANIFEST_SHA256}\n\n"
        "Source license: Creative Commons Attribution-ShareAlike 4.0 "
        "International (CC BY-SA 4.0)\n\n"
        "Changes: EWT UPOS annotations were converted through the frozen project "
        "loader and used to train a six-epoch averaged structured perceptron with "
        "feature cutoff 1, 131072 feature buckets, and seed 20260720. The "
        "distribution contains the trained parameter artifact, model card, "
        "manifest, this notice, and the CC license; it contains no source corpus "
        "or gold-label files. The model was trained by the recorded command at "
        f"source revision {TRAINER_REVISION}.\n"
    ).encode()


def _manifest(
    model: bytes,
    license_text: bytes,
    model_card: bytes,
    attribution: bytes,
    source_revision: str,
) -> bytes:
    value = {
        "schema_version": 1,
        "distribution": MODEL_DISTRIBUTION,
        "version": VERSION,
        "tag": TAG,
        "code_compatibility": f"remerge-pos=={VERSION}",
        "model_id": MODEL_ID,
        "artifact_schema": MODEL_ID,
        "artifact_filename": "model.rmsp",
        "artifact_sha256": _sha_bytes(model),
        "artifact_bytes": len(model),
        "license": "CC BY-SA 4.0",
        "license_filename": "LICENSES/CC-BY-SA-4.0.txt",
        "license_sha256": _sha_bytes(license_text),
        "source": "UD English EWT r2.10",
        "source_repository": EWT_REPOSITORY,
        "source_revision": EWT_REVISION,
        "source_license_url": EWT_LICENSE_URL,
        "source_acquisition_date": EWT_ACQUISITION_DATE,
        "source_acquisition_commands": list(EWT_ACQUISITION_COMMANDS),
        "evaluation_manifest": EVALUATION_MANIFEST,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "release_source_revision": source_revision,
        "qualified_source_revision": QUALIFIED_SOURCE_REVISION,
        "trainer_revision": TRAINER_REVISION,
        "trainer_command": TRAINER_COMMAND,
        "dependency_lock": "uv.lock",
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "feature_schema": "remerge-pos-structured-perceptron-features-v1",
        "model_format": MODEL_ID,
        "qualification_evidence": "docs/evidence/pos-package-qualification-q1.json",
        "reproducible_export": "pass",
        "training_input_sha256": TRAIN_SHA256,
        "development_input_sha256": DEV_SHA256,
        "protected_final_input_sha256": FINAL_SHA256,
        "config": {
            "epochs": 6,
            "feature_buckets": 131072,
            "feature_cutoff": 1,
        },
        "seed": 20260720,
        "raw_corpus_in_distribution": False,
        "gold_labels_in_distribution": False,
        "files": {
            "ATTRIBUTION.md": {
                "bytes": len(attribution),
                "sha256": _sha_bytes(attribution),
            },
            "LICENSES/CC-BY-SA-4.0.txt": {
                "bytes": len(license_text),
                "sha256": _sha_bytes(license_text),
            },
            "MODEL_CARD.md": {
                "bytes": len(model_card),
                "sha256": _sha_bytes(model_card),
            },
            "model.rmsp": {
                "bytes": len(model),
                "sha256": _sha_bytes(model),
            },
        },
    }
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _model_members(
    model_path: Path, license_path: Path, source_revision: str
) -> dict[str, bytes]:
    model = model_path.read_bytes()
    license_text = license_path.read_bytes()
    if len(model) != MODEL_BYTES or _sha_bytes(model) != MODEL_SHA256:
        raise ReleaseError("model bytes differ from the qualified Q1 artifact")
    if len(model) > MAX_MODEL_BYTES:
        raise ReleaseError("model exceeds the release size boundary")
    if _sha_bytes(license_text) != LICENSE_SHA256:
        raise ReleaseError("CC BY-SA license text differs from the pinned EWT copy")
    if len(license_text) > MAX_TEXT_BYTES:
        raise ReleaseError("license text exceeds the release size boundary")
    card = _model_card()
    attribution = _attribution()
    manifest = _manifest(model, license_text, card, attribution, source_revision)
    return {
        "ATTRIBUTION.md": attribution,
        "LICENSES/CC-BY-SA-4.0.txt": license_text,
        "MANIFEST.json": manifest,
        "MODEL_CARD.md": card,
        "model.rmsp": model,
    }


def _build_model_archive(
    model_path: Path,
    license_path: Path,
    output: Path,
    source_revision: str,
) -> None:
    members = _model_members(model_path, license_path, source_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_destination:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_destination,
            mtime=SOURCE_DATE_EPOCH,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for relative, contents in sorted(members.items()):
                    member = tarfile.TarInfo(f"{MODEL_DISTRIBUTION}/{relative}")
                    member.size = len(contents)
                    member.mode = 0o644
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    member.mtime = SOURCE_DATE_EPOCH
                    archive.addfile(member, fileobj=io.BytesIO(contents))


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "" not in path.parts


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    maximum = MAX_MODEL_BYTES if member.name.endswith("/model.rmsp") else MAX_TEXT_BYTES
    if member.size > maximum:
        raise ReleaseError(f"model distribution member is too large: {member.name}")
    source = archive.extractfile(member)
    if source is None:
        raise ReleaseError(f"cannot read model distribution member: {member.name}")
    value = source.read(maximum + 1)
    if len(value) != member.size or len(value) > maximum:
        raise ReleaseError(f"model distribution member size changed: {member.name}")
    return value


def _verify_model_archive(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ReleaseError("compressed model distribution exceeds 24 MiB")
    expected = {
        f"{MODEL_DISTRIBUTION}/ATTRIBUTION.md",
        f"{MODEL_DISTRIBUTION}/LICENSES/CC-BY-SA-4.0.txt",
        f"{MODEL_DISTRIBUTION}/MANIFEST.json",
        f"{MODEL_DISTRIBUTION}/MODEL_CARD.md",
        f"{MODEL_DISTRIBUTION}/model.rmsp",
    }
    with tarfile.open(path, "r|gz") as archive:
        names = set()
        contents = {}
        total_bytes = 0
        for member in archive:
            if len(names) >= len(expected):
                raise ReleaseError("model distribution has too many members")
            if member.name in names:
                raise ReleaseError("model distribution has duplicate members")
            names.add(member.name)
            if not _safe_member_name(member.name):
                raise ReleaseError(f"unsafe model distribution path: {member.name}")
            if not member.isfile() or member.issym() or member.islnk():
                raise ReleaseError(
                    f"model distribution member is not a regular file: {member.name}"
                )
            if member.mode & 0o777 != 0o644:
                raise ReleaseError(
                    f"model distribution member has wrong mode: {member.name}"
                )
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
                or member.mtime != SOURCE_DATE_EPOCH
            ):
                raise ReleaseError(
                    f"model distribution member metadata is not canonical: {member.name}"
                )
            value = _read_member(archive, member)
            total_bytes += len(value)
            if total_bytes > MAX_MODEL_BYTES + 4 * MAX_TEXT_BYTES:
                raise ReleaseError(
                    "model distribution expands beyond its size boundary"
                )
            contents[member.name.removeprefix(f"{MODEL_DISTRIBUTION}/")] = value
        if names != expected:
            raise ReleaseError(
                f"unexpected model distribution members: {sorted(names ^ expected)}"
            )
    manifest = json.loads(contents["MANIFEST.json"])
    if not isinstance(manifest, dict) or not _valid_commit(
        manifest.get("release_source_revision")
    ):
        raise ReleaseError("model distribution source revision is invalid")
    expected_card = _model_card()
    expected_attribution = _attribution()
    if contents["MODEL_CARD.md"] != expected_card:
        raise ReleaseError("model distribution model card differs")
    if contents["ATTRIBUTION.md"] != expected_attribution:
        raise ReleaseError("model distribution attribution differs")
    expected_manifest = _manifest(
        contents["model.rmsp"],
        contents["LICENSES/CC-BY-SA-4.0.txt"],
        expected_card,
        expected_attribution,
        manifest["release_source_revision"],
    )
    if contents["MANIFEST.json"] != expected_manifest:
        raise ReleaseError("model distribution manifest differs")
    if _sha_bytes(contents["model.rmsp"]) != MODEL_SHA256:
        raise ReleaseError("model archive changed the qualified artifact")
    if _sha_bytes(contents["LICENSES/CC-BY-SA-4.0.txt"]) != LICENSE_SHA256:
        raise ReleaseError("model archive changed the CC BY-SA license")
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "members": sorted(expected),
        "manifest": manifest,
    }


def _build_code_distributions(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True)
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = str(SOURCE_DATE_EPOCH)
    command = _run(
        [
            "uv",
            "build",
            "--package",
            "remerge-pos",
            "--wheel",
            "--sdist",
            "--out-dir",
            str(output),
        ],
        env=environment,
    )
    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    if wheel.name != f"remerge_pos-{VERSION}-py3-none-any.whl":
        raise ReleaseError(f"unexpected wheel filename: {wheel.name}")
    if sdist.name != f"remerge_pos-{VERSION}.tar.gz":
        raise ReleaseError(f"unexpected sdist filename: {sdist.name}")
    return {
        "command": command,
        "wheel": wheel,
        "sdist": sdist,
    }


def _venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _model_from_archive(path: Path, destination: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(f"{MODEL_DISTRIBUTION}/model.rmsp")
        destination.write_bytes(_read_member(archive, member))


def _smoke_distributions(
    wheel: Path, sdist: Path, model_archive: Path, work: Path
) -> dict[str, Any]:
    model = work / "qualified-model.rmsp"
    _model_from_archive(model_archive, model)
    evidence = {}
    for name, distribution in (("wheel", wheel), ("sdist", sdist)):
        environment = work / f"{name}-environment"
        _run(["uv", "venv", "--python", sys.executable, str(environment)])
        python = _venv_python(environment)
        install = _run(
            ["uv", "pip", "install", "--python", str(python), str(distribution)]
        )
        version = _run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('remerge-pos'))",
            ],
            cwd=work,
        )
        if version["stdout"].strip() != VERSION:
            raise ReleaseError(f"{name} installed the wrong version")
        smoke = _run([str(python), "-I", str(PACKAGING_SMOKE)], cwd=work)
        api_output = work / f"{name}-qualified-model.json"
        qualified_model = _run(
            [
                str(python),
                "-I",
                str(PROBE),
                "--artifact",
                str(model),
                "--output",
                str(api_output),
                "--api-controls",
            ],
            cwd=work,
        )
        api = json.loads(api_output.read_text(encoding="utf-8"))
        if api["artifact_sha256"] != MODEL_SHA256:
            raise ReleaseError(f"{name} smoke changed the qualified model identity")
        offline = _run(
            [
                str(python),
                "-I",
                "-c",
                "import http.client,os,socket,subprocess,urllib.request; "
                "blocked=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network')); "
                "socket.create_connection=blocked; socket.socket=blocked; "
                "urllib.request.urlopen=blocked; "
                "http.client.HTTPConnection.connect=blocked; "
                "http.client.HTTPSConnection.connect=blocked; "
                "subprocess.Popen=blocked; os.system=blocked; "
                "import remerge_pos; assert callable(remerge_pos.load_model)",
            ],
            cwd=work,
        )
        evidence[name] = {
            "install": install,
            "version": VERSION,
            "synthetic_smoke": smoke,
            "qualified_model_smoke": qualified_model,
            "api_controls": api["controls"],
            "offline_import": offline,
        }
    return evidence


def _verify_code_distributions(wheel: Path, sdist: Path) -> dict[str, Any]:
    command = _run([sys.executable, str(VERIFY_DISTRIBUTION), str(wheel), str(sdist)])
    return {
        "command": command,
        "wheel": {
            "filename": wheel.name,
            "sha256": _sha(wheel),
            "bytes": wheel.stat().st_size,
        },
        "sdist": {
            "filename": sdist.name,
            "sha256": _sha(sdist),
            "bytes": sdist.stat().st_size,
        },
    }


def _load_plan(plan_path: Path) -> dict[str, Any]:
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError("pre-publication manifest is not an object")
    if (
        value.get("schema_version") != 1
        or value.get("tag") != TAG
        or value.get("version") != VERSION
        or value.get("model_id") != MODEL_ID
        or value.get("qualified_source_revision") != QUALIFIED_SOURCE_REVISION
        or value.get("code_compatibility") != f"remerge-pos=={VERSION}"
        or value.get("model_distribution") != MODEL_DISTRIBUTION
    ):
        raise ReleaseError("pre-publication manifest identity differs")
    target_commit = value.get("target_commit")
    if not _valid_commit(target_commit):
        raise ReleaseError("pre-publication manifest target commit is invalid")
    expected_assets = value.get("assets")
    if not isinstance(expected_assets, dict) or set(expected_assets) != {
        "wheel",
        "sdist",
        "model",
    }:
        raise ReleaseError("pre-publication manifest asset set differs")
    for name, details in expected_assets.items():
        if (
            not isinstance(details, dict)
            or set(details) != {"filename", "sha256", "bytes"}
            or not isinstance(details["filename"], str)
            or not details["filename"]
            or Path(details["filename"]).name != details["filename"]
            or not _valid_sha256(details["sha256"])
            or not isinstance(details["bytes"], int)
            or isinstance(details["bytes"], bool)
            or details["bytes"] <= 0
        ):
            raise ReleaseError(f"pre-publication manifest asset is invalid: {name}")
    if len({details["filename"] for details in expected_assets.values()}) != 3:
        raise ReleaseError("pre-publication manifest asset filenames are not unique")
    notes = value.get("notes")
    if (
        not isinstance(notes, dict)
        or set(notes) != {"filename", "sha256", "bytes"}
        or notes.get("filename") != "docs/pos_package_prerelease.md"
        or not _valid_sha256(notes.get("sha256"))
        or not isinstance(notes.get("bytes"), int)
        or isinstance(notes.get("bytes"), bool)
        or notes["bytes"] <= 0
    ):
        raise ReleaseError("pre-publication manifest release notes are invalid")
    return value


def _verify_plan(plan_path: Path, assets: Mapping[str, Path]) -> dict[str, Any]:
    value = _load_plan(plan_path)
    expected_assets = value["assets"]
    if set(expected_assets) != set(assets):
        raise ReleaseError("pre-publication manifest asset set differs")
    for name, path in assets.items():
        expected = expected_assets[name]
        if expected != {
            "filename": path.name,
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
        }:
            raise ReleaseError(f"pre-publication manifest differs for {name}")
    return value


def _verify_notes(plan: Mapping[str, Any], path: Path) -> None:
    resolved = path.resolve(strict=True)
    if {
        "filename": resolved.relative_to(ROOT.resolve()).as_posix(),
        "sha256": _sha(resolved),
        "bytes": resolved.stat().st_size,
    } != plan["notes"]:
        raise ReleaseError("release notes differ from the pre-publication manifest")


def _verify_assets(
    wheel: Path,
    sdist: Path,
    model_archive: Path,
    *,
    plan: Path | None = None,
) -> dict[str, Any]:
    code = _verify_code_distributions(wheel, sdist)
    model = _verify_model_archive(model_archive)
    assets = {"wheel": wheel, "sdist": sdist, "model": model_archive}
    plan_value = _verify_plan(plan, assets) if plan else None
    if plan_value and (
        model["manifest"]["release_source_revision"] != plan_value["target_commit"]
    ):
        raise ReleaseError(
            "model distribution source revision differs from the release target"
        )
    with TemporaryDirectory(prefix="remerge-pos-prerelease-smoke-") as directory:
        smoke = _smoke_distributions(wheel, sdist, model_archive, Path(directory))
    return {
        "code": code,
        "model": model,
        "smoke": smoke,
        "prepublication_manifest": plan_value,
    }


def _build(arguments: argparse.Namespace) -> dict[str, Any]:
    source_revision = _clean_source_revision()
    module_hashes = _qualified_module_parity()
    with TemporaryDirectory(prefix="remerge-pos-prerelease-build-") as directory:
        work = Path(directory)
        builds = []
        for index in range(2):
            destination = work / f"build-{index}"
            code = _build_code_distributions(destination)
            model = destination / MODEL_ARCHIVE_NAME
            _build_model_archive(
                arguments.model_artifact,
                arguments.license_file,
                model,
                source_revision,
            )
            builds.append(
                {
                    "code": code,
                    "model": model,
                    "hashes": {
                        "wheel": _sha(code["wheel"]),
                        "sdist": _sha(code["sdist"]),
                        "model": _sha(model),
                    },
                }
            )
        if builds[0]["hashes"] != builds[1]["hashes"]:
            raise ReleaseError("two isolated prerelease builds are not byte-identical")
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        final_paths = {}
        for name, source in {
            "wheel": builds[0]["code"]["wheel"],
            "sdist": builds[0]["code"]["sdist"],
            "model": builds[0]["model"],
        }.items():
            destination = arguments.output_dir / source.name
            shutil.copy2(source, destination)
            final_paths[name] = destination
        verification = _verify_assets(
            final_paths["wheel"], final_paths["sdist"], final_paths["model"]
        )
    return {
        "schema_version": 1,
        "phase": "prepublication-build",
        "status": "pass",
        "tag": TAG,
        "version": VERSION,
        "source_revision": source_revision,
        "qualified_source_revision": QUALIFIED_SOURCE_REVISION,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "qualified_module_hashes": module_hashes,
        "reproducibility": {
            "isolated_builds": 2,
            "byte_identical": True,
            "hashes": builds[0]["hashes"],
        },
        "verification": verification,
        "assets": {
            name: {
                "filename": path.name,
                "path": str(path.resolve()),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
            for name, path in final_paths.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def _verify(arguments: argparse.Namespace) -> dict[str, Any]:
    verification = _verify_assets(
        arguments.wheel,
        arguments.sdist,
        arguments.model_archive,
        plan=arguments.plan,
    )
    return {
        "schema_version": 1,
        "phase": "download-verification",
        "status": "pass",
        "tag": TAG,
        "version": VERSION,
        "assets": {
            name: {
                "filename": path.name,
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
            for name, path in {
                "wheel": arguments.wheel,
                "sdist": arguments.sdist,
                "model": arguments.model_archive,
            }.items()
        },
        "verification": verification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def _release_inputs(
    plan_path: Path,
    assets: Mapping[str, Path] | None = None,
) -> tuple[str, dict[str, Any]]:
    source_revision = _clean_source_revision()
    _assert_committed(plan_path)
    plan = (
        _verify_plan(plan_path, assets) if assets is not None else _load_plan(plan_path)
    )
    _assert_target_in_head(plan["target_commit"])
    _assert_remote_head(source_revision)
    return source_revision, plan


def _stage(arguments: argparse.Namespace) -> dict[str, Any]:
    local_assets = {
        "wheel": arguments.wheel,
        "sdist": arguments.sdist,
        "model": arguments.model_archive,
    }
    source_revision, plan = _release_inputs(arguments.plan, local_assets)
    _assert_committed(arguments.notes)
    _verify_notes(plan, arguments.notes)
    verification = _verify_assets(
        arguments.wheel,
        arguments.sdist,
        arguments.model_archive,
        plan=arguments.plan,
    )
    if _release_view_optional() is not None:
        raise ReleaseError(f"GitHub release {TAG} already exists")
    _write_json(
        arguments.evidence,
        {
            "schema_version": 1,
            "phase": "draft-release-verification",
            "status": "in-progress",
            "tag": TAG,
            "source_revision": source_revision,
            "target_commit": plan["target_commit"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
        },
    )
    try:
        tag = _ensure_remote_tag(plan["target_commit"])
        create = _run(
            [
                "gh",
                "release",
                "create",
                TAG,
                str(arguments.wheel),
                str(arguments.sdist),
                str(arguments.model_archive),
                "--repo",
                REPOSITORY,
                "--target",
                plan["target_commit"],
                "--verify-tag",
                "--draft",
                "--prerelease",
                "--latest=false",
                "--title",
                f"remerge-pos {VERSION}",
                "--notes-file",
                str(arguments.notes),
            ]
        )
        release = _release_view()
        remote_assets = _release_assets(release, plan, draft=True)
        downloaded, downloads = _download_release_assets(
            remote_assets, plan, arguments.download_dir
        )
        downloaded_verification = _verify_assets(
            downloaded["wheel"],
            downloaded["sdist"],
            downloaded["model"],
            plan=arguments.plan,
        )
        if _remote_tag_target() != plan["target_commit"]:
            raise ReleaseError(f"remote tag {TAG} changed during draft verification")
        release_after = _release_view()
        assets_after = _release_assets(release_after, plan, draft=True)
        if _asset_snapshot(assets_after) != _asset_snapshot(remote_assets):
            raise ReleaseError("GitHub draft assets changed during verification")
    except Exception as error:
        raise PublicationOutcomeUnknown(
            "GitHub draft staging started but did not verify; run the read-only "
            "status command, then the state-checked abort command if cleanup is needed"
        ) from error
    return {
        "schema_version": 1,
        "phase": "draft-release-verification",
        "status": "pass",
        "tag": TAG,
        "version": VERSION,
        "source_revision": source_revision,
        "target_commit": plan["target_commit"],
        "remote_tag": tag,
        "release": {
            "api_url": release_after["apiUrl"],
            "url": release_after["url"],
            "is_draft": release_after["isDraft"],
            "is_prerelease": release_after["isPrerelease"],
            "assets": _asset_snapshot(assets_after),
        },
        "create": create,
        "downloads": downloads,
        "local_verification": verification,
        "downloaded_verification": downloaded_verification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def _publish(arguments: argparse.Namespace) -> dict[str, Any]:
    source_revision, plan = _release_inputs(arguments.plan)
    if _remote_tag_target() != plan["target_commit"]:
        raise ReleaseError(f"remote tag {TAG} differs from the release target")
    draft = _release_view()
    draft_assets = _release_assets(draft, plan, draft=True)
    with TemporaryDirectory(prefix="remerge-pos-prerelease-download-") as directory:
        downloaded, downloads = _download_release_assets(
            draft_assets, plan, Path(directory) / "assets"
        )
        verification = _verify_assets(
            downloaded["wheel"],
            downloaded["sdist"],
            downloaded["model"],
            plan=arguments.plan,
        )
    prepublish = _release_view()
    prepublish_assets = _release_assets(prepublish, plan, draft=True)
    if _asset_snapshot(prepublish_assets) != _asset_snapshot(draft_assets):
        raise ReleaseError("GitHub draft assets changed before publication")
    if _remote_tag_target() != plan["target_commit"]:
        raise ReleaseError(f"remote tag {TAG} changed before publication")
    _write_json(
        arguments.evidence,
        {
            "schema_version": 1,
            "phase": "prerelease-publication",
            "status": "in-progress",
            "tag": TAG,
            "target_commit": plan["target_commit"],
            "draft_url": draft["url"],
            "verified_assets": _asset_snapshot(draft_assets),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
        },
    )
    try:
        edit = _run(
            [
                "gh",
                "release",
                "edit",
                TAG,
                "--repo",
                REPOSITORY,
                "--draft=false",
                "--prerelease=true",
                "--latest=false",
            ]
        )
        published = _release_view()
        published_assets = _release_assets(published, plan, draft=False)
        if _asset_snapshot(published_assets) != _asset_snapshot(draft_assets):
            raise ReleaseError("GitHub assets changed during publication")
    except Exception as error:
        raise PublicationOutcomeUnknown(
            "GitHub publication started but its final state is unknown; run the "
            "read-only status command before any retry or cleanup"
        ) from error
    return {
        "schema_version": 1,
        "phase": "prerelease-publication",
        "status": "pass",
        "tag": TAG,
        "version": VERSION,
        "source_revision": source_revision,
        "target_commit": plan["target_commit"],
        "release": {
            "api_url": published["apiUrl"],
            "url": published["url"],
            "is_draft": published["isDraft"],
            "is_prerelease": published["isPrerelease"],
            "assets": _asset_snapshot(published_assets),
        },
        "downloads": downloads,
        "verification": verification,
        "publication": edit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def _status(arguments: argparse.Namespace) -> dict[str, Any]:
    _assert_committed(arguments.plan)
    plan = _load_plan(arguments.plan)
    remote_target = _remote_tag_target()
    if remote_target is not None and remote_target != plan["target_commit"]:
        raise ReleaseError("GitHub release tag differs from the manifest target")
    release = _release_view_optional()
    if release is None:
        return {
            "schema_version": 1,
            "phase": "release-reconciliation",
            "status": "pass",
            "tag": TAG,
            "remote_state": "tag-only" if remote_target else "absent",
            "target_commit": plan["target_commit"],
            "remote_tag_target": remote_target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
        }
    if remote_target != plan["target_commit"]:
        raise ReleaseError("GitHub release tag differs from the manifest target")
    is_draft = release.get("isDraft")
    if not isinstance(is_draft, bool):
        raise ReleaseError("GitHub release draft state is invalid")
    assets = _release_assets(release, plan, draft=is_draft)
    with TemporaryDirectory(prefix="remerge-pos-prerelease-status-") as directory:
        downloaded, downloads = _download_release_assets(
            assets, plan, Path(directory) / "assets"
        )
        verification = _verify_assets(
            downloaded["wheel"],
            downloaded["sdist"],
            downloaded["model"],
            plan=arguments.plan,
        )
    return {
        "schema_version": 1,
        "phase": "release-reconciliation",
        "status": "pass",
        "tag": TAG,
        "remote_state": "draft" if is_draft else "public-prerelease",
        "target_commit": plan["target_commit"],
        "remote_tag_target": remote_target,
        "release": {
            "api_url": release["apiUrl"],
            "url": release["url"],
            "is_draft": release["isDraft"],
            "is_prerelease": release["isPrerelease"],
            "assets": _asset_snapshot(assets),
        },
        "downloads": downloads,
        "verification": verification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def _abortable_draft(
    release: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        release.get("tagName") != TAG
        or release.get("targetCommitish") != plan["target_commit"]
        or release.get("name") != f"remerge-pos {VERSION}"
        or release.get("isDraft") is not True
        or release.get("isPrerelease") is not True
    ):
        raise ReleaseError("refusing to remove a release outside the expected draft")
    assets = release.get("assets")
    if not isinstance(assets, list) or any(
        not isinstance(asset, dict) for asset in assets
    ):
        raise ReleaseError("draft asset metadata is invalid")
    planned = {details["filename"]: details for details in plan["assets"].values()}
    names = [asset.get("name") for asset in assets]
    if len(set(names)) != len(names) or any(name not in planned for name in names):
        raise ReleaseError("refusing to remove a draft with unplanned assets")
    for asset in assets:
        if asset.get("size") != planned[asset["name"]]["bytes"]:
            raise ReleaseError("refusing to remove a draft with changed asset metadata")
    return {
        "api_url": release.get("apiUrl"),
        "url": release.get("url"),
        "body_sha256": _sha_bytes(str(release.get("body", "")).encode()),
        "assets": _asset_snapshot(
            {asset["name"]: asset for asset in assets if isinstance(asset, dict)}
        ),
    }


def _abort(arguments: argparse.Namespace) -> dict[str, Any]:
    source_revision, plan = _release_inputs(arguments.plan)
    remote_target = _remote_tag_target()
    if remote_target is not None and remote_target != plan["target_commit"]:
        raise ReleaseError("refusing to remove a tag outside the release target")
    release = _release_view_optional()
    if release is None and remote_target is None:
        raise ReleaseError("no expected draft or tag exists")
    draft_snapshot = _abortable_draft(release, plan) if release else None
    _write_json(
        arguments.evidence,
        {
            "schema_version": 1,
            "phase": "draft-release-abort",
            "status": "in-progress",
            "tag": TAG,
            "source_revision": source_revision,
            "target_commit": plan["target_commit"],
            "remote_tag_target": remote_target,
            "draft": draft_snapshot,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
        },
    )
    try:
        if release:
            removal = _run(
                [
                    "gh",
                    "release",
                    "delete",
                    TAG,
                    "--repo",
                    REPOSITORY,
                    "--cleanup-tag",
                    "--yes",
                ]
            )
        else:
            removal = _run(["git", "push", REMOTE, "--delete", TAG])
        if _release_view_optional() is not None or _remote_tag_target() is not None:
            raise ReleaseError("draft or tag remains after abort")
    except Exception as error:
        raise PublicationOutcomeUnknown(
            "GitHub cleanup started but its final state is unknown; run the "
            "read-only status command before any retry"
        ) from error
    return {
        "schema_version": 1,
        "phase": "draft-release-abort",
        "status": "pass",
        "tag": TAG,
        "source_revision": source_revision,
        "target_commit": plan["target_commit"],
        "removed_draft": draft_snapshot,
        "removal": removal,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--model-artifact", type=Path, required=True)
    build.add_argument("--license-file", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--evidence", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--wheel", type=Path, required=True)
    verify.add_argument("--sdist", type=Path, required=True)
    verify.add_argument("--model-archive", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--evidence", type=Path, required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--wheel", type=Path, required=True)
    stage.add_argument("--sdist", type=Path, required=True)
    stage.add_argument("--model-archive", type=Path, required=True)
    stage.add_argument("--plan", type=Path, required=True)
    stage.add_argument("--notes", type=Path, required=True)
    stage.add_argument("--download-dir", type=Path, required=True)
    stage.add_argument("--evidence", type=Path, required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--plan", type=Path, required=True)
    publish.add_argument("--evidence", type=Path, required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument("--evidence", type=Path, required=True)
    abort = subparsers.add_parser("abort")
    abort.add_argument("--plan", type=Path, required=True)
    abort.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        commands = {
            "build": _build,
            "verify": _verify,
            "stage": _stage,
            "publish": _publish,
            "status": _status,
            "abort": _abort,
        }
        report = commands[arguments.command](arguments)
    except PublicationOutcomeUnknown as error:
        prior = {}
        try:
            prior_value = json.loads(arguments.evidence.read_text(encoding="utf-8"))
            if isinstance(prior_value, dict):
                prior = prior_value
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        report = {
            **prior,
            "schema_version": 1,
            "status": "needs-human",
            "failure": f"{type(error).__name__}: {error}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
        }
        _write_json(arguments.evidence, report)
        raise SystemExit(3) from error
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "reject",
            "failure": f"{type(error).__name__}: {error}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
        }
        _write_json(arguments.evidence, report)
        raise SystemExit(2) from error
    _write_json(arguments.evidence, report)


if __name__ == "__main__":
    main()

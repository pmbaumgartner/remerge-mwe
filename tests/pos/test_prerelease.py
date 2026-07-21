"""Focused controls for the prerelease model archive verifier."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "tools/pos_prerelease.py"


def _module():
    spec = importlib.util.spec_from_file_location("pos_prerelease_test", TOOL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prerelease = _module()


@pytest.fixture
def identities(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, bytes]:
    model = b"small qualified model"
    license_text = b"small CC BY-SA license"
    monkeypatch.setattr(prerelease, "MODEL_BYTES", len(model))
    monkeypatch.setattr(prerelease, "MODEL_SHA256", prerelease._sha_bytes(model))
    monkeypatch.setattr(
        prerelease, "LICENSE_SHA256", prerelease._sha_bytes(license_text)
    )
    return model, license_text


def _archive_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            assert source is not None
            members[member.name.removeprefix(f"{prerelease.MODEL_DISTRIBUTION}/")] = (
                source.read()
            )
        return members


def _write_archive(
    path: Path,
    entries: list[tuple[str, bytes]],
    *,
    mode: int = 0o644,
    uid: int = 0,
    mtime: int | None = None,
    link: str | None = None,
) -> None:
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name, contents in entries:
            member = tarfile.TarInfo(name)
            member.mode = mode
            member.uid = uid
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = prerelease.SOURCE_DATE_EPOCH if mtime is None else mtime
            if link is not None:
                member.type = tarfile.SYMTYPE
                member.linkname = link
                member.size = 0
                archive.addfile(member)
            else:
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))


@pytest.fixture
def archive(tmp_path: Path, identities: tuple[bytes, bytes]) -> Path:
    model, license_text = identities
    model_path = tmp_path / "model.rmsp"
    license_path = tmp_path / "license.txt"
    archive_path = tmp_path / "model.tar.gz"
    model_path.write_bytes(model)
    license_path.write_bytes(license_text)
    prerelease._build_model_archive(model_path, license_path, archive_path, "a" * 40)
    return archive_path


def test_model_archive_is_deterministic_and_verifies_with_complete_manifest(
    tmp_path: Path, identities: tuple[bytes, bytes]
) -> None:
    model, license_text = identities
    model_path = tmp_path / "model.rmsp"
    license_path = tmp_path / "license.txt"
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    model_path.write_bytes(model)
    license_path.write_bytes(license_text)

    prerelease._build_model_archive(model_path, license_path, first, "a" * 40)
    prerelease._build_model_archive(model_path, license_path, second, "a" * 40)
    verified = prerelease._verify_model_archive(first)

    assert first.read_bytes() == second.read_bytes()
    assert verified["manifest"]["artifact_sha256"] == prerelease.MODEL_SHA256
    assert verified["manifest"]["artifact_bytes"] == len(model)
    assert verified["manifest"]["source_acquisition_date"] == "2026-07-19"
    assert verified["manifest"]["source_acquisition_commands"] == list(
        prerelease.EWT_ACQUISITION_COMMANDS
    )
    assert (
        verified["manifest"]["evaluation_manifest_sha256"]
        == prerelease.EVALUATION_MANIFEST_SHA256
    )
    assert verified["manifest"]["trainer_revision"] == prerelease.TRAINER_REVISION
    assert (
        verified["manifest"]["dependency_lock_sha256"]
        == prerelease.DEPENDENCY_LOCK_SHA256
    )
    assert verified["members"] == sorted(
        f"{prerelease.MODEL_DISTRIBUTION}/{name}"
        for name in (
            "ATTRIBUTION.md",
            "LICENSES/CC-BY-SA-4.0.txt",
            "MANIFEST.json",
            "MODEL_CARD.md",
            "model.rmsp",
        )
    )


def test_verifier_rejects_duplicate_members(tmp_path: Path, archive: Path) -> None:
    members = _archive_members(archive)
    entries = [
        (f"{prerelease.MODEL_DISTRIBUTION}/{member_name}", contents)
        for member_name, contents in members.items()
    ]
    entries.append((f"{prerelease.MODEL_DISTRIBUTION}/model.rmsp", b"duplicate"))
    corrupted = tmp_path / "corrupted.tar.gz"
    _write_archive(corrupted, entries)

    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(corrupted)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        f"{prerelease.MODEL_DISTRIBUTION}/../model.rmsp",
        "/model.rmsp",
    ),
)
def test_verifier_rejects_path_traversal_and_absolute_members(
    tmp_path: Path, archive: Path, unsafe_name: str
) -> None:
    members = _archive_members(archive)
    entries = [
        (f"{prerelease.MODEL_DISTRIBUTION}/{member_name}", contents)
        for member_name, contents in members.items()
    ]
    entries.append((unsafe_name, b"unsafe"))
    corrupted = tmp_path / "unsafe.tar.gz"
    _write_archive(corrupted, entries)

    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(corrupted)


def test_verifier_rejects_symlink_and_noncanonical_metadata(
    tmp_path: Path, archive: Path
) -> None:
    members = _archive_members(archive)
    entries = [
        (f"{prerelease.MODEL_DISTRIBUTION}/{name}", contents)
        for name, contents in members.items()
    ]
    symlink = tmp_path / "symlink.tar.gz"
    _write_archive(symlink, entries, link="model.rmsp")
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(symlink)

    wrong_metadata = tmp_path / "wrong-metadata.tar.gz"
    _write_archive(wrong_metadata, entries, mode=0o600, uid=4, mtime=0)
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(wrong_metadata)


def test_verifier_rejects_oversize_and_manifest_or_payload_corruption(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prerelease, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(archive)
    monkeypatch.undo()

    members = _archive_members(archive)
    members["MANIFEST.json"] = json.dumps({"distribution": "wrong"}).encode()
    manifest_corruption = tmp_path / "manifest-corruption.tar.gz"
    _write_archive(
        manifest_corruption,
        [
            (f"{prerelease.MODEL_DISTRIBUTION}/{name}", value)
            for name, value in members.items()
        ],
    )
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(manifest_corruption)

    members = _archive_members(archive)
    members["model.rmsp"] += b"changed"
    hash_corruption = tmp_path / "hash-corruption.tar.gz"
    _write_archive(
        hash_corruption,
        [
            (f"{prerelease.MODEL_DISTRIBUTION}/{name}", value)
            for name, value in members.items()
        ],
    )
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_model_archive(hash_corruption)


def test_prepublication_plan_rejects_missing_asset_or_hash_mismatch(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in ("wheel", "sdist", "model"):
        path = tmp_path / f"{name}.asset"
        path.write_bytes(name.encode())
        assets[name] = path
    plan = tmp_path / "plan.json"
    expected_assets = {
        name: {
            "filename": path.name,
            "sha256": prerelease._sha(path),
            "bytes": path.stat().st_size,
        }
        for name, path in assets.items()
    }
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tag": prerelease.TAG,
                "version": prerelease.VERSION,
                "model_id": prerelease.MODEL_ID,
                "qualified_source_revision": prerelease.QUALIFIED_SOURCE_REVISION,
                "code_compatibility": f"remerge-pos=={prerelease.VERSION}",
                "model_distribution": prerelease.MODEL_DISTRIBUTION,
                "target_commit": "a" * 40,
                "assets": expected_assets,
                "notes": {
                    "filename": "docs/pos_package_prerelease.md",
                    "sha256": "0" * 64,
                    "bytes": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    assert prerelease._verify_plan(plan, assets)["assets"] == expected_assets

    expected_assets.pop("model")
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    plan_value["assets"] = expected_assets
    plan.write_text(json.dumps(plan_value), encoding="utf-8")
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_plan(plan, assets)

    expected_assets["model"] = {
        "filename": assets["model"].name,
        "sha256": "0" * 64,
        "bytes": assets["model"].stat().st_size,
    }
    plan_value["assets"] = expected_assets
    plan.write_text(json.dumps(plan_value), encoding="utf-8")
    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_plan(plan, assets)


def test_prepublication_plan_rejects_invalid_target_commit(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in ("wheel", "sdist", "model"):
        path = tmp_path / f"{name}.asset"
        path.write_bytes(name.encode())
        assets[name] = path
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tag": prerelease.TAG,
                "version": prerelease.VERSION,
                "model_id": prerelease.MODEL_ID,
                "qualified_source_revision": prerelease.QUALIFIED_SOURCE_REVISION,
                "code_compatibility": f"remerge-pos=={prerelease.VERSION}",
                "model_distribution": prerelease.MODEL_DISTRIBUTION,
                "target_commit": "candidate",
                "assets": {
                    name: {
                        "filename": path.name,
                        "sha256": prerelease._sha(path),
                        "bytes": path.stat().st_size,
                    }
                    for name, path in assets.items()
                },
                "notes": {
                    "filename": "docs/pos_package_prerelease.md",
                    "sha256": "0" * 64,
                    "bytes": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(prerelease.ReleaseError):
        prerelease._verify_plan(plan, assets)


def test_github_release_requires_exact_draft_state_and_asset_set() -> None:
    body = "frozen release notes\n"
    plan = {
        "target_commit": "a" * 40,
        "assets": {
            name: {
                "filename": f"{name}.asset",
                "sha256": str(index) * 64,
                "bytes": index,
            }
            for index, name in enumerate(("wheel", "sdist", "model"), start=1)
        },
        "notes": {
            "filename": "docs/pos_package_prerelease.md",
            "sha256": prerelease._sha_bytes(body.encode()),
            "bytes": len(body.encode()),
        },
    }
    release = {
        "tagName": prerelease.TAG,
        "targetCommitish": "a" * 40,
        "name": f"remerge-pos {prerelease.VERSION}",
        "body": body,
        "isDraft": True,
        "isPrerelease": True,
        "assets": [
            {
                "name": details["filename"],
                "size": details["bytes"],
                "apiUrl": f"https://api.github.test/assets/{name}",
                "id": name,
            }
            for name, details in plan["assets"].items()
        ],
    }

    assert set(prerelease._release_assets(release, plan, draft=True)) == {
        "wheel.asset",
        "sdist.asset",
        "model.asset",
    }

    release["isDraft"] = False
    with pytest.raises(prerelease.ReleaseError):
        prerelease._release_assets(release, plan, draft=True)

    release["isDraft"] = True
    release["body"] = "changed release notes\n"
    with pytest.raises(prerelease.ReleaseError):
        prerelease._release_assets(release, plan, draft=True)

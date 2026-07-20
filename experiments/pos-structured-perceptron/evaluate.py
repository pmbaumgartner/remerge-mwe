#!/usr/bin/env python3
"""Run the bounded structured-perceptron pilot through the dev-only harness."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path(__file__).resolve().parent
POS_TAGGER = ROOT / "experiments/pos-tagger"
POS_TNT = ROOT / "experiments/pos-tnt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pos.evaluation.loader import load_manifest, load_tagged_split  # noqa: E402


SEEDS = (20260720, 20260721, 20260722, 20260723, 20260724)
GRID = (
    {"epochs": 4, "feature_cutoff": 2, "feature_buckets": 131_072},
    {"epochs": 6, "feature_cutoff": 1, "feature_buckets": 131_072},
)
C2_ARTIFACT_SHA256 = "3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d"
C2_MODEL_ID = "remerge-pos-linear-v1"
C2_TOKENIZER_ID = "unicode-whitespace-v1"
TNT_ARTIFACT_SHA256 = "76e850be9510aa64d69299592884da5a39e17e3a5b9a22844179e876231a9e94"
TNT_REPORT_SHA256 = "fdb003f9f862837ced77bc198ed672942501b907d2b6d195bb0dab740ebb567d"
TNT_MODEL_ID = "remerge-pos-tnt-v1"
CANONICAL_TOKENIZER_ID = "canonical-boundaries-v1"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sentences(rows):
    return tuple(
        tuple((token.form, token.upos) for token in sentence.tokens)
        for sentence in rows
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_path(manifest: dict, root: Path, split: str) -> Path:
    details = manifest["splits"][split]
    source = manifest["sources"][details["source"]]
    return root / source["relative_root"] / details["path"]


def _source_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("development evidence requires a clean committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    paths = (
        PILOT / "structured_perceptron.py",
        PILOT / "structured_perceptron_adapter.py",
        PILOT / "evaluate.py",
        POS_TAGGER / "bakeoff.py",
        POS_TAGGER / "c2_adapter.py",
        POS_TNT / "tnt.py",
        POS_TNT / "tnt_adapter.py",
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _adapter_environment(**variables: str) -> dict[str, str]:
    inherited = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        (str(PILOT), str(POS_TNT), str(POS_TAGGER), str(ROOT), inherited)
    )
    return os.environ | variables | {"PYTHONPATH": pythonpath}


def _registration(
    *,
    candidate_id: str,
    command: list[str],
    seed: int,
    data_hashes: dict[str, str],
    artifact_sha256: str,
    model_id: str,
    tokenizer_id: str,
    variable: str,
    artifact: Path,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_revision": _source_revision(),
        "command": shlex.join(command),
        "seed": seed,
        "data_hashes": data_hashes,
        "artifact_sha256": artifact_sha256,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "PYTHONPATH": _adapter_environment()["PYTHONPATH"],
        },
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "artifact_environment": {"variable": variable, "path": str(artifact)},
        "source_hashes": _source_hashes(),
    }


def _run_harness(
    *,
    specification: str,
    variable: str,
    artifact: Path,
    candidate_id: str,
    seed: int,
    acquisition_root: Path,
    output_dir: Path,
    data_hashes: dict[str, str],
    model_id: str,
    tokenizer_id: str,
) -> dict:
    registration = output_dir / f"{candidate_id}.registration.json"
    report = output_dir / f"{candidate_id}.bakeoff.json"
    command = [
        sys.executable,
        str(POS_TAGGER / "bakeoff.py"),
        "--candidate",
        specification,
        "--acquisition-root",
        str(acquisition_root),
        "--registration",
        str(registration),
        "--output",
        str(report),
    ]
    registration.write_text(
        json.dumps(
            _registration(
                candidate_id=candidate_id,
                command=command,
                seed=seed,
                data_hashes=data_hashes,
                artifact_sha256=_sha(artifact),
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                variable=variable,
                artifact=artifact,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_adapter_environment(**{variable: str(artifact)}),
        text=True,
        capture_output=True,
    )
    if completed.returncode not in (0, 2) or not report.is_file():
        raise RuntimeError(
            f"common harness failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    evidence = json.loads(report.read_text(encoding="utf-8"))
    if evidence.get("protected_final_evaluated") is not False:
        raise RuntimeError("common harness report did not remain development-only")
    return evidence | {
        "report_path": str(report),
        "registration_path": str(registration),
        "harness_returncode": completed.returncode,
    }


@contextmanager
def _development_snapshot(manifest: dict, acquisition_root: Path):
    copies = [
        ("train", manifest["splits"]["train"], "ud_ewt"),
        ("dev", manifest["splits"]["dev"], "ud_ewt"),
        (
            "mwe-dev",
            manifest["mwe_gold"]["splits"]["dev"],
            manifest["mwe_gold"]["source"],
        ),
    ]
    with TemporaryDirectory(prefix="remerge-pos-perceptron-dev-") as directory:
        root = Path(directory)
        hashes = {}
        for name, split, source_name in copies:
            relative_root = manifest["sources"][source_name]["relative_root"]
            source = acquisition_root / relative_root / split["path"]
            contents = source.read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            if digest != split["sha256"]:
                raise RuntimeError(f"{name} input differs from its manifest hash")
            destination = root / relative_root / split["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            destination.chmod(stat.S_IRUSR)
            hashes[name] = digest
        yield root, hashes


def _copy_pinned(source: Path, destination: Path, expected: str, label: str) -> Path:
    contents = source.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise RuntimeError(f"{label} digest does not match the pinned reference")
    destination.write_bytes(contents)
    destination.chmod(stat.S_IRUSR)
    return destination


def _improvements(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "overall_accuracy": candidate["quality"]["overall_accuracy"]
        - baseline["quality"]["overall_accuracy"],
        "macro_f1": candidate["quality"]["macro_f1"] - baseline["quality"]["macro_f1"],
        "oov_accuracy": candidate["quality"]["oov_accuracy"]
        - baseline["quality"]["oov_accuracy"],
        "ambiguous_accuracy": candidate["quality"]["ambiguous_accuracy"]
        - baseline["quality"]["ambiguous_accuracy"],
        "mwe_precision_generated": candidate["mwe_utility"]["precision_generated"]
        - baseline["mwe_utility"]["precision_generated"],
        "mwe_recall_generated": candidate["mwe_utility"]["recall_generated"]
        - baseline["mwe_utility"]["recall_generated"],
    }


def _decision(candidate: dict, c2: dict) -> tuple[str, list[str]]:
    failures = candidate["hard_dev_analog_gates"]["failures"]
    gains = _improvements(candidate, c2)
    credible = any(
        gains[name] > 0
        for name in (
            "oov_accuracy",
            "ambiguous_accuracy",
            "mwe_precision_generated",
            "mwe_recall_generated",
        )
    )
    if failures:
        return "reject", [f"common harness: {failure}" for failure in failures]
    if not credible:
        return "reject", ["no hard-slice or downstream MWE improvement over c2"]
    return "retain", []


SELECTION_METRICS = (
    ("quality", "overall_accuracy"),
    ("quality", "macro_f1"),
    ("quality", "oov_accuracy"),
    ("quality", "ambiguous_accuracy"),
    ("mwe_utility", "precision_generated"),
    ("mwe_utility", "recall_generated"),
)


def _grid_summaries(results: list[dict]) -> list[dict]:
    summaries = []
    for grid_index in range(len(GRID)):
        records = [record for record in results if record["grid_index"] == grid_index]
        if {record["seed"] for record in records} != set(SEEDS):
            raise RuntimeError("grid lacks the five predeclared true training seeds")
        means = {
            f"{section}.{metric}": sum(record[section][metric] for record in records)
            / len(records)
            for section, metric in SELECTION_METRICS
        }
        summaries.append(
            {"grid_index": grid_index, "config": GRID[grid_index], "means": means}
        )
    return summaries


def _select(results: list[dict]) -> tuple[dict, list[dict]]:
    summaries = _grid_summaries(results)
    selected_grid = max(
        summaries,
        key=lambda summary: (
            tuple(
                summary["means"][f"{section}.{metric}"]
                for section, metric in SELECTION_METRICS
            )
            + (-summary["grid_index"],)
        ),
    )
    selected = next(
        record
        for record in results
        if record["grid_index"] == selected_grid["grid_index"]
        and record["seed"] == SEEDS[0]
    )
    return selected, summaries


def evaluate(
    acquisition_root: Path,
    output_dir: Path,
    c2_artifact: Path,
    tnt_artifact: Path,
    tnt_report: Path,
) -> dict:
    model_module = _module(
        "pos_structured_perceptron_pilot", PILOT / "structured_perceptron.py"
    )
    c2_module = _module("pos_structured_perceptron_c2", POS_TAGGER / "c2_adapter.py")
    tnt_module = _module("pos_structured_perceptron_tnt", POS_TNT / "tnt.py")
    manifest = load_manifest()
    revision = _source_revision()
    try:
        output_dir.resolve().relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("development evidence output must be outside the checkout")
    output_dir.mkdir(parents=True, exist_ok=True)
    c2_snapshot = _copy_pinned(
        c2_artifact, output_dir / "retained-c2.rmpos", C2_ARTIFACT_SHA256, "c2 artifact"
    )
    tnt_snapshot = _copy_pinned(
        tnt_artifact,
        output_dir / "rejected-tnt.json",
        TNT_ARTIFACT_SHA256,
        "TnT artifact",
    )
    if _sha(tnt_report) != TNT_REPORT_SHA256:
        raise RuntimeError(
            "TnT evidence report digest does not match the completed pilot"
        )
    completed_tnt = json.loads(tnt_report.read_text(encoding="utf-8"))
    if (
        completed_tnt.get("status") != "reject"
        or completed_tnt.get("protected_final_evaluated") is not False
    ):
        raise RuntimeError(
            "TnT evidence is not the completed development-only rejection"
        )
    if c2_module.C2Adapter(c2_snapshot).model_id != C2_MODEL_ID:
        raise RuntimeError("c2 identity does not match the retained reference")
    if (
        tnt_module.Model.from_artifact(tnt_snapshot.read_bytes()).artifact()
        != tnt_snapshot.read_bytes()
    ):
        raise RuntimeError("TnT artifact does not strictly round-trip")
    with _development_snapshot(manifest, acquisition_root) as (
        snapshot_root,
        snapshot_hashes,
    ):
        train = load_tagged_split(manifest, snapshot_root, "train")
        data_hashes = {
            split: _sha(_split_path(manifest, snapshot_root, split))
            for split in ("train", "dev")
        }
        c2 = _run_harness(
            specification="c2_adapter:create_candidate",
            variable="REMERGE_POS_C2_ARTIFACT",
            artifact=c2_snapshot,
            candidate_id="c2-retained-reference",
            seed=20260719,
            acquisition_root=snapshot_root,
            output_dir=output_dir,
            data_hashes=data_hashes,
            model_id=C2_MODEL_ID,
            tokenizer_id=C2_TOKENIZER_ID,
        )
        tnt = _run_harness(
            specification="tnt_adapter:create_candidate",
            variable="REMERGE_POS_TNT_ARTIFACT",
            artifact=tnt_snapshot,
            candidate_id="tnt-rejected-reference",
            seed=20260720,
            acquisition_root=snapshot_root,
            output_dir=output_dir,
            data_hashes=data_hashes,
            model_id=TNT_MODEL_ID,
            tokenizer_id=CANONICAL_TOKENIZER_ID,
        )
        results = []
        training_sentences = _sentences(train)
        for grid_index, raw_config in enumerate(GRID):
            digests = set()
            for seed in SEEDS:
                artifact = (
                    output_dir / f"structured-perceptron-g{grid_index}-s{seed}.rmsp"
                )
                model = model_module.train(
                    training_sentences, model_module.Config(**raw_config), seed
                )
                digest = model_module.write_artifact(model, artifact)
                digests.add(digest)
                report = _run_harness(
                    specification="structured_perceptron_adapter:create_candidate",
                    variable="REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT",
                    artifact=artifact,
                    candidate_id=f"structured-perceptron-g{grid_index}-s{seed}",
                    seed=seed,
                    acquisition_root=snapshot_root,
                    output_dir=output_dir,
                    data_hashes=data_hashes,
                    model_id="remerge-pos-structured-perceptron-v1",
                    tokenizer_id=CANONICAL_TOKENIZER_ID,
                )
                record = report | {
                    "grid_index": grid_index,
                    "seed": seed,
                    "config": raw_config,
                }
                record["improvements_vs_c2"] = _improvements(record, c2)
                record["improvements_vs_tnt"] = _improvements(record, tnt)
                record["decision"], record["rejection_reasons"] = _decision(record, c2)
                results.append(record)
            if len(digests) != len(SEEDS):
                raise RuntimeError(
                    "true shuffled training seeds did not produce five distinct artifacts"
                )
    selected, summaries = _select(results)
    return {
        "schema_version": 1,
        "status": selected["decision"],
        "development_only": True,
        "protected_final_evaluated": False,
        "protocol": {
            "grid": GRID,
            "seeds": SEEDS,
            "seed_semantics": "true per-epoch sentence shuffles; every seed is trained and measured",
            "common_harness": str(POS_TAGGER / "bakeoff.py"),
            "selection": "maximize six five-seed means in declared order; lower grid index tie; select lowest fixed seed artifact",
        },
        "provenance": {
            "source_revision": revision,
            "source_hashes": _source_hashes(),
            "snapshot_hashes": snapshot_hashes,
            "c2_artifact_sha256": C2_ARTIFACT_SHA256,
            "tnt_artifact_sha256": TNT_ARTIFACT_SHA256,
            "tnt_report_sha256": TNT_REPORT_SHA256,
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "comparators": {"c2": c2, "tnt_rejected": tnt},
        "grid_summaries": summaries,
        "selected": {
            "grid_index": selected["grid_index"],
            "seed": selected["seed"],
            "artifact_sha256": selected["candidate"]["artifact_sha256"],
            "artifact_path": selected["candidate"]["artifact_path"],
            "decision": selected["decision"],
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--c2-artifact", type=Path, required=True)
    parser.add_argument("--tnt-artifact", type=Path, required=True)
    parser.add_argument("--tnt-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = {"status": "reject", "protected_final_evaluated": False}
    try:
        evidence = evaluate(
            arguments.acquisition_root,
            arguments.output_dir,
            arguments.c2_artifact,
            arguments.tnt_artifact,
            arguments.tnt_report,
        )
    except Exception as error:
        evidence["failure"] = f"{type(error).__name__}: {error}"
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(2) from error
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

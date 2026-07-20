#!/usr/bin/env python3
"""Run the bounded TnT development pilot through the common bakeoff harness.

There is deliberately no final-split argument.  Every evaluated artifact gets
its own attributable registration and is run through ``pos-tagger/bakeoff.py``
against the retained c2 artifact.
"""

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
POS_TNT = Path(__file__).resolve().parent
POS_TAGGER = ROOT / "experiments/pos-tagger"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pos.evaluation.loader import load_manifest, load_tagged_split  # noqa: E402


SEEDS = (20260720, 20260721, 20260722, 20260723, 20260724)
C2_ARTIFACT_SHA256 = "3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d"
C2_MODEL_ID = "remerge-pos-linear-v1"
C2_TOKENIZER_ID = "unicode-whitespace-v1"
GRID = (
    {
        "transition_alpha": 0.10,
        "emission_alpha": 0.10,
        "suffix_alpha": 0.25,
        "suffix_length": 3,
        "rare_word_max_count": 1,
    },
    {
        "transition_alpha": 0.25,
        "emission_alpha": 0.25,
        "suffix_alpha": 0.50,
        "suffix_length": 4,
        "rare_word_max_count": 2,
    },
)


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


def _split_path(manifest: dict, acquisition_root: Path, split: str) -> Path:
    source = manifest["sources"][manifest["splits"][split]["source"]]
    return (
        acquisition_root / source["relative_root"] / manifest["splits"][split]["path"]
    )


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


def _adapter_environment(**artifact_variables: str) -> dict[str, str]:
    inherited = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join((str(POS_TNT), str(POS_TAGGER), str(ROOT), inherited))
    return os.environ | artifact_variables | {"PYTHONPATH": pythonpath}


def _registration(
    *,
    candidate_id: str,
    command: list[str],
    seed: int,
    data_hashes: dict[str, str],
    artifact_sha256: str,
    model_id: str,
    tokenizer_id: str,
    artifact_variable: str,
    artifact_path: Path,
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
        "artifact_environment": {
            "variable": artifact_variable,
            "path": str(artifact_path),
        },
    }


def _run_common_harness(
    *,
    specification: str,
    artifact_variable: str,
    artifact_path: Path,
    candidate_id: str,
    seed: int,
    acquisition_root: Path,
    output_dir: Path,
    data_hashes: dict[str, str],
    model_id: str,
    tokenizer_id: str,
) -> dict:
    artifact_sha256 = _sha(artifact_path)
    registration_path = output_dir / f"{candidate_id}.registration.json"
    report_path = output_dir / f"{candidate_id}.bakeoff.json"
    command = [
        sys.executable,
        str(POS_TAGGER / "bakeoff.py"),
        "--candidate",
        specification,
        "--acquisition-root",
        str(acquisition_root),
        "--registration",
        str(registration_path),
        "--output",
        str(report_path),
    ]
    registration_path.write_text(
        json.dumps(
            _registration(
                candidate_id=candidate_id,
                command=command,
                seed=seed,
                data_hashes=data_hashes,
                artifact_sha256=artifact_sha256,
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                artifact_variable=artifact_variable,
                artifact_path=artifact_path,
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
        env=_adapter_environment(**{artifact_variable: str(artifact_path)}),
        text=True,
        capture_output=True,
    )
    if completed.returncode not in (0, 2) or not report_path.is_file():
        raise RuntimeError(
            f"common harness failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protected_final_evaluated") is not False:
        raise RuntimeError("common harness report did not remain development-only")
    return report | {
        "report_path": str(report_path),
        "registration_path": str(registration_path),
        "harness_returncode": completed.returncode,
    }


def _improvements(candidate: dict, c2: dict) -> dict[str, float]:
    quality = candidate["quality"]
    baseline = c2["quality"]
    utility = candidate["mwe_utility"]
    baseline_utility = c2["mwe_utility"]
    return {
        "overall_accuracy": quality["overall_accuracy"] - baseline["overall_accuracy"],
        "macro_f1": quality["macro_f1"] - baseline["macro_f1"],
        "oov_accuracy": quality["oov_accuracy"] - baseline["oov_accuracy"],
        "ambiguous_accuracy": quality["ambiguous_accuracy"]
        - baseline["ambiguous_accuracy"],
        "mwe_precision_generated": utility["precision_generated"]
        - baseline_utility["precision_generated"],
        "mwe_recall_generated": utility["recall_generated"]
        - baseline_utility["recall_generated"],
    }


def _pilot_decision(candidate: dict, c2: dict) -> tuple[str, list[str]]:
    improvements = _improvements(candidate, c2)
    failures = candidate["hard_dev_analog_gates"]["failures"]
    credible = any(
        improvements[name] > 0
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


@contextmanager
def _development_snapshot(manifest: dict, acquisition_root: Path):
    """Use immutable dev-only input bytes for both training and bakeoff.

    This prevents a source-path replacement between checksum verification and
    the child common-harness process, without ever opening a final-split path.
    """

    copies = [
        ("train", manifest["splits"]["train"], "ud_ewt"),
        ("dev", manifest["splits"]["dev"], "ud_ewt"),
        (
            "mwe-dev",
            manifest["mwe_gold"]["splits"]["dev"],
            manifest["mwe_gold"]["source"],
        ),
    ]
    with TemporaryDirectory(prefix="remerge-pos-tnt-dev-") as directory:
        root = Path(directory)
        for name, split, source_name in copies:
            relative_root = manifest["sources"][source_name]["relative_root"]
            source_path = acquisition_root / relative_root / split["path"]
            contents = source_path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != split["sha256"]:
                raise RuntimeError(
                    f"{name} input changed or does not match its manifest hash"
                )
            destination = root / relative_root / split["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            destination.chmod(stat.S_IRUSR)
        yield root


def _selected_grid(results: list[dict]) -> dict:
    """Choose one predeclared grid without any-run cherry-picking."""

    references = [record for record in results if record["seed"] == SEEDS[0]]
    return max(
        references,
        key=lambda record: (
            record["decision"] == "retain",
            record["quality"]["overall_accuracy"],
            record["quality"]["macro_f1"],
            record["quality"]["oov_accuracy"],
            record["quality"]["ambiguous_accuracy"],
            -record["grid_index"],
        ),
    )


def evaluate(acquisition_root: Path, output_dir: Path, c2_artifact: Path) -> dict:
    tnt = _module("pos_tnt_pilot", POS_TNT / "tnt.py")
    c2_adapter = _module("pos_tnt_c2_adapter", POS_TAGGER / "c2_adapter.py")
    manifest = load_manifest()
    _source_revision()
    try:
        output_dir.resolve().relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("development evidence output must be outside the checkout")
    output_dir.mkdir(parents=True, exist_ok=True)
    c2_bytes = c2_artifact.read_bytes()
    if hashlib.sha256(c2_bytes).hexdigest() != C2_ARTIFACT_SHA256:
        raise RuntimeError("c2 artifact digest does not match the retained reference")
    c2_snapshot = output_dir / "retained-c2.rmpos"
    c2_snapshot.write_bytes(c2_bytes)
    c2_snapshot.chmod(stat.S_IRUSR)
    with _development_snapshot(manifest, acquisition_root) as snapshot_root:
        c2 = c2_adapter.C2Adapter(c2_snapshot)
        if (c2.model_id, c2.tokenizer_id) != (C2_MODEL_ID, C2_TOKENIZER_ID):
            raise RuntimeError(
                "c2 artifact identity does not match the retained reference"
            )
        train = load_tagged_split(manifest, snapshot_root, "train")
        data_hashes = {
            split: _sha(_split_path(manifest, snapshot_root, split))
            for split in ("train", "dev")
        }
        c2_report = _run_common_harness(
            specification="c2_adapter:create_candidate",
            artifact_variable="REMERGE_POS_C2_ARTIFACT",
            artifact_path=c2_snapshot,
            candidate_id="c2-retained-reference",
            seed=20260719,
            acquisition_root=snapshot_root,
            output_dir=output_dir,
            data_hashes=data_hashes,
            model_id=c2.model_id,
            tokenizer_id=c2.tokenizer_id,
        )

        results = []
        for grid_index, raw_config in enumerate(GRID):
            repetitions = []
            for seed in SEEDS:
                # These are deterministic repetitions, not independent stochastic
                # samples.  Verify byte identity before reusing one harness run.
                artifact = output_dir / f"tnt-g{grid_index}-s{seed}.json"
                model = tnt.train(_sentences(train), tnt.Config(**raw_config))
                digest = tnt.write_artifact(model, artifact)
                repetitions.append(
                    {
                        "seed": seed,
                        "artifact_path": artifact,
                        "artifact_sha256": digest,
                        "artifact_bytes": artifact.stat().st_size,
                        "artifact": artifact.read_bytes(),
                    }
                )
            reference = repetitions[0]
            if not all(
                repetition["artifact"] == reference["artifact"]
                and repetition["artifact_sha256"] == reference["artifact_sha256"]
                for repetition in repetitions
            ):
                raise RuntimeError("deterministic TnT repetition artifacts differ")
            report = _run_common_harness(
                specification="tnt_adapter:create_candidate",
                artifact_variable="REMERGE_POS_TNT_ARTIFACT",
                artifact_path=reference["artifact_path"],
                candidate_id=f"tnt-g{grid_index}-s{reference['seed']}",
                seed=reference["seed"],
                acquisition_root=snapshot_root,
                output_dir=output_dir,
                data_hashes=data_hashes,
                model_id="remerge-pos-tnt-v1",
                tokenizer_id="canonical-boundaries-v1",
            )
            for repetition in repetitions:
                candidate = report["candidate"] | {
                    "artifact_sha256": repetition["artifact_sha256"],
                    "artifact_bytes": repetition["artifact_bytes"],
                    "artifact_path": str(repetition["artifact_path"]),
                }
                record = report | {
                    "candidate": candidate,
                    "grid_index": grid_index,
                    "seed": repetition["seed"],
                    "config": raw_config,
                    "repetition_kind": "deterministic",
                    "artifact_identity_verified_against_seed": reference["seed"],
                    "harness_evidence_reused_from_seed": reference["seed"],
                    "timing_evidence": "single common-harness run for byte-identical deterministic repetitions",
                }
                record["improvements_vs_c2"] = _improvements(record, c2_report)
                record["decision"], record["rejection_reasons"] = _pilot_decision(
                    record, c2_report
                )
                results.append(record)
    selected = _selected_grid(results)
    return {
        "schema_version": 1,
        "status": selected["decision"],
        "protected_final_evaluated": False,
        "development_only": True,
        "candidate": {
            "model_id": "remerge-pos-tnt-v1",
            "tokenizer_id": "canonical-boundaries-v1",
        },
        "protocol": {
            "grid": GRID,
            "seeds": SEEDS,
            "repetitions": "deterministic; not independent evidence",
            "harness_runs": "one per grid after exact byte-identity verification",
            "common_harness": str(POS_TAGGER / "bakeoff.py"),
        },
        "provenance": {
            "train_sha256": data_hashes["train"],
            "dev_sha256": data_hashes["dev"],
            "source_revision": _source_revision(),
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "c2_input_path": str(c2_artifact),
            "c2_input_sha256": C2_ARTIFACT_SHA256,
        },
        "c2": c2_report,
        "selected": {
            "grid_index": selected["grid_index"],
            "seed": selected["seed"],
            "artifact_sha256": selected["candidate"]["artifact_sha256"],
            "artifact_path": selected["candidate"]["artifact_path"],
            "decision": selected["decision"],
            "tie_rule": "retain, overall accuracy, macro F1, OOV accuracy, ambiguous accuracy, lower grid index",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--c2-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = {"status": "reject", "protected_final_evaluated": False}
    try:
        evidence = evaluate(
            arguments.acquisition_root, arguments.output_dir, arguments.c2_artifact
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

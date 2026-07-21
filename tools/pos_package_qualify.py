#!/usr/bin/env python3
"""Qualify the installed EWT-only ``remerge-pos`` candidate under Q1.

Preparation is development-only and writes a frozen external registration. The
ordinary four-argument command validates that registration before consuming the
single authorized protected-final run.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from typing import Any, cast
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools/pos_package_probe.py"
MANIFEST_PATH = ROOT / "tests/pos/evaluation/manifest.json"
PACKAGE = ROOT / "packages/remerge-pos"
PACKAGE_SOURCE = PACKAGE / "src/remerge_pos"
VERIFY_DISTRIBUTION = PACKAGE / "tests/verify_distribution.py"
PACKAGING_SMOKE = PACKAGE / "tests/packaging_smoke.py"
C2_ADAPTER = ROOT / "experiments/pos-tagger/c2_adapter.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remerge_pos import UPOS_TAGS  # noqa: E402
from remerge_pos.training import (  # noqa: E402
    SELECTED_CONFIG,
    SELECTED_SEED,
)
from tests.pos.evaluation.loader import (  # noqa: E402
    Sentence,
    load_manifest,
    load_tagged_split,
)


ORACLE_VERSION = 1
PILOT_ARTIFACT_SHA256 = (
    "a394204c44c737ba9c60cc135b3176f5f86bd5728fd0ad68c5fe156384ee748d"
)
PILOT_REPORT_SHA256 = "ac1012332573355ce207ff99d6630532cee7caebb60a0e5bb1db5d65735d2c04"
C2_ARTIFACT_SHA256 = "3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d"
EXPECTED_DEV_METRICS = {
    "overall_accuracy": 0.9413423575860124,
    "macro_f1": 0.9093299376922143,
    "oov_accuracy": 0.7842652152399802,
    "ambiguous_accuracy": 0.9394268851440553,
}
PILOT_THROUGHPUT_TOKENS_PER_SECOND = 8111.315402452354
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
HEADER = struct.Struct("<8sII32s")
SOURCE_PATHS = (
    Path("tools/pos_package_qualify.py"),
    Path("tools/pos_package_probe.py"),
    Path("packages/remerge-pos/pyproject.toml"),
    Path("packages/remerge-pos/src/remerge_pos/__init__.py"),
    Path("packages/remerge-pos/src/remerge_pos/_api.py"),
    Path("packages/remerge-pos/src/remerge_pos/_perceptron.py"),
    Path("packages/remerge-pos/src/remerge_pos/training.py"),
    Path("packages/remerge-pos/tests/verify_distribution.py"),
    Path("packages/remerge-pos/tests/packaging_smoke.py"),
    Path("tests/pos/evaluation/loader.py"),
    Path("tests/pos/evaluation/manifest.json"),
    Path("tests/pos/test_package_qualification.py"),
    Path("docs/pos_package_acceptance_contract.md"),
    Path("experiments/pos-tagger/c2_adapter.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("prek.toml"),
)


class QualificationError(RuntimeError):
    """A frozen qualification invariant failed."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run(command: Sequence[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    result = {
        "argv": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode:
        raise QualificationError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-2000:]}"
        )
    return result


def _source_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise QualificationError("qualification requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise QualificationError(f"qualification source files missing: {missing}")
    return {str(path): _sha(ROOT / path) for path in SOURCE_PATHS}


def _split_path(manifest: Mapping[str, Any], root: Path, split: str) -> Path:
    details = manifest["splits"][split]
    source = manifest["sources"][details["source"]]
    return root / source["relative_root"] / details["path"]


def _forms(sentences: Sequence[Sentence]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(token.form for token in sentence.tokens) for sentence in sentences
    )


def _training_rows(
    sentences: Sequence[Sentence],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        tuple((token.form, token.upos) for token in sentence.tokens)
        for sentence in sentences
    )


def _validate_predictions(
    expected: Sequence[Sequence[str]], actual: object
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise QualificationError("candidate changed the sentence count")
    checked = []
    for index, (sentence, tags) in enumerate(zip(expected, actual, strict=True)):
        if not isinstance(tags, (list, tuple)) or len(tags) != len(sentence):
            raise QualificationError(
                f"candidate changed token count in sentence {index}"
            )
        if any(tag not in UPOS_TAGS for tag in tags):
            raise QualificationError(
                f"candidate emitted invalid UPOS in sentence {index}"
            )
        checked.append(tuple(tags))
    return tuple(checked)


def _quality_metrics(
    train_sentences: Sequence[Sentence],
    evaluated: Sequence[Sentence],
    predictions: Sequence[Sequence[str]],
) -> dict[str, Any]:
    checked = _validate_predictions(_forms(evaluated), predictions)
    train_tags: dict[str, set[str]] = defaultdict(set)
    for sentence in train_sentences:
        for token in sentence.tokens:
            train_tags[token.form].add(token.upos)
    gold = [token for sentence in evaluated for token in sentence.tokens]
    predicted = [tag for sentence in checked for tag in sentence]
    pairs = [(token.upos, tag) for token, tag in zip(gold, predicted, strict=True)]
    labels = tuple(UPOS_TAGS)

    def accuracy(items: Sequence[tuple[str, str]]) -> float | None:
        return (
            None
            if not items
            else sum(gold == guess for gold, guess in items) / len(items)
        )

    def f1(label: str) -> float:
        true_positive = sum(g == label and p == label for g, p in pairs)
        false_positive = sum(g != label and p == label for g, p in pairs)
        false_negative = sum(g == label and p != label for g, p in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if not denominator else 2 * true_positive / denominator

    domains: dict[str, list[tuple[str, str]]] = defaultdict(list)
    oov = []
    ambiguous = []
    for token, guess in zip(gold, predicted, strict=True):
        pair = token.upos, guess
        domains[token.domain].append(pair)
        if token.form not in train_tags:
            oov.append(pair)
        if len(train_tags.get(token.form, ())) >= 2:
            ambiguous.append(pair)
    return {
        "token_count": len(gold),
        "overall_accuracy": accuracy(pairs),
        "macro_f1": statistics.fmean(f1(label) for label in labels),
        "oov_accuracy": accuracy(oov),
        "oov_token_count": len(oov),
        "ambiguous_accuracy": accuracy(ambiguous),
        "ambiguous_token_count": len(ambiguous),
        "per_domain_accuracy": {
            domain: accuracy(items) for domain, items in sorted(domains.items())
        },
        "per_upos_f1": {
            label: {
                "count": sum(token.upos == label for token in gold),
                "f1": f1(label),
            }
            for label in labels
        },
    }


def _candidate_probe(
    python: Path,
    artifact: Path,
    sentences: Sequence[Sequence[str]],
    work: Path,
    name: str,
    *,
    warmups: int = 3,
    repetitions: int = 15,
) -> dict[str, Any]:
    input_path = work / f"{name}.input.json"
    output_path = work / f"{name}.output.json"
    _write_json(input_path, sentences)
    _run(
        [
            str(python),
            "-I",
            str(PROBE),
            "--artifact",
            str(artifact),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--warmups",
            str(warmups),
            "--repetitions",
            str(repetitions),
        ],
        cwd=work,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _api_controls(python: Path, artifact: Path, work: Path) -> dict[str, Any]:
    output = work / "api-controls.json"
    _run(
        [
            str(python),
            "-I",
            str(PROBE),
            "--artifact",
            str(artifact),
            "--output",
            str(output),
            "--api-controls",
        ],
        cwd=work,
    )
    return json.loads(output.read_text(encoding="utf-8"))["controls"]


def _installed_training(
    python: Path,
    sentences: Sequence[Sentence],
    artifact: Path,
    work: Path,
) -> dict[str, Any]:
    input_path = work / "selected-training-input.json"
    output_path = work / "selected-training-evidence.json"
    _write_json(input_path, _training_rows(sentences))
    _run(
        [
            str(python),
            "-I",
            str(PROBE),
            "--train-input",
            str(input_path),
            "--train-output",
            str(artifact),
            "--output",
            str(output_path),
        ],
        cwd=work,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _distribution_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return sorted(name for name in archive.namelist() if not name.endswith("/"))
    with tarfile.open(path, "r:gz") as archive:
        return sorted(member.name for member in archive if member.isfile())


def _distribution_controls(
    python: Path, wheel: Path, sdist: Path, work: Path
) -> dict[str, Any]:
    controls = {}
    for name, omit_license, add_corpus in (
        ("missing-license", True, False),
        ("corpus-leak", False, True),
    ):
        mutated = work / f"{name}.whl"
        with (
            zipfile.ZipFile(wheel) as source,
            zipfile.ZipFile(
                mutated, "w", compression=zipfile.ZIP_DEFLATED
            ) as destination,
        ):
            for member in source.infolist():
                if omit_license and member.filename.endswith("/licenses/LICENSE"):
                    continue
                destination.writestr(member, source.read(member.filename))
            if add_corpus:
                destination.writestr("corpus/train.conllu", "1\tsecret\t_\tNOUN\n")
        completed = subprocess.run(
            [str(python), str(VERIFY_DISTRIBUTION), str(mutated), str(sdist)],
            cwd=work,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            raise QualificationError(
                f"distribution control {name!r} unexpectedly passed"
            )
        controls[name] = {
            "returncode": completed.returncode,
            "error": completed.stderr.strip().splitlines()[-1],
        }
    return controls


def _venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _exercise_distributions(work: Path, wheel: Path, sdist: Path) -> dict[str, Any]:
    verifier = _run([sys.executable, str(VERIFY_DISTRIBUTION), str(wheel), str(sdist)])
    controls = _distribution_controls(Path(sys.executable), wheel, sdist, work)
    environments = {}
    for name, source in (("wheel", wheel), ("sdist", sdist)):
        environment = work / f"{name}-environment"
        _run(["uv", "venv", "--python", sys.executable, str(environment)])
        python = _venv_python(environment)
        _run(["uv", "pip", "install", "--python", str(python), str(source)])
        smoke = _run([str(python), "-I", str(PACKAGING_SMOKE)], cwd=work)
        offline_program = (
            "import http.client,os,socket,subprocess,urllib.request; "
            "blocked=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network')); "
            "socket.create_connection=blocked; socket.socket=blocked; "
            "urllib.request.urlopen=blocked; "
            "http.client.HTTPConnection.connect=blocked; "
            "http.client.HTTPSConnection.connect=blocked; "
            "subprocess.Popen=blocked; os.system=blocked; "
            "import remerge_pos; assert callable(remerge_pos.load_model)"
        )
        offline = _run([str(python), "-I", "-c", offline_program], cwd=work)
        environments[name] = {
            "python": str(python),
            "smoke": smoke,
            "offline_import": offline,
        }
    return {
        "verifier": verifier,
        "negative_controls": controls,
        "wheel": {
            "path": str(wheel),
            "sha256": _sha(wheel),
            "files": _distribution_names(wheel),
        },
        "sdist": {
            "path": str(sdist),
            "sha256": _sha(sdist),
            "files": _distribution_names(sdist),
        },
        "environments": environments,
    }


def _build_distributions(work: Path, retain_dir: Path | None = None) -> dict[str, Any]:
    distribution = work / "dist"
    distribution.mkdir()
    build = _run(
        [
            "uv",
            "build",
            "--package",
            "remerge-pos",
            "--wheel",
            "--sdist",
            "--out-dir",
            str(distribution),
        ]
    )
    wheel = next(distribution.glob("*.whl"))
    sdist = next(distribution.glob("*.tar.gz"))
    if retain_dir is not None:
        retain_dir.mkdir(parents=True, exist_ok=True)
        retained_wheel = retain_dir / wheel.name
        retained_sdist = retain_dir / sdist.name
        shutil.copy2(wheel, retained_wheel)
        shutil.copy2(sdist, retained_sdist)
        wheel, sdist = retained_wheel, retained_sdist
    return {"build": build} | _exercise_distributions(work, wheel, sdist)


def _artifact_payload(artifact: bytes) -> dict[str, Any]:
    _magic, raw_size, _compressed_size, _digest = HEADER.unpack_from(artifact)
    raw = zlib.decompress(artifact[HEADER.size :])
    if len(raw) != raw_size:
        raise QualificationError("pilot artifact payload size is inconsistent")
    return json.loads(raw)


def _encode_payload(payload: object) -> bytes:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    compressed = zlib.compress(raw, level=9)
    return (
        HEADER.pack(
            b"RMSP0001", len(raw), len(compressed), hashlib.sha256(raw).digest()
        )
        + compressed
    )


def _duplicate_key_artifact(artifact: bytes) -> bytes:
    payload = _artifact_payload(artifact)
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    raw = raw.replace('"schema":', '"schema":"duplicate","schema":', 1).encode()
    compressed = zlib.compress(raw, level=9)
    return (
        HEADER.pack(
            b"RMSP0001", len(raw), len(compressed), hashlib.sha256(raw).digest()
        )
        + compressed
    )


def _artifact_mutations(artifact: bytes) -> dict[str, bytes | None]:
    payload = _artifact_payload(artifact)
    wrong_schema = json.loads(json.dumps(payload))
    wrong_schema["schema"] = "RMSP9999"
    invalid_tags = json.loads(json.dumps(payload))
    invalid_tags["tags"][0] = "INVALID"
    invalid_index = json.loads(json.dumps(payload))
    invalid_index["feature_weights"][0][1] = 99
    non_finite = json.loads(json.dumps(payload))
    non_finite["feature_weights"][0][2] = float("nan")
    aggregate_shape = json.loads(json.dumps(payload))
    aggregate_shape["config"]["feature_buckets"] = 1024
    aggregate_shape["feature_weights"][0][0] = 1024
    checksum = bytearray(artifact)
    checksum[16] ^= 1
    compression = bytearray(artifact)
    compression[-1] ^= 1
    magic, _raw_size, compressed_size, digest = HEADER.unpack_from(artifact)
    zip_bomb = (
        HEADER.pack(magic, MAX_PAYLOAD_BYTES + 1, compressed_size, digest)
        + artifact[HEADER.size :]
    )
    return {
        "missing": None,
        "truncated": artifact[:-1],
        "trailing": artifact + b"trailing",
        "checksum": bytes(checksum),
        "compression": bytes(compression),
        "oversize": artifact + bytes(MAX_ARTIFACT_BYTES + 1 - len(artifact)),
        "zip-bomb": zip_bomb,
        "duplicate-key": _duplicate_key_artifact(artifact),
        "wrong-schema": _encode_payload(wrong_schema),
        "invalid-tag": _encode_payload(invalid_tags),
        "invalid-index": _encode_payload(invalid_index),
        "non-finite": _encode_payload(non_finite),
        "aggregate-shape": _encode_payload(aggregate_shape),
    }


def _artifact_controls(python: Path, artifact_path: Path, work: Path) -> dict[str, Any]:
    artifact = artifact_path.read_bytes()
    results = {}
    program = "from remerge_pos import Tagger; import sys; Tagger.load(sys.argv[1])"
    for name, contents in _artifact_mutations(artifact).items():
        path = work / f"control-{name}.rmsp"
        if contents is not None:
            path.write_bytes(contents)
        completed = subprocess.run(
            [str(python), "-I", "-c", program, str(path)],
            cwd=work,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            raise QualificationError(f"artifact control {name!r} unexpectedly loaded")
        results[name] = {
            "returncode": completed.returncode,
            "error": completed.stderr.strip().splitlines()[-1],
        }
    return results


def _assert_exact_parity(expected: object, actual: object) -> None:
    if actual != expected:
        raise QualificationError("complete prediction tensor differs")


def _assert_input_digest(expected_sha256: str, actual: object) -> None:
    if _json_sha(actual) != expected_sha256:
        raise QualificationError("registered input digest differs")


def _expected_gate_failure(name: str, expression: str) -> dict[str, Any]:
    program = (
        "import runpy; "
        f"q=runpy.run_path({str(Path(__file__).resolve())!r}); "
        f"q[{expression.partition('(')[0]!r}]({expression.partition('(')[2]}"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        raise QualificationError(f"negative gate control {name!r} unexpectedly passed")
    return {
        "returncode": completed.returncode,
        "error": completed.stderr.strip().splitlines()[-1],
    }


def _negative_gate_controls() -> dict[str, Any]:
    forms = (("one", "two"), ("three",))
    predictions = (("NOUN", "VERB"), ("DET",))
    inserted = (("NOUN", "VERB", "DET"), ("DET",))
    return {
        "rotated-output-parity": _expected_gate_failure(
            "rotated-output-parity",
            '_assert_exact_parity((("NOUN",),), (("VERB",),))',
        ),
        "dropped-sentence": _expected_gate_failure(
            "dropped-sentence",
            f"_validate_predictions({forms!r}, {predictions[:-1]!r})",
        ),
        "inserted-token": _expected_gate_failure(
            "inserted-token",
            f"_validate_predictions({forms!r}, {inserted!r})",
        ),
        "reordered-input": _expected_gate_failure(
            "reordered-input",
            f"_assert_input_digest({_json_sha(forms)!r}, {tuple(reversed(forms))!r})",
        ),
        "dropped-input": _expected_gate_failure(
            "dropped-input",
            f"_assert_input_digest({_json_sha(forms)!r}, {forms[:-1]!r})",
        ),
        "inserted-input": _expected_gate_failure(
            "inserted-input",
            f"_assert_input_digest({_json_sha(forms)!r}, {(forms + (forms[-1],))!r})",
        ),
        "deliberately-slowed-performance": _expected_gate_failure(
            "deliberately-slowed-performance",
            "_assert_performance_gate(7200.0, 0.0, 0.1, 0.0, 1048576)",
        ),
    }


def _alignment_controls(
    forms: tuple[tuple[str, ...], ...], predictions: tuple[tuple[str, ...], ...]
) -> dict[str, str]:
    controls: dict[str, object] = {
        "dropped-sentence": predictions[:-1],
        "inserted-sentence": predictions + (predictions[-1],),
        "dropped-token": (predictions[0][:-1],) + predictions[1:],
        "inserted-token": (predictions[0] + ("NOUN",),) + predictions[1:],
        "invalid-tag": (("INVALID",) + predictions[0][1:],) + predictions[1:],
    }
    results = {}
    for name, value in controls.items():
        try:
            _validate_predictions(forms, value)
        except QualificationError as error:
            results[name] = f"rejected: {error}"
        else:
            raise QualificationError(f"alignment control {name!r} unexpectedly passed")
    return results | _negative_gate_controls()


def _source_safety() -> dict[str, Any]:
    forbidden_imports = {"pickle", "socket", "urllib", "http", "requests", "subprocess"}
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    forbidden_attributes = {
        "Popen",
        "connect",
        "create_connection",
        "open_connection",
        "system",
        "urlopen",
    }
    findings = []
    for path in sorted(PACKAGE_SOURCE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.partition(".")[0] in forbidden_imports:
                        findings.append(f"{path.name}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.partition(".")[0] in forbidden_imports:
                    findings.append(f"{path.name}: imports {node.module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in forbidden_calls:
                    findings.append(f"{path.name}: calls {node.func.id}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden_attributes:
                    findings.append(f"{path.name}: calls .{node.func.attr}")
    if findings:
        raise QualificationError(f"unsafe package source operations: {findings}")
    return {
        "passed": True,
        "forbidden_operations": sorted(
            forbidden_imports | forbidden_calls | forbidden_attributes
        ),
    }


def _measurement(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("measurement requires at least one sample")
    ordered = sorted(values)
    median = statistics.median(ordered)
    if len(ordered) == 1:
        iqr = 0.0
    else:
        lower = ordered[: len(ordered) // 2]
        upper = ordered[(len(ordered) + 1) // 2 :]
        iqr = statistics.median(upper) - statistics.median(lower)
    return {
        "seconds": values,
        "median_seconds": median,
        "iqr_over_median": iqr / median if median else 0.0,
    }


def _isolated_load_attempt(python: Path, artifact: Path, work: Path) -> dict[str, Any]:
    program = (
        "import json,resource,sys,time; "
        "before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
        "started=time.perf_counter(); "
        "from remerge_pos import Tagger; tagger=Tagger.load(sys.argv[1]); "
        "elapsed=time.perf_counter()-started; "
        "after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
        "unit=1 if sys.platform=='darwin' else 1024; "
        "print(json.dumps({'seconds':elapsed,'rss':max(0,after-before)*unit,'sha':tagger.artifact_sha256}))"
    )
    samples = []
    for _ in range(15):
        completed = subprocess.run(
            [str(python), "-I", "-c", program, str(artifact)],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        samples.append(json.loads(completed.stdout))
    if {sample["sha"] for sample in samples} != {PILOT_ARTIFACT_SHA256}:
        raise QualificationError("isolated loads changed artifact identity")
    return {
        "load": _measurement([sample["seconds"] for sample in samples]),
        "incremental_peak_rss": {
            "samples_bytes": [sample["rss"] for sample in samples],
            "maximum_bytes": max(sample["rss"] for sample in samples),
            "median_bytes": statistics.median(sample["rss"] for sample in samples),
        },
    }


def _isolated_load(python: Path, artifact: Path, work: Path) -> dict[str, Any]:
    attempts = []
    for _ in range(2):
        attempt = _isolated_load_attempt(python, artifact, work)
        attempts.append(attempt)
        if attempt["load"]["iqr_over_median"] <= 0.10:
            break
    return attempts[-1] | {"attempts": attempts}


def _synthetic_forms() -> tuple[tuple[str, ...], ...]:
    vocabulary = (
        "The",
        "quick",
        "fox",
        "jumps",
        "over",
        "seven",
        "bright",
        "lamps",
        ".",
        "I",
        "watch",
        "quietly",
        "today",
        "and",
        "smile",
    )
    generator = random.Random(20260720)
    return tuple(
        tuple(generator.choice(vocabulary) for _ in range(25)) for _ in range(400)
    )


def _performance_gate_failures(
    throughput: float,
    inference_iqr_over_median: float,
    load_median_seconds: float,
    load_iqr_over_median: float,
    peak_rss_bytes: int,
) -> list[str]:
    failures = []
    if inference_iqr_over_median > 0.10:
        failures.append("warm inference remained noisy after one retry")
    if throughput < PILOT_THROUGHPUT_TOKENS_PER_SECOND * 0.90:
        failures.append("throughput regressed by more than 10% from frozen reference")
    if load_median_seconds > 0.250:
        failures.append("cold-load median exceeds 250 ms")
    if load_iqr_over_median > 0.10:
        failures.append("cold-load timing IQR/median exceeds 10%")
    if peak_rss_bytes > 128 * 1024 * 1024:
        failures.append("incremental peak RSS exceeds 128 MiB")
    return failures


def _assert_performance_gate(
    throughput: float,
    inference_iqr_over_median: float,
    load_median_seconds: float,
    load_iqr_over_median: float,
    peak_rss_bytes: int,
) -> None:
    failures = _performance_gate_failures(
        throughput,
        inference_iqr_over_median,
        load_median_seconds,
        load_iqr_over_median,
        peak_rss_bytes,
    )
    if failures:
        raise QualificationError("; ".join(failures))


def _performance(
    python: Path,
    artifact: Path,
    dev_forms: tuple[tuple[str, ...], ...],
    work: Path,
) -> dict[str, Any]:
    attempts = []
    for attempt in range(2):
        evidence = _candidate_probe(
            python, artifact, dev_forms, work, f"performance-{attempt}"
        )
        attempts.append(evidence["timing"])
        if evidence["timing"]["iqr_over_median"] <= 0.10:
            break
    timing = attempts[-1]
    tokens = sum(map(len, dev_forms))
    throughput = tokens / timing["median_seconds"]
    load = _isolated_load(python, artifact, work)
    synthetic = _synthetic_forms()
    synthetic_evidence = _candidate_probe(
        python, artifact, synthetic, work, "synthetic-performance"
    )
    failures = _performance_gate_failures(
        throughput,
        timing["iqr_over_median"],
        load["load"]["median_seconds"],
        load["load"]["iqr_over_median"],
        load["incremental_peak_rss"]["maximum_bytes"],
    )
    slowed = PILOT_THROUGHPUT_TOKENS_PER_SECOND * 0.89
    try:
        _assert_performance_gate(slowed, 0.0, 0.1, 0.0, 1024 * 1024)
    except QualificationError:
        pass
    else:
        raise QualificationError("deliberately slowed performance control passed")
    return {
        "passed": not failures,
        "failures": failures,
        "dev": {
            "attempts": attempts,
            "token_count": tokens,
            "tokens_per_second": throughput,
            "reference_tokens_per_second": PILOT_THROUGHPUT_TOKENS_PER_SECOND,
            "minimum_tokens_per_second": PILOT_THROUGHPUT_TOKENS_PER_SECOND * 0.90,
        },
        "load": load,
        "synthetic": {
            "generator": "python-random-v1 choice from fixed 15-form vocabulary",
            "seed": 20260720,
            "sentence_count": len(synthetic),
            "token_count": sum(map(len, synthetic)),
            "input_sha256": _json_sha(synthetic),
            "timing": synthetic_evidence["timing"],
        },
        "negative_control": {
            "deliberately_slowed_tokens_per_second": slowed,
            "expected": "reject",
            "observed": "reject",
        },
    }


def _gate(name: str, details: object, failures: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "name": name,
        "passed": not failures,
        "failures": list(failures),
        "details": details,
    }


def _model_material(
    registration: Path, artifact: Path, train_hash: str
) -> dict[str, Any]:
    stem = registration.with_suffix("")
    card = stem.with_name(stem.name + ".model-card.md")
    manifest = stem.with_name(stem.name + ".model-manifest.json")
    card.write_text(
        "# Experimental English UPOS model\n\n"
        "Averaged structured perceptron trained only on UD English EWT r2.10. "
        "The model artifact is separate from the MIT code package and is treated "
        "conservatively as CC BY-SA 4.0. It is experimental, caller-tokenized, "
        "English-web-domain evidence and is not a broad English quality claim.\n\n"
        "Source: https://github.com/UniversalDependencies/UD_English-EWT\n\n"
        f"Training input SHA-256: `{train_hash}`\n\n"
        f"Artifact SHA-256: `{_sha(artifact)}`\n",
        encoding="utf-8",
    )
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "artifact_sha256": _sha(artifact),
            "license": "CC BY-SA 4.0",
            "source": "UD English EWT r2.10",
            "source_repository": "https://github.com/UniversalDependencies/UD_English-EWT.git",
            "source_revision": "b33472d3ce50a62056d6057f1b8a723e9d211176",
            "training_input_sha256": train_hash,
            "raw_corpus_in_artifact": False,
            "gold_labels_in_artifact": False,
        },
    )
    return {
        "model_card_path": str(card),
        "model_card_sha256": _sha(card),
        "model_manifest_path": str(manifest),
        "model_manifest_sha256": _sha(manifest),
    }


def _development(
    acquisition_root: Path,
    pilot_artifact: Path,
    pilot_report: Path,
    c2_artifact: Path,
    output: Path,
    registration: Path,
) -> dict[str, Any]:
    revision = _source_revision()
    if _sha(pilot_artifact) != PILOT_ARTIFACT_SHA256:
        raise QualificationError("pilot artifact does not match the frozen digest")
    if _sha(pilot_report) != PILOT_REPORT_SHA256:
        raise QualificationError("pilot report does not match the frozen digest")
    if _sha(c2_artifact) != C2_ARTIFACT_SHA256:
        raise QualificationError("c2 artifact does not match the frozen digest")
    manifest = load_manifest()
    train_sentences = load_tagged_split(manifest, acquisition_root, "train")
    dev_sentences = load_tagged_split(manifest, acquisition_root, "dev")
    splits = cast(dict[str, dict[str, Any]], manifest["splits"])
    data_hashes = {
        "train": _sha(_split_path(manifest, acquisition_root, "train")),
        "dev": _sha(_split_path(manifest, acquisition_root, "dev")),
        "final_registered_not_read": splits["final"]["sha256"],
    }
    with TemporaryDirectory(prefix="remerge-pos-qualification-dev-") as directory:
        work = Path(directory)
        retained_distributions = registration.resolve().with_name(
            registration.stem + ".distributions"
        )
        distributions = _build_distributions(work, retained_distributions)
        wheel_python = Path(distributions["environments"]["wheel"]["python"])
        rebuild = work / "rebuilt-selected.rmsp"
        training_evidence = _installed_training(
            wheel_python, train_sentences, rebuild, work
        )
        digest = training_evidence["artifact_sha256"]
        training_seconds = training_evidence["training_seconds"]
        rebuild_equal = rebuild.read_bytes() == pilot_artifact.read_bytes()
        if (
            digest != PILOT_ARTIFACT_SHA256
            or not rebuild_equal
            or training_evidence["config"] != asdict(SELECTED_CONFIG)
            or training_evidence["seed"] != SELECTED_SEED
        ):
            raise QualificationError(
                "installed selected recipe did not reproduce the pilot bytes"
            )
        dev_forms = _forms(dev_sentences)
        dev_input_sha256 = _json_sha(dev_forms)
        _assert_input_digest(dev_input_sha256, dev_forms)
        probe = _candidate_probe(
            wheel_python, pilot_artifact, dev_forms, work, "dev-parity"
        )
        predictions = _validate_predictions(dev_forms, probe["predictions"])
        _assert_exact_parity(predictions, predictions)
        rebuilt_probe = _candidate_probe(
            wheel_python,
            rebuild,
            dev_forms,
            work,
            "rebuilt-dev-parity",
            warmups=0,
            repetitions=1,
        )
        rebuilt_predictions = _validate_predictions(
            dev_forms, rebuilt_probe["predictions"]
        )
        try:
            _assert_exact_parity(predictions, rebuilt_predictions)
        except QualificationError:
            exact_prediction_parity = False
        else:
            exact_prediction_parity = True
        metrics = _quality_metrics(train_sentences, dev_sentences, predictions)
        metric_failures = [
            f"{name} differs from frozen pilot"
            for name, expected in EXPECTED_DEV_METRICS.items()
            if not math.isclose(metrics[name], expected, rel_tol=0.0, abs_tol=1e-12)
        ]
        parity_failures = []
        if not exact_prediction_parity:
            parity_failures.append("rebuilt and pilot tag tensors differ")
        if metrics["token_count"] != 24822:
            parity_failures.append("development token count differs from 24,822")
        api = _api_controls(wheel_python, pilot_artifact, work)
        artifacts = _artifact_controls(wheel_python, pilot_artifact, work)
        alignment = _alignment_controls(dev_forms, predictions)
        performance = _performance(wheel_python, pilot_artifact, dev_forms, work)
        safety = _source_safety()
        repository_checks = [
            _run(
                [
                    "uv",
                    "run",
                    "--package",
                    "remerge-pos",
                    "--no-sync",
                    "pytest",
                    "-q",
                    "packages/remerge-pos/tests",
                ]
            ),
            _run(["uv", "run", "--no-sync", "pytest", "-q", "-m", "not performance"]),
            _run(["uv", "run", "--no-sync", "prek", "run", "--all-files"]),
        ]
        gates = {
            "extraction_parity": _gate(
                "extraction_parity",
                {
                    "artifact_sha256": digest,
                    "artifact_bytes": rebuild.stat().st_size,
                    "rebuilt_bytes_equal": rebuild_equal,
                    "prediction_tensor_equal": exact_prediction_parity,
                    "metrics": metrics,
                },
                parity_failures + metric_failures,
            ),
            "determinism": _gate(
                "determinism",
                {"repetitions": 15, "probe_timing": probe["timing"]},
            ),
            "api_alignment": _gate(
                "api_alignment", {"api": api, "alignment": alignment}
            ),
            "artifact_safety": _gate(
                "artifact_safety", {"controls": artifacts, "source_scan": safety}
            ),
            "reproducible_export": _gate(
                "reproducible_export",
                {
                    "config": asdict(SELECTED_CONFIG),
                    "seed": SELECTED_SEED,
                    "training_seconds": training_seconds,
                    "training_input_sha256": data_hashes["train"],
                    "development_input_sha256": dev_input_sha256,
                    "artifact_sha256": digest,
                    "execution_boundary": "training module imported from installed wheel",
                },
            ),
            "distribution": _gate("distribution", distributions),
            "performance": _gate(
                "performance",
                performance,
                performance["failures"],
            ),
            "root_compatibility": _gate(
                "root_compatibility", {"commands": repository_checks}
            ),
        }
        failures = [name for name, gate in gates.items() if not gate["passed"]]
        report = {
            "schema_version": 1,
            "oracle_version": ORACLE_VERSION,
            "phase": "development-registration",
            "status": "pass" if not failures else "reject",
            "protected_final_attempted": False,
            "protected_final_evaluated": False,
            "candidate": {
                "package": "remerge-pos==0.1.0",
                "model_id": "remerge-pos-structured-perceptron-v1",
                "artifact_sha256": PILOT_ARTIFACT_SHA256,
                "artifact_bytes": pilot_artifact.stat().st_size,
                "config": asdict(SELECTED_CONFIG),
                "seed": SELECTED_SEED,
            },
            "source_revision": revision,
            "source_hashes": _source_hashes(),
            "data_hashes": data_hashes,
            "gates": gates,
            "hard_gate_failures": failures,
            "diagnostics": {
                "development_quality": metrics,
                "training_seconds": training_seconds,
                "mwe": "not evaluated; absence does not gate the standalone package",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "actor": getpass.getuser(),
            },
            "argv": sys.argv,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    _write_json(output, report)
    if report["status"] != "pass":
        return report
    material = _model_material(registration, pilot_artifact, data_hashes["train"])
    registration_value = {
        "schema_version": 1,
        "candidate_id": "remerge-pos-ewt-perceptron-q1",
        "source_revision": revision,
        "source_hashes": report["source_hashes"],
        "package": "remerge-pos==0.1.0",
        "model_id": "remerge-pos-structured-perceptron-v1",
        "artifact_sha256": PILOT_ARTIFACT_SHA256,
        "artifact_path": str(pilot_artifact.resolve()),
        "pilot_report_sha256": PILOT_REPORT_SHA256,
        "pilot_report_path": str(pilot_report.resolve()),
        "c2_artifact_sha256": C2_ARTIFACT_SHA256,
        "c2_artifact_path": str(c2_artifact.resolve()),
        "config": asdict(SELECTED_CONFIG),
        "seed": SELECTED_SEED,
        "data_hashes": data_hashes,
        "development_report_path": str(output.resolve()),
        "development_report_sha256": _sha(output),
        "hard_gates_passed": True,
        "performance_reference_tokens_per_second": PILOT_THROUGHPUT_TOKENS_PER_SECOND,
        "protected_final_authorization": "one registered candidate, one run, diagnostic only",
        "model_material": material,
        "distributions": {
            "wheel_path": distributions["wheel"]["path"],
            "wheel_sha256": distributions["wheel"]["sha256"],
            "sdist_path": distributions["sdist"]["path"],
            "sdist_sha256": distributions["sdist"]["sha256"],
        },
        "prepared_argv": sys.argv,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "actor": getpass.getuser(),
    }
    _write_json(registration, registration_value)
    return report


def _validate_registration(
    path: Path, pilot_artifact: Path
) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read registration: {error}") from error
    required = {
        "schema_version",
        "candidate_id",
        "source_revision",
        "source_hashes",
        "package",
        "model_id",
        "artifact_sha256",
        "artifact_path",
        "pilot_report_sha256",
        "pilot_report_path",
        "c2_artifact_sha256",
        "c2_artifact_path",
        "config",
        "seed",
        "data_hashes",
        "development_report_path",
        "development_report_sha256",
        "hard_gates_passed",
        "model_material",
        "distributions",
        "protected_final_authorization",
        "performance_reference_tokens_per_second",
    }
    if not isinstance(value, dict) or required - value.keys():
        raise QualificationError("registration is malformed or incomplete")
    revision = _source_revision()
    if (
        value["source_revision"] != revision
        or value["source_hashes"] != _source_hashes()
    ):
        raise QualificationError("registered source differs from the clean checkout")
    if (
        value["artifact_sha256"] != PILOT_ARTIFACT_SHA256
        or _sha(pilot_artifact) != PILOT_ARTIFACT_SHA256
    ):
        raise QualificationError("registered pilot artifact identity differs")
    if value["package"] != "remerge-pos==0.1.0":
        raise QualificationError("registered package identity differs")
    if value["model_id"] != "remerge-pos-structured-perceptron-v1":
        raise QualificationError("registered model identity differs")
    if value["config"] != asdict(SELECTED_CONFIG) or value["seed"] != SELECTED_SEED:
        raise QualificationError("registered training recipe differs")
    if (
        value["performance_reference_tokens_per_second"]
        != PILOT_THROUGHPUT_TOKENS_PER_SECOND
    ):
        raise QualificationError("registered performance reference differs")
    if Path(value["artifact_path"]).resolve() != pilot_artifact.resolve():
        raise QualificationError("registered pilot artifact path differs")
    checks = (
        ("pilot_report", PILOT_REPORT_SHA256),
        ("c2_artifact", C2_ARTIFACT_SHA256),
        ("development_report", value["development_report_sha256"]),
    )
    for name, expected in checks:
        material = Path(value[f"{name}_path"])
        if not material.is_file() or _sha(material) != expected:
            raise QualificationError(f"registered {name.replace('_', ' ')} differs")
    for name in ("model_card", "model_manifest"):
        material = Path(value["model_material"][f"{name}_path"])
        expected = value["model_material"][f"{name}_sha256"]
        if not material.is_file() or _sha(material) != expected:
            raise QualificationError(f"registered {name.replace('_', ' ')} differs")
    for name in ("wheel", "sdist"):
        material = Path(value["distributions"][f"{name}_path"])
        expected = value["distributions"][f"{name}_sha256"]
        if not material.is_file() or _sha(material) != expected:
            raise QualificationError(f"registered {name} distribution differs")
    development = json.loads(
        Path(value["development_report_path"]).read_text(encoding="utf-8")
    )
    if development.get("status") != "pass" or value["hard_gates_passed"] is not True:
        raise QualificationError("registered development hard gates did not pass")
    return value, revision


def _consume_final_registration(path: Path, registration_sha256: str) -> Path:
    marker = path.with_name(path.name + ".final-consumed.json")
    try:
        with marker.open("x", encoding="utf-8") as destination:
            json.dump(
                {
                    "registration_sha256": registration_sha256,
                    "consumed_at": datetime.now(timezone.utc).isoformat(),
                    "argv": sys.argv,
                    "actor": getpass.getuser(),
                },
                destination,
                indent=2,
                sort_keys=True,
            )
            destination.write("\n")
    except FileExistsError as error:
        raise QualificationError(
            "protected final registration was already consumed"
        ) from error
    return marker


def _c2_predictions(
    artifact: Path, forms: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    specification = importlib.util.spec_from_file_location("pos_q1_c2", C2_ADAPTER)
    if specification is None or specification.loader is None:
        raise QualificationError("cannot load retained c2 diagnostic adapter")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    candidate = module.C2Adapter(artifact)
    documents = tuple((sentence,) for sentence in forms)
    raw = candidate.tag(documents)
    flattened = tuple(document[0] for document in raw)
    return _validate_predictions(forms, flattened)


def _paired_differences(
    sentences: Sequence[Sentence],
    candidate: Sequence[Sequence[str]],
    baseline: Sequence[Sequence[str]],
) -> dict[str, int]:
    gold = [token.upos for sentence in sentences for token in sentence.tokens]
    left = [tag for sentence in candidate for tag in sentence]
    right = [tag for sentence in baseline for tag in sentence]
    counts = {
        "both_correct": 0,
        "candidate_only_correct": 0,
        "c2_only_correct": 0,
        "both_wrong": 0,
    }
    for expected, candidate_tag, c2_tag in zip(gold, left, right, strict=True):
        candidate_correct = expected == candidate_tag
        c2_correct = expected == c2_tag
        key = (
            "both_correct"
            if candidate_correct and c2_correct
            else "candidate_only_correct"
            if candidate_correct
            else "c2_only_correct"
            if c2_correct
            else "both_wrong"
        )
        counts[key] += 1
    return counts


def _protected_final(
    acquisition_root: Path,
    registration_path: Path,
    pilot_artifact: Path,
    output: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    registration, revision = _validate_registration(registration_path, pilot_artifact)
    registration_sha256 = _sha(registration_path)
    manifest = load_manifest()
    train_sentences = load_tagged_split(manifest, acquisition_root, "train")
    with TemporaryDirectory(prefix="remerge-pos-qualification-final-") as directory:
        work = Path(directory)
        registered_distributions = registration["distributions"]
        distributions = _exercise_distributions(
            work,
            Path(registered_distributions["wheel_path"]),
            Path(registered_distributions["sdist_path"]),
        )
        wheel_python = Path(distributions["environments"]["wheel"]["python"])
        development = json.loads(
            Path(registration["development_report_path"]).read_text(encoding="utf-8")
        )
        marker = _consume_final_registration(registration_path, registration_sha256)
        state.update(
            {
                "phase": "protected-final-authorized",
                "registration_sha256": registration_sha256,
                "consumption_marker": str(marker),
            }
        )
        marker_sha256 = _sha(marker)
        state["consumption_marker_sha256"] = marker_sha256
        partial = {
            "schema_version": 1,
            "oracle_version": ORACLE_VERSION,
            "phase": "protected-final-authorized",
            "status": "in-progress",
            "protected_final_attempted": False,
            "protected_final_evaluated": False,
            "protected_final_role": "required diagnostic for cp7t; no Q1 quality floor",
            "candidate": development["candidate"],
            "registration": {
                "path": str(registration_path),
                "sha256": registration_sha256,
                "source_revision": revision,
                "consumption_marker": str(marker),
                "consumption_marker_sha256": marker_sha256,
            },
            "hard_gates": development["gates"],
            "hard_gate_failures": development["hard_gate_failures"],
            "distribution_preflight": distributions,
            "data_hashes": {
                "train": registration["data_hashes"]["train"],
                "dev": registration["data_hashes"]["dev"],
                "final": "protected read not yet attempted",
            },
            "argv": sys.argv,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.update(
            {
                "phase": partial["phase"],
                "partial_report": partial,
            }
        )
        _write_json(output, partial)

        # From this point onward the run is conservatively recorded as having
        # evaluated protected data even if opening, decoding, or validation fails.
        state.update(
            {
                "attempted": True,
                "evaluated": True,
                "phase": "protected-final-read-attempted",
            }
        )
        partial.update(
            {
                "phase": state["phase"],
                "protected_final_attempted": True,
                "protected_final_evaluated": True,
            }
        )
        _write_json(output, partial)
        final_sentences = load_tagged_split(
            manifest, acquisition_root, "final", allow_final=True
        )
        final_hash = _sha(_split_path(manifest, acquisition_root, "final"))
        state["final_sha256"] = final_hash
        partial["data_hashes"]["final"] = final_hash
        _write_json(output, partial)
        if final_hash != registration["data_hashes"]["final_registered_not_read"]:
            raise QualificationError("protected final bytes differ from registration")

        final_forms = _forms(final_sentences)
        probe = _candidate_probe(
            wheel_python,
            pilot_artifact,
            final_forms,
            work,
            "protected-final",
            warmups=0,
            repetitions=1,
        )
        candidate_predictions = _validate_predictions(final_forms, probe["predictions"])
        candidate_metrics = _quality_metrics(
            train_sentences, final_sentences, candidate_predictions
        )
        state["phase"] = "protected-final-candidate-diagnostic"
        partial["phase"] = state["phase"]
        partial["diagnostics"] = {
            "development_quality": development["diagnostics"]["development_quality"],
            "protected_final_quality": candidate_metrics,
        }
        _write_json(output, partial)
        c2_path = Path(registration["c2_artifact_path"])
        c2_predictions = _c2_predictions(c2_path, final_forms)
        c2_metrics = _quality_metrics(train_sentences, final_sentences, c2_predictions)
        metric_deltas = {
            name: candidate_metrics[name] - c2_metrics[name]
            for name in (
                "overall_accuracy",
                "macro_f1",
                "oov_accuracy",
                "ambiguous_accuracy",
            )
        }
        partial["diagnostics"].update(
            {
                "c2_protected_final_quality": c2_metrics,
                "candidate_minus_c2": metric_deltas,
                "paired_correctness": _paired_differences(
                    final_sentences, candidate_predictions, c2_predictions
                ),
            }
        )
        _write_json(output, partial)
        return {
            "schema_version": 1,
            "oracle_version": ORACLE_VERSION,
            "phase": "protected-final",
            "status": "pass",
            "protected_final_attempted": True,
            "protected_final_evaluated": True,
            "protected_final_role": "required diagnostic for cp7t; no Q1 quality floor",
            "candidate": development["candidate"],
            "registration": {
                "path": str(registration_path),
                "sha256": registration_sha256,
                "source_revision": revision,
                "consumption_marker": str(marker),
                "consumption_marker_sha256": marker_sha256,
            },
            "hard_gates": development["gates"],
            "hard_gate_failures": development["hard_gate_failures"],
            "diagnostics": {
                "development_quality": development["diagnostics"][
                    "development_quality"
                ],
                "protected_final_quality": candidate_metrics,
                "c2_protected_final_quality": c2_metrics,
                "candidate_minus_c2": metric_deltas,
                "paired_correctness": partial["diagnostics"]["paired_correctness"],
                "artifact_bytes": pilot_artifact.stat().st_size,
                "training_seconds": development["diagnostics"]["training_seconds"],
                "mwe": "not evaluated; absence does not gate the standalone package",
            },
            "data_hashes": {
                "train": registration["data_hashes"]["train"],
                "dev": registration["data_hashes"]["dev"],
                "final": final_hash,
            },
            "distribution_preflight": distributions,
            "source_hashes": registration["source_hashes"],
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "actor": getpass.getuser(),
            },
            "argv": sys.argv,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "EWT is English web-domain evidence, not unrestricted English.",
                "Q1 defines no post-hoc numerical quality floor.",
                "Publication and REMERGE exposure remain Human decisions.",
            ],
        }


def _failure_report(error: Exception, *, state: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(cast(dict[str, Any], state.get("partial_report", {})))
    report.update(
        {
            "schema_version": 1,
            "oracle_version": ORACLE_VERSION,
            "phase": f"{state.get('phase', 'qualification')}-failure",
            "status": "reject",
            "protected_final_attempted": bool(state.get("attempted", False)),
            "protected_final_evaluated": bool(state.get("evaluated", False)),
            "failure": f"{type(error).__name__}: {error}",
            "retained_state": {
                key: value for key, value in state.items() if key != "partial_report"
            },
            "argv": sys.argv,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--pilot-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--c2-artifact", type=Path)
    arguments = parser.parse_args()
    protected_state: dict[str, Any] = {
        "phase": "development-registration" if arguments.prepare else "qualification",
        "attempted": False,
        "evaluated": False,
    }
    try:
        if arguments.prepare:
            if arguments.pilot_report is None or arguments.c2_artifact is None:
                parser.error("--prepare requires --pilot-report and --c2-artifact")
            report = _development(
                arguments.acquisition_root,
                arguments.pilot_artifact,
                arguments.pilot_report,
                arguments.c2_artifact,
                arguments.output,
                arguments.registration,
            )
        else:
            report = _protected_final(
                arguments.acquisition_root,
                arguments.registration,
                arguments.pilot_artifact,
                arguments.output,
                protected_state,
            )
        _write_json(arguments.output, report)
    except Exception as error:
        _write_json(
            arguments.output,
            _failure_report(error, state=protected_state),
        )
        raise SystemExit(2) from error
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

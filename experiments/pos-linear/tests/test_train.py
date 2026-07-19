"""Focused contract tests for the deterministic POS trainer reference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
TRAINER = ROOT / "train.py"

CONLLU = """# sent_id = 1
1\tBright\tbright\tADJ\t_\t_\t0\troot\t_\t_
2\tdogs\tdog\tNOUN\t_\t_\t1\tnsubj\t_\t_
3\trun\trun\tVERB\t_\t_\t1\tacl\t_\t_
4\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_

# sent_id = 2
1\t42\t42\tNUM\t_\t_\t0\troot\t_\t_
2\tbright\tbright\tADJ\t_\t_\t1\tamod\t_\t_
3\tdogs\tdog\tNOUN\t_\t_\t1\tnsubj\t_\t_
4\t!\t!\tPUNCT\t_\t_\t1\tpunct\t_\t_
"""


def run_trainer(tmp_path: Path, suffix: str) -> tuple[Path, Path, Path]:
    train = tmp_path / "train.conllu"
    dev = tmp_path / "dev.conllu"
    train.write_text(CONLLU, encoding="utf-8")
    dev.write_text(CONLLU, encoding="utf-8")
    output = tmp_path / f"model-{suffix}.bin"
    report = tmp_path / f"report-{suffix}.json"
    template = tmp_path / f"registration-{suffix}.json"
    command = [
        sys.executable,
        str(TRAINER),
        "--train",
        str(train),
        "--dev",
        str(dev),
        "--output",
        str(output),
        "--report",
        str(report),
        "--registration-template",
        str(template),
        "--epochs",
        "3",
        "--buckets",
        "32",
        "--seed",
        "7",
        "--source-revision",
        "test-revision",
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    return output, report, template


def test_trainer_is_deterministic_and_emits_v1_header(tmp_path: Path) -> None:
    first, first_report, _ = run_trainer(tmp_path, "first")
    second, _, _ = run_trainer(tmp_path, "second")
    assert first.read_bytes() == second.read_bytes()
    header = struct.unpack("<8sHHIHHHHIII", first.read_bytes()[:36])
    assert header[:6] == (b"RMPOS001", 1, 17, 32, 22, 0)
    assert header[-1] == 32 * 17
    report = json.loads(first_report.read_text(encoding="utf-8"))
    assert report["artifact_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert report["dev_accuracy_full_precision_all_labels"] >= 0.0
    assert report["dev_accuracy_quantized_all_labels"] >= 0.0
    assert report["dev_accuracy_quantized_candidate_pruned"] >= 0.0
    assert report["feature_schema"]
    assert report["quantization_scale"] > 0


def test_reports_fast_path_and_template(tmp_path: Path) -> None:
    _, report_path, template_path = run_trainer(tmp_path, "one")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert report["direct_lexical_coverage"] > 0
    assert report["direct_lexical_accuracy"] == 1.0
    assert report["quantization_loss_percentage_points"] <= 0.25
    assert report["candidate_pruning_loss_percentage_points"] <= 0.25
    assert template["artifact_sha256"] == report["artifact_sha256"]
    assert set(template) >= {
        "candidate_id",
        "source_revision",
        "trainer_command",
        "seed",
        "model_id",
        "artifact_sha256",
        "dev_report",
        "final_evaluated",
    }
    assert template["source_revision"] == "test-revision"
    assert template["final_evaluated"] is False


def test_collision_policy_removes_colliding_form_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("pos_linear_test", TRAINER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "fnv1a", lambda value: 1 if value in {"a", "b"} else 2)
    example = module.Example(
        (module.Token("a", 0), module.Token("b", 1), module.Token("c", 2))
    )
    direct, candidates, collisions = module.lexicons((example,))
    assert direct == {2: (2, 1)}
    assert candidates == {}
    assert collisions == 1


def test_final_argument_is_not_accepted(tmp_path: Path) -> None:
    train = tmp_path / "train.conllu"
    train.write_text(CONLLU, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(TRAINER), "--final", str(train)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "usage:" in completed.stderr


def test_features_match_the_rust_binding_contract() -> None:
    spec = importlib.util.spec_from_file_location("pos_linear_features", TRAINER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.features(("ÄBC", "Dog", "I"), 1) == (
        "W=Dog",
        "L=dog",
        "S=Xxx",
        "P1=d",
        "P2=do",
        "P3=dog",
        "P4=dog",
        "U1=g",
        "U2=og",
        "U3=dog",
        "U4=dog",
        "PL=Äbc",
        "PS=XXX",
        "NL=i",
        "NS=X",
        "PC=Äbc|dog",
        "CN=dog|i",
        "F=upper",
        "F=title",
    )


def test_rust_golden_feature_vector() -> None:
    spec = importlib.util.spec_from_file_location("pos_linear_golden", TRAINER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.features(("Hi", "FOO!", "é"), 1) == (
        "W=FOO!",
        "L=foo!",
        "S=XXX!",
        "P1=f",
        "P2=fo",
        "P3=foo",
        "P4=foo!",
        "U1=!",
        "U2=o!",
        "U3=oo!",
        "U4=foo!",
        "PL=hi",
        "PS=Xx",
        "NL=é",
        "NS=x",
        "PC=hi|foo!",
        "CN=foo!|é",
        "F=upper",
    )

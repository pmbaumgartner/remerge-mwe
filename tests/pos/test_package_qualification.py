"""Unit controls for the POS package qualification scripts.

These tests deliberately use tiny, generated artifacts and ``Sentence`` values.
They never acquire or read the protected-final corpus.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from remerge_pos import Tagger
from remerge_pos.training import Config, train, write_artifact
from tests.pos.evaluation.loader import Sentence, Token


ROOT = Path(__file__).parents[2]


def _module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


qualification = _module(
    "pos_package_qualify_test", ROOT / "tools/pos_package_qualify.py"
)
probe = _module("pos_package_probe_test", ROOT / "tools/pos_package_probe.py")


def _sentence(
    document: str, sentence: str, domain: str, rows: tuple[tuple[str, str], ...]
) -> Sentence:
    return Sentence(
        document_id=document,
        sentence_id=sentence,
        tokens=tuple(
            Token(document, sentence, index, form, upos, domain)
            for index, (form, upos) in enumerate(rows)
        ),
    )


@pytest.fixture
def tiny_artifact(tmp_path: Path) -> Path:
    rows = (
        (("The", "DET"), ("watch", "NOUN")),
        (("I", "PRON"), ("watch", "VERB")),
    )
    artifact = tmp_path / "tiny.rmsp"
    write_artifact(
        train(rows, Config(epochs=1, feature_cutoff=1, feature_buckets=1024), 7),
        artifact,
    )
    assert (
        Tagger.load(artifact).artifact_sha256
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    return artifact


def test_quality_metrics_measure_oov_ambiguity_domains_and_macro_f1() -> None:
    train_sentences = (
        _sentence("train", "1", "news", (("watch", "NOUN"),)),
        _sentence("train", "2", "news", (("watch", "VERB"),)),
    )
    evaluated = (
        _sentence("dev", "1", "news", (("The", "DET"), ("watch", "NOUN"))),
        _sentence("dev", "2", "spoken", (("new", "VERB"),)),
    )

    metrics = qualification._quality_metrics(
        train_sentences, evaluated, (("DET", "VERB"), ("VERB",))
    )

    assert metrics["token_count"] == 3
    assert metrics["overall_accuracy"] == pytest.approx(2 / 3)
    assert metrics["oov_token_count"] == 2
    assert metrics["oov_accuracy"] == pytest.approx(1.0)
    assert metrics["ambiguous_token_count"] == 1
    assert metrics["ambiguous_accuracy"] == pytest.approx(0.0)
    assert metrics["per_domain_accuracy"] == {"news": 0.5, "spoken": 1.0}
    assert metrics["per_upos_f1"]["DET"] == {"count": 1, "f1": 1.0}
    assert metrics["per_upos_f1"]["NOUN"] == {"count": 1, "f1": 0.0}
    assert metrics["per_upos_f1"]["VERB"] == {
        "count": 1,
        "f1": pytest.approx(2 / 3),
    }
    assert metrics["macro_f1"] == pytest.approx((1 + 2 / 3) / 17)


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        ((("NOUN",),), "sentence count"),
        ((("NOUN", "VERB", "DET"), ("NOUN",)), "token count"),
        ((("NOUN", "INVALID"), ("NOUN",)), "invalid UPOS"),
    ],
)
def test_qualification_prediction_validation_rejects_alignment_changes(
    actual: object, message: str
) -> None:
    with pytest.raises(qualification.QualificationError, match=message):
        qualification._validate_predictions((("one", "two"), ("three",)), actual)


def test_alignment_controls_reject_shape_tag_and_registered_input_changes() -> None:
    controls = qualification._alignment_controls(
        (("one", "two"), ("three",)), (("NOUN", "VERB"), ("DET",))
    )

    assert set(controls) == {
        "dropped-sentence",
        "inserted-sentence",
        "dropped-token",
        "inserted-token",
        "invalid-tag",
        "rotated-output-parity",
        "reordered-input",
        "dropped-input",
        "inserted-input",
        "deliberately-slowed-performance",
    }
    for value in controls.values():
        if isinstance(value, str):
            assert value.startswith("rejected:")
        else:
            assert value["returncode"] != 0


def test_tiny_artifact_mutations_are_rejected(
    tiny_artifact: Path, tmp_path: Path
) -> None:
    mutations = qualification._artifact_mutations(tiny_artifact.read_bytes())
    assert set(mutations) >= {
        "missing",
        "truncated",
        "trailing",
        "checksum",
        "compression",
        "oversize",
        "zip-bomb",
        "duplicate-key",
        "wrong-schema",
        "invalid-tag",
        "invalid-index",
        "non-finite",
        "aggregate-shape",
    }
    for name, contents in mutations.items():
        path = tmp_path / f"{name}.rmsp"
        if contents is not None:
            path.write_bytes(contents)
        with pytest.raises((OSError, ValueError)):
            Tagger.load(path)


def _registration(tmp_path: Path, artifact: Path) -> dict[str, Any]:
    report = tmp_path / "development.json"
    report.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    pilot_report = tmp_path / "pilot.json"
    pilot_report.write_text("pilot", encoding="utf-8")
    c2_artifact = tmp_path / "c2.rmsp"
    c2_artifact.write_text("c2", encoding="utf-8")
    card = tmp_path / "model-card.md"
    card.write_text("card", encoding="utf-8")
    model_manifest = tmp_path / "model-manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")
    wheel = tmp_path / "tiny.whl"
    wheel.write_text("wheel", encoding="utf-8")
    sdist = tmp_path / "tiny.tar.gz"
    sdist.write_text("sdist", encoding="utf-8")
    return {
        "schema_version": 1,
        "candidate_id": "tiny",
        "source_revision": "clean-revision",
        "source_hashes": {"tiny": "source"},
        "package": "remerge-pos==0.1.0",
        "model_id": "remerge-pos-structured-perceptron-v1",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_path": str(artifact),
        "pilot_report_sha256": hashlib.sha256(pilot_report.read_bytes()).hexdigest(),
        "pilot_report_path": str(pilot_report),
        "c2_artifact_sha256": hashlib.sha256(c2_artifact.read_bytes()).hexdigest(),
        "c2_artifact_path": str(c2_artifact),
        "config": asdict(qualification.SELECTED_CONFIG),
        "seed": qualification.SELECTED_SEED,
        "data_hashes": {},
        "development_report_path": str(report),
        "development_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "hard_gates_passed": True,
        "protected_final_authorization": "one",
        "performance_reference_tokens_per_second": qualification.PILOT_THROUGHPUT_TOKENS_PER_SECOND,
        "model_material": {
            "model_card_path": str(card),
            "model_card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
            "model_manifest_path": str(model_manifest),
            "model_manifest_sha256": hashlib.sha256(
                model_manifest.read_bytes()
            ).hexdigest(),
        },
        "distributions": {
            "wheel_path": str(wheel),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "sdist_path": str(sdist),
            "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        },
    }


def test_registration_rejects_malformed_or_changed_evidence_before_final_read(
    monkeypatch: pytest.MonkeyPatch, tiny_artifact: Path, tmp_path: Path
) -> None:
    expected_sha = hashlib.sha256(tiny_artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(qualification, "PILOT_ARTIFACT_SHA256", expected_sha)
    monkeypatch.setattr(qualification, "_source_revision", lambda: "clean-revision")
    monkeypatch.setattr(qualification, "_source_hashes", lambda: {"tiny": "source"})
    registration = _registration(tmp_path, tiny_artifact)
    monkeypatch.setattr(
        qualification, "PILOT_REPORT_SHA256", registration["pilot_report_sha256"]
    )
    monkeypatch.setattr(
        qualification, "C2_ARTIFACT_SHA256", registration["c2_artifact_sha256"]
    )
    path = tmp_path / "registration.json"

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="malformed"):
        qualification._validate_registration(path, tiny_artifact)

    path.write_text(json.dumps(registration), encoding="utf-8")
    value, revision = qualification._validate_registration(path, tiny_artifact)
    assert value["candidate_id"] == "tiny"
    assert revision == "clean-revision"

    Path(registration["pilot_report_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="pilot report differs"):
        qualification._validate_registration(path, tiny_artifact)


def test_final_registration_marker_is_one_shot(tmp_path: Path) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text("{}", encoding="utf-8")

    marker = qualification._consume_final_registration(registration, "a" * 64)

    assert (
        json.loads(marker.read_text(encoding="utf-8"))["registration_sha256"]
        == "a" * 64
    )
    with pytest.raises(qualification.QualificationError, match="already consumed"):
        qualification._consume_final_registration(registration, "a" * 64)


def test_failure_report_retains_conservative_protected_state() -> None:
    partial = {
        "phase": "protected-final-read-attempted",
        "status": "in-progress",
        "registration": {"sha256": "a" * 64},
    }

    report = qualification._failure_report(
        ValueError("loader failed"),
        state={
            "phase": "protected-final-read-attempted",
            "attempted": True,
            "evaluated": True,
            "final_sha256": "b" * 64,
            "partial_report": partial,
        },
    )

    assert report["status"] == "reject"
    assert report["protected_final_attempted"] is True
    assert report["protected_final_evaluated"] is True
    assert report["registration"] == {"sha256": "a" * 64}
    assert report["retained_state"]["final_sha256"] == "b" * 64


def test_probe_measurement_and_prediction_validation() -> None:
    measurement = probe._measurement([4.0, 1.0, 3.0, 2.0])
    assert measurement == {
        "seconds": [4.0, 1.0, 3.0, 2.0],
        "median_seconds": 2.5,
        "iqr_over_median": 0.8,
    }
    assert probe._validate_predictions((("one", "two"),), (("NOUN", "VERB"),)) == (
        ("NOUN", "VERB"),
    )
    with pytest.raises(ValueError, match="invalid UPOS"):
        probe._validate_predictions((("one",),), (("INVALID",),))
    with pytest.raises(ValueError, match="sentence count"):
        probe._validate_predictions((("one",),), ())

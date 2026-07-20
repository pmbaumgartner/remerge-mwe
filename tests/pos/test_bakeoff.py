"""Project-owned controls for the isolated POS bakeoff harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import struct

import pytest

from tests.pos.evaluation.loader import Sentence, Token


MODULE_PATH = Path(__file__).parents[2] / "experiments" / "pos-tagger" / "bakeoff.py"
SPEC = importlib.util.spec_from_file_location("pos_bakeoff", MODULE_PATH)
assert SPEC and SPEC.loader
bakeoff = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bakeoff
SPEC.loader.exec_module(bakeoff)


class _Candidate:
    model_id = "control-v1"
    tokenizer_id = "fixed-boundaries-v1"
    artifact_bytes = 12
    artifact_sha256 = "a" * 64
    artifact_path = Path("/tmp/not-read-by-registration")

    def tag(self, documents: bakeoff.Boundaries) -> bakeoff.Predictions:
        return tuple(
            tuple(tuple("NOUN" for _form in sentence) for sentence in document)
            for document in documents
        )


def _sentence(form: str, tag: str, *, domain: str = "reviews") -> Sentence:
    return Sentence(
        domain + "-1",
        domain + "-1-1",
        (Token(domain + "-1", domain + "-1-1", 0, form, tag, domain),),
    )


def test_alignment_and_invalid_tag_controls_reject() -> None:
    expected = ((("word",),),)
    with pytest.raises(bakeoff.RejectedEvaluation, match="document count"):
        bakeoff.validate_predictions(expected, ())
    with pytest.raises(bakeoff.RejectedEvaluation, match="invalid UPOS"):
        bakeoff.validate_predictions(expected, ((("NOT-A-TAG",),),))


def test_quality_sensor_distinguishes_known_good_and_rotated_control() -> None:
    train = (_sentence("noun", "NOUN"), _sentence("verb", "VERB"))
    dev = (_sentence("noun", "NOUN"), _sentence("verb", "VERB"))
    known_good = ((("NOUN",), ("VERB",)),)
    broken = ((("VERB",), ("NOUN",)),)

    assert bakeoff.quality_metrics(train, dev, known_good)["overall_accuracy"] == 1.0
    assert bakeoff.quality_metrics(train, dev, broken)["overall_accuracy"] == 0.0


def test_quality_sensor_aggregates_oov_ambiguity_domain_and_tag_slices() -> None:
    train = (
        _sentence("lead", "NOUN"),
        _sentence("lead", "VERB"),
        _sentence("known", "NOUN"),
    )
    dev = (
        _sentence("lead", "VERB", domain="answers"),
        _sentence("novel", "ADJ", domain="reviews"),
    )
    predicted = ((("VERB",),), (("NOUN",),))

    metrics = bakeoff.quality_metrics(train, dev, predicted)

    assert metrics["overall_accuracy"] == 0.5
    assert metrics["ambiguous_accuracy"] == 1.0
    assert metrics["oov_accuracy"] == 0.0
    assert metrics["per_domain_accuracy"] == {"answers": 1.0, "reviews": 0.0}
    assert metrics["supported_tag_f1"]["VERB"]["f1"] == 1.0


def test_registration_rejects_different_data_hashes(tmp_path: Path) -> None:
    registration = {
        "candidate_id": "control",
        "source_revision": "b" * 40,
        "command": "python control.py",
        "seed": 7,
        "data_hashes": {"train": "1" * 64, "dev": "2" * 64},
        "artifact_sha256": "a" * 64,
        "environment": {"python": "test"},
        "model_id": "control-v1",
        "tokenizer_id": "fixed-boundaries-v1",
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(bakeoff.RejectedEvaluation, match="data hashes"):
        bakeoff.load_registration(
            path, _Candidate(), data_hashes={"train": "0" * 64, "dev": "2" * 64}
        )


def test_registration_rejects_malformed_provenance(tmp_path: Path) -> None:
    registration = {
        "candidate_id": "control",
        "source_revision": "short",
        "command": "",
        "seed": "7",
        "data_hashes": {"train": "1" * 64, "dev": "2" * 64},
        "artifact_sha256": "a" * 64,
        "environment": {},
        "model_id": "control-v1",
        "tokenizer_id": "fixed-boundaries-v1",
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(bakeoff.RejectedEvaluation, match="provenance"):
        bakeoff.load_registration(
            path, _Candidate(), data_hashes=registration["data_hashes"]
        )


def test_candidate_loader_rejects_self_reported_artifact_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual")
    module = tmp_path / "candidate.py"
    module.write_text(
        "from pathlib import Path\n"
        "class Candidate:\n"
        " model_id='x'; tokenizer_id='x'; artifact_bytes=1; artifact_sha256='a'*64; artifact_path=Path(r'"
        + str(artifact)
        + "')\n"
        " def tag(self, documents): return documents\n"
        "def create(): return Candidate()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(bakeoff.RejectedEvaluation, match="self-reported"):
        bakeoff.load_candidate("candidate:create")


def test_dev_analog_gate_rejects_bad_metrics() -> None:
    quality = {
        "overall_accuracy": 0.0,
        "macro_f1": 0.0,
        "oov_accuracy": 0.0,
        "ambiguous_accuracy": 0.0,
        "per_domain_accuracy": {"x": 0.0},
        "supported_tag_f1": {"NOUN": {"count": 50, "f1": 0.0}},
    }
    utility = {
        "precision_generated": 0.0,
        "precision_unfiltered": 0.0,
        "unfiltered_count": 1,
        "generated_overlap_with_unfiltered": 1,
        "recall_generated": 0.0,
        "recall_unfiltered": 1.0,
        "recall_gold_filtered": 1.0,
        "precision_improvement_interval": {"lower": 0.0, "upper": 0.0},
        "recall_loss_unfiltered_interval": {"lower": 1.0, "upper": 1.0},
        "recall_loss_gold_interval": {"lower": 1.0, "upper": 1.0},
    }
    assert bakeoff.dev_analog_gates(quality, utility)


def test_subset_predictions_preserves_target_document_and_sentence_order() -> None:
    source = (
        _sentence("first", "NOUN", domain="answers"),
        _sentence("second", "VERB", domain="reviews"),
    )
    predicted = ((("NOUN",),), (("VERB",),))

    assert bakeoff.subset_predictions(source, predicted, source[1:]) == ((("VERB",),),)


def test_common_command_has_no_final_split_option() -> None:
    assert 'add_argument("--final"' not in MODULE_PATH.read_text(encoding="utf-8")


def test_c2_reproduction_rejects_unpinned_inputs_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.conllu"
    dev = tmp_path / "dev.conllu"
    train.write_text("not the pinned train split", encoding="utf-8")
    dev.write_text("not the pinned dev split", encoding="utf-8")
    monkeypatch.setattr(
        bakeoff.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("trainer must not run"),
    )

    with pytest.raises(bakeoff.RejectedEvaluation, match="pinned training split"):
        bakeoff.reproduce_retained_c2(
            train, dev, tmp_path / "artifact", tmp_path / "report"
        )


def test_warm_measurement_rejects_stateful_candidate_output() -> None:
    class StatefulCandidate(_Candidate):
        calls = 0

        def tag(self, documents: bakeoff.Boundaries) -> bakeoff.Predictions:
            self.calls += 1
            tag = "NOUN" if self.calls == 1 else "VERB"
            return (((tag,),),)

    candidate = StatefulCandidate()
    inputs = ((("word",),),)
    expected = candidate.tag(inputs)

    with pytest.raises(bakeoff.RejectedEvaluation, match="changed during warmup"):
        bakeoff.warm_inference_measurements(candidate, inputs, expected)


def test_c2_adapter_parses_and_predicts_a_tiny_golden_artifact(tmp_path: Path) -> None:
    adapter_path = MODULE_PATH.with_name("c2_adapter.py")
    adapter_spec = importlib.util.spec_from_file_location(
        "c2_adapter_test", adapter_path
    )
    assert adapter_spec and adapter_spec.loader
    adapter_module = importlib.util.module_from_spec(adapter_spec)
    sys.modules[adapter_spec.name] = adapter_module
    adapter_spec.loader.exec_module(adapter_module)
    key = adapter_module._fnv1a("alpha")
    header = adapter_module.HEADER.pack(b"RMPOS001", 1, 17, 2, 22, 0, 5, 5, 1, 0, 34)
    payload = header + b"model" + b"token" + struct.pack("<17h", *([0] * 17))
    payload += struct.pack("<QB", key, adapter_module.TAG_INDEX["NOUN"])
    payload += bytes(34)
    artifact = tmp_path / "golden.rmpos"
    artifact.write_bytes(payload)

    candidate = adapter_module.C2Adapter(artifact)

    assert candidate.tag(((("alpha", "!"),),)) == ((("NOUN", "PUNCT"),),)
    artifact.write_bytes(payload + b"trailing")
    with pytest.raises(ValueError, match="lengths"):
        adapter_module.C2Adapter(artifact)


def test_c2_feature_and_forced_tag_parity_with_retained_trainer() -> None:
    trainer_path = MODULE_PATH.parents[1] / "pos-linear" / "train.py"
    trainer_spec = importlib.util.spec_from_file_location(
        "retained_trainer", trainer_path
    )
    assert trainer_spec and trainer_spec.loader
    trainer = importlib.util.module_from_spec(trainer_spec)
    sys.modules[trainer_spec.name] = trainer
    trainer_spec.loader.exec_module(trainer)
    adapter_path = MODULE_PATH.with_name("c2_adapter.py")
    adapter_spec = importlib.util.spec_from_file_location(
        "c2_adapter_parity", adapter_path
    )
    assert adapter_spec and adapter_spec.loader
    adapter = importlib.util.module_from_spec(adapter_spec)
    sys.modules[adapter_spec.name] = adapter
    adapter_spec.loader.exec_module(adapter)

    forms = ("ÉCOLE", "Café-2", "42", "…")
    for index, form in enumerate(forms):
        assert adapter._features(forms, index) == trainer.features(forms, index)
        assert adapter._forced_tag(form) == trainer.forced_tag(form)


def test_retained_c2_constants_document_the_historical_reproduction_target() -> None:
    assert bakeoff.RETAINED_C2_ACCURACY == pytest.approx(0.937691)
    assert bakeoff.RETAINED_C2_TOLERANCE < 0.00001

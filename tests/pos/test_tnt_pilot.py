"""Controls for the bounded, development-only TnT pilot."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import pytest


ROOT = Path(__file__).parents[2]
TNT_PATH = ROOT / "experiments/pos-tnt"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tnt = _module("pos_tnt_test", TNT_PATH / "tnt.py")
bakeoff = _module("pos_tnt_bakeoff_test", ROOT / "experiments/pos-tagger/bakeoff.py")
evaluator = _module("pos_tnt_evaluate_test", TNT_PATH / "evaluate.py")


CONFIG = tnt.Config(
    transition_alpha=0.1,
    emission_alpha=0.1,
    suffix_alpha=0.25,
    suffix_length=3,
    rare_word_max_count=1,
)


def _model():
    return tnt.train(
        (
            (("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")),
            (("A", "DET"), ("dog", "NOUN"), ("runs", "VERB")),
            (("bright", "ADJ"),),
            (("running", "VERB"),),
        ),
        CONFIG,
    )


def test_training_is_deterministic_and_has_normalized_deleted_lambdas() -> None:
    first = _model()
    second = _model()

    assert first.artifact() == second.artifact()
    assert math.isclose(sum(first.lambdas), 1.0)
    assert all(value >= 0 for value in first.lambdas)
    assert first.event_counts[tnt.EOS] == 4


def test_transition_interpolation_and_eos_change_the_viterbi_winner() -> None:
    # The first token locally prefers ADJ, but its termination is very unlikely;
    # exact Viterbi must include EOS and choose NOUN instead.
    adj = tnt.TAG_INDEX["ADJ"]
    noun = tnt.TAG_INDEX["NOUN"]
    model = tnt.Model(
        config=CONFIG,
        lambdas=(0.0, 0.0, 1.0),
        event_counts=(1,) * tnt.EVENT_COUNT,
        bigram={},
        context={(tnt.BOS, tnt.BOS): 10, (tnt.BOS, adj): 10, (tnt.BOS, noun): 10},
        trigram={
            (tnt.BOS, tnt.BOS, adj): 9,
            (tnt.BOS, tnt.BOS, noun): 1,
            (tnt.BOS, adj, adj): 10,
            (tnt.BOS, noun, tnt.EOS): 10,
        },
        emissions={},
        tag_totals=(0,) * len(tnt.TAGS),
        suffixes={},
        suffix_totals={},
        vocabulary_size=1,
    )
    forms = ("novel",)
    exhaustive = []
    for path in itertools.product(range(len(tnt.TAGS)), repeat=len(forms)):
        score = math.log(model.transition(tnt.BOS, tnt.BOS, path[0]))
        score += math.log(model.emission(forms[0], path[0]))
        score += math.log(model.transition(tnt.BOS, path[0], tnt.EOS))
        exhaustive.append((score, path))
    expected = max(
        exhaustive, key=lambda item: (item[0], tuple(-tag for tag in item[1]))
    )

    assert model.tag_sentence(forms) == tuple(tnt.TAGS[tag] for tag in expected[1])
    assert expected[1] == (noun,)
    assert model.transition(tnt.BOS, tnt.BOS, adj) > model.transition(
        tnt.BOS, tnt.BOS, noun
    )

    # Exhaustively score every two-token path as an independent check of the
    # dynamic program's recurrence and its final EOS transition.
    two_forms = ("novel", "novel")
    two_token_scores = []
    for path in itertools.product(range(len(tnt.TAGS)), repeat=2):
        score = math.log(model.transition(tnt.BOS, tnt.BOS, path[0]))
        score += math.log(model.emission(two_forms[0], path[0]))
        score += math.log(model.transition(tnt.BOS, path[0], path[1]))
        score += math.log(model.emission(two_forms[1], path[1]))
        score += math.log(model.transition(path[0], path[1], tnt.EOS))
        two_token_scores.append((score, path))
    expected_two = max(
        two_token_scores, key=lambda item: (item[0], tuple(-tag for tag in item[1]))
    )
    assert model.tag_sentence(two_forms) == tuple(
        tnt.TAGS[tag] for tag in expected_two[1]
    )


def test_viterbi_ties_are_stable_and_scores_stay_in_log_space() -> None:
    model = tnt.Model(
        CONFIG,
        (1 / 3, 1 / 3, 1 / 3),
        (1,) * tnt.EVENT_COUNT,
        {},
        {},
        {},
        {},
        (0,) * len(tnt.TAGS),
        {},
        {},
        1,
    )
    assert model.tag_sentence(("x",) * 300) == ("ADJ",) * 300


def test_unknown_suffixes_use_rare_evidence_and_seen_words_stay_lexical() -> None:
    model = _model()
    verb = tnt.TAG_INDEX["VERB"]
    noun = tnt.TAG_INDEX["NOUN"]

    assert model.emission("dancing", verb) > model.emission("dancing", noun)
    seen_noun = model.emission("running", noun)
    lexical_verb = (
        model.emissions[tnt.token_hash("running"), verb] + CONFIG.emission_alpha
    ) / (
        model.tag_totals[verb]
        + CONFIG.emission_alpha * model.tag_vocabulary_sizes[verb]
    )
    assert seen_noun == 0.0
    assert model.emission("running", verb) == lexical_verb
    assert model.candidate_tags("running") == (verb,)

    repeated = tnt.train(((("common", "NOUN"),), (("common", "NOUN"),)), CONFIG)
    suffix = tnt.suffix_hash("common", 3)
    assert (3, suffix) not in repeated.suffix_totals


def test_config_and_serialization_rejection_controls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        tnt.Config(math.nan, 0.1, 0.1, 3, 1).validate()
    with pytest.raises(ValueError, match="finite"):
        tnt.Config(math.inf, 0.1, 0.1, 3, 1).validate()
    with pytest.raises(ValueError, match="integers"):
        tnt.Config(0.1, 0.1, 0.1, 3.0, 1).validate()
    with pytest.raises(ValueError, match="integers"):
        tnt.Config(0.1, 0.1, 0.1, True, 1).validate()

    model = _model()
    artifact = model.artifact()
    path = tmp_path / "tnt.json"
    digest = tnt.write_artifact(model, path)
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert tnt.Model.from_artifact(path.read_bytes()).artifact() == artifact
    assert b"running" not in artifact
    with pytest.raises(ValueError, match="JSON"):
        tnt.Model.from_artifact(artifact[:-1])
    corrupted = json.loads(artifact)
    corrupted["emissions"][0][1] = len(tnt.TAGS)
    with pytest.raises(ValueError):
        tnt.Model.from_artifact(json.dumps(corrupted).encode())
    duplicate_schema = artifact.replace(
        b'"schema":"remerge-pos-tnt-v2",',
        b'"schema":"remerge-pos-tnt-v2","schema":"remerge-pos-tnt-v2",',
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        tnt.Model.from_artifact(duplicate_schema)
    bool_count = json.loads(artifact)
    bool_count["event_counts"][0] = True
    with pytest.raises(ValueError, match="aggregate counts"):
        tnt.Model.from_artifact(json.dumps(bool_count).encode())


def test_hash_collision_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tnt, "token_hash", lambda _value: 1)
    with pytest.raises(ValueError, match="hash collision"):
        tnt.train(((("alpha", "NOUN"), ("beta", "VERB")),), CONFIG)


def test_viterbi_hot_path_uses_precomputed_aggregates() -> None:
    class NoItemsDict(dict):
        def items(self):  # pragma: no cover - an invocation is the failure.
            raise AssertionError("Viterbi must not rescan the bigram table")

    model = _model()
    object.__setattr__(model, "bigram", NoItemsDict(model.bigram))
    started = perf_counter()
    assert len(model.tag_sentence(("novel",) * 100)) == 100
    # A smoke bound catches accidental quadratic/table-scan regressions without
    # claiming a machine-specific product throughput threshold.
    assert perf_counter() - started < 5.0


def test_grid_selection_is_deterministic_and_not_any_run_cherry_picking() -> None:
    def record(grid_index: int, decision: str, accuracy: float) -> dict:
        return {
            "grid_index": grid_index,
            "seed": evaluator.SEEDS[0],
            "decision": decision,
            "quality": {
                "overall_accuracy": accuracy,
                "macro_f1": accuracy,
                "oov_accuracy": accuracy,
                "ambiguous_accuracy": accuracy,
            },
        }

    selected = evaluator._selected_grid(
        [record(0, "reject", 1.0), record(1, "retain", 0.5)]
    )

    assert selected["grid_index"] == 1
    assert evaluator.C2_ARTIFACT_SHA256 == (
        "3607ac68ba6750ab3d1fc6a88a7d3105761250f72b2c4852c30594b44d6b640d"
    )


def test_common_harness_adapter_smoke_and_alignment_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "tnt.json"
    tnt.write_artifact(_model(), artifact)
    monkeypatch.setenv("REMERGE_POS_TNT_ARTIFACT", str(artifact))
    monkeypatch.syspath_prepend(str(TNT_PATH))

    candidate = bakeoff.load_candidate("tnt_adapter:create_candidate")
    inputs = ((("A", "dog"),),)
    prediction = bakeoff.validate_predictions(inputs, candidate.tag(inputs))

    assert (
        candidate.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert len(prediction[0][0]) == 2
    with pytest.raises(bakeoff.RejectedEvaluation, match="invalid UPOS"):
        bakeoff.validate_predictions(inputs, ((("BROKEN", "NOUN"),),))

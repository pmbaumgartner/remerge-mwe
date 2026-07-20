"""Controls for the bounded, development-only structured-perceptron pilot."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).parents[2]
PILOT = ROOT / "experiments/pos-structured-perceptron"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


perceptron = _module(
    "pos_structured_perceptron_test", PILOT / "structured_perceptron.py"
)
evaluator = _module("pos_structured_perceptron_evaluate_test", PILOT / "evaluate.py")
bakeoff = _module(
    "pos_structured_perceptron_bakeoff_test", ROOT / "experiments/pos-tagger/bakeoff.py"
)

CONFIG = perceptron.Config(epochs=2, feature_cutoff=1, feature_buckets=4096)
DATA = (
    (("The", "DET"), ("watch", "NOUN")),
    (("I", "PRON"), ("watch", "VERB")),
    (("bright", "ADJ"), ("lights", "NOUN")),
    (("lights", "NOUN"), ("shine", "VERB")),
)


def _empty_model(**changes):
    values = {
        "config": CONFIG,
        "seed": 1,
        "steps": 1,
        "training_digest": "0" * 64,
        "feature_weights": {},
        "transition_weights": {},
    }
    values.update(changes)
    return perceptron.Model(**values)


def _path_score(model, forms, path):
    score = 0.0
    for index, tag in enumerate(path):
        for feature in perceptron.feature_ids(
            forms, index, model.config.feature_buckets
        ):
            score += model.feature_weights.get((feature, tag), 0.0)
    score += model.transition_weights.get((perceptron.BOS, path[0]), 0.0)
    score += sum(
        model.transition_weights.get(pair, 0.0) for pair in zip(path, path[1:])
    )
    score += model.transition_weights.get((path[-1], perceptron.EOS), 0.0)
    return score


def test_feature_family_is_bounded_morphological_and_preserves_hash_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forms = ("Pre-tested42", "forms")
    ordinary = perceptron.feature_ids(forms, 0, 4096)
    assert ordinary == tuple(sorted(ordinary))
    assert 10 < len(ordinary) < 24

    monkeypatch.setattr(perceptron, "stable_hash", lambda _value: 9)
    collided = perceptron.feature_ids(forms, 0, 4096)
    assert len(collided) == len(ordinary)
    assert collided == (9,) * len(ordinary)


def test_lexicon_identity_collision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perceptron, "lexicon_hash", lambda _value: 1)
    with pytest.raises(ValueError, match="identity hash collision"):
        perceptron.train(((("alpha", "NOUN"), ("beta", "VERB")),), CONFIG, 1)


def test_exact_viterbi_matches_exhaustive_paths_including_eos() -> None:
    forms = ("A", "novel", "path")
    feature_weights = {}
    for position in range(len(forms)):
        for feature in perceptron.feature_ids(forms, position, CONFIG.feature_buckets):
            for tag in perceptron.ALL_TAGS:
                value = ((feature + tag * 7 + position) % 11) - 5
                if value:
                    feature_weights[feature, tag] = float(value)
    transitions = {
        (previous, tag): float(((previous + 3) * (tag + 5)) % 7 - 3)
        for previous in perceptron.ALL_TAGS + (perceptron.BOS,)
        for tag in perceptron.ALL_TAGS
    }
    transitions.update(
        {(tag, perceptron.EOS): float(tag % 3 - 1) for tag in perceptron.ALL_TAGS}
    )
    transitions = {key: value for key, value in transitions.items() if value}
    model = _empty_model(
        feature_weights=feature_weights, transition_weights=transitions
    )
    exhaustive = max(
        itertools.product(perceptron.ALL_TAGS, repeat=len(forms)),
        key=lambda path: (_path_score(model, forms, path), tuple(-tag for tag in path)),
    )
    assert model.decode_indices(forms) == exhaustive

    # EOS reverses the locally preferred winner.
    noun = perceptron.TAG_INDEX["NOUN"]
    eos_model = _empty_model(
        transition_weights={
            (perceptron.BOS, 0): 2.0,
            (perceptron.BOS, noun): 1.0,
            (0, perceptron.EOS): -10.0,
        }
    )
    assert eos_model.decode_indices(("x",)) == (noun,)


def test_full_path_ties_choose_lexicographically_smallest_and_cache_local_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = perceptron.feature_ids

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(perceptron, "feature_ids", counted)
    model = _empty_model()
    assert model.decode_indices(("x", "y", "z")) == (0, 0, 0)
    assert calls == 3


def test_structured_update_touches_each_delta_once_and_scores_all_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = perceptron._Averager.update

    def recording(self, key, delta, step):
        calls.append((id(self), key, step))
        return original(self, key, delta, step)

    monkeypatch.setattr(perceptron._Averager, "update", recording)
    model = perceptron.train(
        ((("one", "NOUN"), ("runs", "VERB")),),
        perceptron.Config(epochs=1, feature_cutoff=1, feature_buckets=4096),
        17,
    )
    assert len(calls) == len(set(calls))
    noun = perceptron.TAG_INDEX["NOUN"]
    verb = perceptron.TAG_INDEX["VERB"]
    assert model.transition_weights[perceptron.BOS, noun] > 0
    assert model.transition_weights[noun, verb] > 0
    assert model.transition_weights[verb, perceptron.EOS] > 0


def test_lazy_averaging_matches_explicit_preupdate_snapshots() -> None:
    averager = perceptron._Averager()
    key = (1, 2)
    snapshots = []
    averager.update(key, 1.0, 0)
    snapshots.append(averager.weights.get(key, 0.0))
    averager.update(key, 1.0, 1)
    snapshots.append(averager.weights.get(key, 0.0))
    assert averager.averaged(2)[key] == sum(snapshots) / len(snapshots) == 1.5

    # Correct examples still advance the single sentence step.
    correct = perceptron.train(
        ((("already", "ADJ"),),),
        perceptron.Config(epochs=3, feature_cutoff=1, feature_buckets=4096),
        2,
    )
    assert correct.steps == 3


def test_true_training_seeds_shuffle_and_produce_distinct_artifacts() -> None:
    models = [perceptron.train(DATA, CONFIG, seed) for seed in evaluator.SEEDS]
    assert len({model.artifact() for model in models}) == len(evaluator.SEEDS)
    assert all(model.steps == len(DATA) * CONFIG.epochs for model in models)


def test_serialization_roundtrip_covers_all_tags_index_zero_and_corruption(
    tmp_path: Path,
) -> None:
    feature_weights = {
        (index, tag): float(tag + 1) for index, tag in enumerate(perceptron.ALL_TAGS)
    }
    transitions = {(perceptron.BOS, tag): float(tag + 1) for tag in perceptron.ALL_TAGS}
    transitions.update(
        {(tag, perceptron.EOS): -float(tag + 1) for tag in perceptron.ALL_TAGS}
    )
    model = _empty_model(
        feature_weights=feature_weights, transition_weights=transitions
    )
    artifact = model.artifact()
    restored = perceptron.Model.from_artifact(artifact)
    assert restored.artifact() == artifact
    assert restored.feature_weights[0, 0] == 1.0
    assert len(artifact) <= 20 * 1024 * 1024
    assert b"gold" not in artifact and b"sentence" not in artifact

    for corrupted in (
        artifact[:-1],
        b"BROKEN!!" + artifact[8:],
        artifact + b"trailing",
    ):
        with pytest.raises(ValueError):
            perceptron.Model.from_artifact(corrupted)

    path = tmp_path / "candidate.rmsp"
    digest = perceptron.write_artifact(model, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_adapter_and_alignment_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "candidate.rmsp"
    perceptron.write_artifact(perceptron.train(DATA, CONFIG, 1), artifact)
    monkeypatch.setenv("REMERGE_POS_STRUCTURED_PERCEPTRON_ARTIFACT", str(artifact))
    monkeypatch.syspath_prepend(str(PILOT))
    candidate = bakeoff.load_candidate("structured_perceptron_adapter:create_candidate")
    inputs = ((("The", "watch"),),)
    checked = bakeoff.validate_predictions(inputs, candidate.tag(inputs))
    assert len(checked[0][0]) == 2
    with pytest.raises(bakeoff.RejectedEvaluation, match="token count"):
        bakeoff.validate_predictions(inputs, ((("NOUN",),),))


def _selection_record(grid: int, seed: int, overall: float) -> dict:
    return {
        "grid_index": grid,
        "seed": seed,
        "candidate": {"artifact_sha256": f"{grid}{seed}"},
        "decision": "reject",
        "quality": {
            "overall_accuracy": overall,
            "macro_f1": overall,
            "oov_accuracy": overall,
            "ambiguous_accuracy": overall,
        },
        "mwe_utility": {
            "precision_generated": overall,
            "recall_generated": overall,
        },
    }


def test_selection_uses_grid_aggregate_then_fixed_seed_not_best_seed() -> None:
    records = []
    for seed in evaluator.SEEDS:
        records.append(_selection_record(0, seed, 0.8))
        records.append(
            _selection_record(1, seed, 0.9 if seed != evaluator.SEEDS[-1] else 0.0)
        )
    records[0]["quality"]["overall_accuracy"] = 1.0
    selected, summaries = evaluator._select(records)
    assert selected["grid_index"] == 0
    assert selected["seed"] == evaluator.SEEDS[0]
    assert len(summaries) == 2
    assert evaluator.C2_ARTIFACT_SHA256.endswith("b640d")
    assert evaluator.TNT_ARTIFACT_SHA256.endswith("e94")
    assert evaluator.TNT_REPORT_SHA256.endswith("567d")


def test_registration_is_shell_safe_and_pins_source_and_artifact_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact with spaces.rmsp"
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(evaluator, "_source_revision", lambda: "a" * 40)
    monkeypatch.setattr(evaluator, "_source_hashes", lambda: {"source.py": "b" * 64})
    registration = evaluator._registration(
        candidate_id="candidate",
        command=["python", "path with spaces", "quoted'argument"],
        seed=1,
        data_hashes={"train": "c" * 64, "dev": "d" * 64},
        artifact_sha256="e" * 64,
        model_id="model",
        tokenizer_id="tokens",
        variable="ARTIFACT",
        artifact=artifact,
    )
    assert (
        registration["command"] == "python 'path with spaces' 'quoted'\"'\"'argument'"
    )
    assert registration["source_hashes"] == {"source.py": "b" * 64}
    assert registration["artifact_sha256"] == "e" * 64


def test_snapshot_verifies_and_freezes_train_dev_and_mwe_bytes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    files = {
        ("ewt", "train.conllu"): b"train",
        ("ewt", "dev.conllu"): b"dev",
        ("streusle", "dev.conllu"): b"mwe",
    }
    for (root, name), contents in files.items():
        path = source_root / root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    manifest = {
        "sources": {
            "ud_ewt": {"relative_root": "ewt"},
            "streusle": {"relative_root": "streusle"},
        },
        "splits": {
            "train": {"path": "train.conllu", "sha256": digest(b"train")},
            "dev": {"path": "dev.conllu", "sha256": digest(b"dev")},
        },
        "mwe_gold": {
            "source": "streusle",
            "splits": {"dev": {"path": "dev.conllu", "sha256": digest(b"mwe")}},
        },
    }
    with evaluator._development_snapshot(manifest, source_root) as (snapshot, hashes):
        assert (snapshot / "ewt/train.conllu").read_bytes() == b"train"
        assert hashes == {
            "train": digest(b"train"),
            "dev": digest(b"dev"),
            "mwe-dev": digest(b"mwe"),
        }
        assert (snapshot / "ewt/train.conllu").stat().st_mode & stat.S_IWUSR == 0
    (source_root / "ewt/dev.conllu").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="dev input"):
        with evaluator._development_snapshot(manifest, source_root):
            pass


def test_config_input_and_artifact_validation_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        perceptron.Config(True, 1).validate()
    with pytest.raises(ValueError, match="feature_cutoff"):
        perceptron.Config(1, 0).validate()
    with pytest.raises(ValueError, match="seed"):
        perceptron.train(DATA, CONFIG, True)
    with pytest.raises(ValueError, match="UPOS"):
        perceptron.train(((("x", "BAD"),),), CONFIG, 1)
    with pytest.raises(ValueError, match="feature weight"):
        _empty_model(feature_weights={(1, True): 1.0})
    with pytest.raises(ValueError, match="transition weight"):
        _empty_model(transition_weights={(0.0, 1): 1.0})

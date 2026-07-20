from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from tests.pos.evaluation.loader import (
    ManifestError,
    load_gold_split,
    load_manifest,
    load_tagged_split,
    read_conllu,
    validate_manifest,
)


def test_frozen_manifest_is_structurally_valid() -> None:
    manifest = load_manifest()

    assert manifest["dataset_id"] == "ud-ewt-r2.10-streusle-v4.5"
    assert manifest["release_adequate"] is True
    protected_final = manifest["protected_final"]
    assert isinstance(protected_final, dict)
    assert (
        cast(dict[str, object], protected_final)["candidate_registration_required"]
        is True
    )
    mwe_dev = cast(dict[str, Any], manifest["mwe_gold"])["splits"]["dev"]
    assert mwe_dev["expected_unretained_sentences"] == 8


def test_manifest_rejects_a_missing_split() -> None:
    manifest = load_manifest()
    invalid = json.loads(json.dumps(manifest))
    del invalid["splits"]["dev"]

    with pytest.raises(ManifestError, match="exactly train, dev, and final"):
        validate_manifest(invalid)


def test_manifest_rejects_an_unapproved_release_dataset() -> None:
    manifest = load_manifest()
    invalid = json.loads(json.dumps(manifest))
    invalid["release_adequate"] = False

    with pytest.raises(ManifestError, match="not authorized for release"):
        validate_manifest(invalid)


def test_final_split_requires_explicit_protected_harness_authorization() -> None:
    with pytest.raises(ManifestError, match="protected-harness authorization"):
        load_tagged_split(load_manifest(), Path("not-used"), "final")

    with pytest.raises(ManifestError, match="protected-harness authorization"):
        load_gold_split(load_manifest(), Path("not-used"), "final")


def test_gold_loader_owns_dev_alignment_span_policy_and_shape(
    tmp_path: Path,
) -> None:
    manifest: dict[str, Any] = json.loads(json.dumps(load_manifest()))
    ud_root = tmp_path / "ud-ewt"
    streusle_root = tmp_path / "streusle" / "dev"
    ud_root.mkdir()
    streusle_root.mkdir(parents=True)
    train = ud_root / "train.conllu"
    dev = ud_root / "dev.conllu"
    mwe = streusle_root / "streusle.ud_dev.conllulex"
    train.write_text(_conllu("train-1", "train-1-1", (("base", "NOUN"),)))
    dev.write_text(
        _conllu("reviews-1", "reviews-1-1", (("good", "ADJ"), ("work", "NOUN")))
    )
    mwe.write_text(
        "# sent_id = reviews-1-1\n"
        "1\tgood\tgood\tADJ\t_\t_\t0\troot\t_\t_\t1:1\n"
        "2\twork\twork\tNOUN\t_\t_\t1\tobj\t_\t_\t1:2\n\n",
        encoding="utf-8",
    )
    manifest["splits"]["train"].update(
        path="train.conllu",
        sha256=_file_sha256(train),
        expected_source_tokens=1,
        expected_tokens=1,
    )
    manifest["splits"]["dev"].update(
        path="dev.conllu",
        sha256=_file_sha256(dev),
        expected_source_tokens=2,
        expected_tokens=2,
    )
    dev_gold = manifest["mwe_gold"]["splits"]["dev"]
    dev_gold.update(
        sha256=_file_sha256(mwe),
        minimum_in_scope_spans=1,
        expected_spans=1,
        expected_unretained_sentences=0,
        expected_documents=1,
        expected_sentences=1,
        expected_tokens=2,
    )

    gold = load_gold_split(manifest, tmp_path, "dev")

    assert gold.split == "dev"
    assert [(token.form, token.upos) for token in gold.sentences[0].tokens] == [
        ("good", "ADJ"),
        ("work", "NOUN"),
    ]
    assert {(span.start_token, span.end_token) for span in gold.mwe_spans} == {(0, 2)}


def test_gold_loader_rejects_a_wrong_unretained_sentence_count(
    tmp_path: Path,
) -> None:
    manifest: dict[str, Any] = json.loads(json.dumps(load_manifest()))
    ud_root = tmp_path / "ud-ewt"
    streusle_root = tmp_path / "streusle" / "dev"
    ud_root.mkdir()
    streusle_root.mkdir(parents=True)
    train = ud_root / "train.conllu"
    dev = ud_root / "dev.conllu"
    mwe = streusle_root / "streusle.ud_dev.conllulex"
    train.write_text(_conllu("train-1", "train-1-1", (("duplicate", "NOUN"),)))
    dev.write_text(
        _conllu("reviews-1", "reviews-1-1", (("duplicate", "NOUN"),))
        + _conllu(
            "reviews-2",
            "reviews-2-1",
            (("good", "ADJ"), ("work", "NOUN")),
        )
    )
    mwe.write_text(
        "# sent_id = reviews-1-1\n"
        "1\tduplicate\tduplicate\tNOUN\t_\t_\t0\troot\t_\t_\t_\n\n"
        "# sent_id = reviews-2-1\n"
        "1\tgood\tgood\tADJ\t_\t_\t0\troot\t_\t_\t1:1\n"
        "2\twork\twork\tNOUN\t_\t_\t1\tobj\t_\t_\t1:2\n\n",
        encoding="utf-8",
    )
    manifest["splits"]["train"].update(
        path="train.conllu",
        sha256=_file_sha256(train),
        expected_source_tokens=1,
        expected_tokens=1,
    )
    manifest["splits"]["dev"].update(
        path="dev.conllu",
        sha256=_file_sha256(dev),
        expected_source_tokens=3,
        expected_tokens=2,
    )
    dev_gold = manifest["mwe_gold"]["splits"]["dev"]
    dev_gold.update(
        sha256=_file_sha256(mwe),
        minimum_in_scope_spans=1,
        expected_spans=1,
        expected_unretained_sentences=0,
        expected_documents=1,
        expected_sentences=1,
        expected_tokens=2,
    )

    with pytest.raises(ManifestError, match="unretained sentence count 1 != frozen 0"):
        load_gold_split(manifest, tmp_path, "dev")

    dev_gold["expected_unretained_sentences"] = 1
    gold = load_gold_split(manifest, tmp_path, "dev")

    assert len(gold.sentences) == 1
    assert len(gold.mwe_spans) == 1


def _conllu(
    document_id: str, sentence_id: str, tokens: tuple[tuple[str, str], ...]
) -> str:
    rows = [f"# newdoc id = {document_id}", f"# sent_id = {sentence_id}"]
    rows.extend(
        f"{index}\t{form}\t{form}\t{upos}\t_\t_\t0\troot\t_\t_"
        for index, (form, upos) in enumerate(tokens, start=1)
    )
    return "\n".join(rows) + "\n\n"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_conllu_reader_preserves_integer_words_and_rejects_non_nfc(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.conllu"
    source.write_text(
        "# newdoc id = reviews-1\n"
        "# sent_id = reviews-1-1\n"
        "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tcan\tcan\tAUX\t_\t_\t0\troot\t_\t_\n"
        "2\tnot\tnot\tPART\t_\t_\t1\tadvmod\t_\t_\n"
        "3.1\tghost\tghost\tX\t_\t_\t_\t_\t_\t_\n\n",
        encoding="utf-8",
    )

    sentences = tuple(read_conllu(source))

    assert [(token.form, token.upos) for token in sentences[0].tokens] == [
        ("can", "AUX"),
        ("not", "PART"),
    ]

    source.write_text(
        "# newdoc id = reviews-1\n# sent_id = reviews-1-1\n1\te\u0301\te\u0301\tNOUN\t_\t_\t0\troot\t_\t_\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="not NFC"):
        tuple(read_conllu(source))

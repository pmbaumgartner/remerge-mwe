from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tests.pos.evaluation.loader import (
    ManifestError,
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

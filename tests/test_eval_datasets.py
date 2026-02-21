from pathlib import Path
import shutil

import pytest

from scripts.mwe_eval.datasets import coam, parseme12, streusle
from scripts.mwe_eval.datasets.base import count_gold_mentions


def _copy_fixture(src_name: str, dst: Path) -> None:
    src = Path("tests/eval_fixtures") / src_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


@pytest.mark.fast
def test_parseme12_load_split_parses_contiguous_and_discontinuous_mentions(tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"
    target = data_root / "parseme" / "sharedtask-data" / "1.2" / "EN" / "dev.cupt"
    _copy_fixture("parseme_sample.cupt", target)

    split_data = parseme12.load_split(data_root, "dev", parseme_langs=["EN"])

    assert split_data.dataset == "parseme12"
    assert split_data.split == "dev"
    assert len(split_data.sentences) == 2

    gold_total, gold_contiguous, gold_discontinuous = count_gold_mentions(split_data)
    assert (gold_total, gold_contiguous, gold_discontinuous) == (3, 2, 1)


@pytest.mark.fast
def test_streusle_load_split_supports_smwes_and_generic_mwes(tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"
    target = data_root / "streusle" / "streusle-5.0" / "dev.json"
    _copy_fixture("streusle_sample.json", target)

    split_data = streusle.load_split(data_root, "dev")
    assert split_data.dataset == "streusle"
    assert len(split_data.sentences) == 2

    first = split_data.sentences[0]
    assert first.tokens == ("kick", "the", "bucket")
    assert first.mentions[0].token_indices == (0, 1, 2)


@pytest.mark.fast
def test_coam_load_split_uses_json_fallback_without_hf_dependency(tmp_path):
    data_root = tmp_path / "data" / "mwe_eval"
    target = data_root / "coam" / "coam_sample.json"
    _copy_fixture("coam_sample.json", target)

    split_data = coam.load_split(data_root, "dev")
    assert split_data.dataset == "coam"
    assert len(split_data.sentences) == 2
    assert split_data.sentences[0].mentions[0].token_indices == (0, 1)

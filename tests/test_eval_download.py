from pathlib import Path
import zipfile

import pytest

from scripts.mwe_eval import download as download_cli
from scripts.mwe_eval.datasets import coam, parseme12, streusle


@pytest.mark.fast
def test_download_cli_dispatches_all(monkeypatch, tmp_path):
    calls: list[tuple[str, Path, bool]] = []

    def fake_download(dataset: str, data_root: Path, *, force: bool = False):
        calls.append((dataset, data_root, force))
        return data_root / dataset

    monkeypatch.setattr("scripts.mwe_eval.download.datasets.download", fake_download)

    exit_code = download_cli.main(
        ["--dataset", "all", "--data-root", str(tmp_path), "--force"]
    )

    assert exit_code == 0
    assert [name for name, _root, _force in calls] == ["parseme12", "streusle", "coam"]
    assert all(force for _name, _root, force in calls)


@pytest.mark.fast
def test_parseme_download_uses_git_clone_and_checkout(monkeypatch, tmp_path):
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, cwd, check, capture_output, text):
        calls.append((cmd, cwd))

    monkeypatch.setattr("scripts.mwe_eval.datasets.parseme12.subprocess.run", fake_run)

    destination = parseme12.download(tmp_path, force=False)

    assert destination == tmp_path / "parseme" / "sharedtask-data"
    assert calls[0][0][:2] == ["git", "clone"]
    assert calls[1][0][:2] == ["git", "checkout"]


@pytest.mark.fast
def test_streusle_download_uses_urlretrieve_and_extracts_archive(monkeypatch, tmp_path):
    def fake_urlretrieve(_url: str, filename: Path):
        with zipfile.ZipFile(filename, "w") as zip_ref:
            zip_ref.writestr("streusle-v5.0/dev.json", "[]")
        return str(filename), None

    monkeypatch.setattr(
        "scripts.mwe_eval.datasets.streusle.urlretrieve", fake_urlretrieve
    )

    extracted = streusle.download(tmp_path, force=True)

    assert extracted.exists()
    assert (extracted / "dev.json").exists()


@pytest.mark.fast
def test_coam_download_uses_hf_datasets_module(monkeypatch, tmp_path):
    calls: list[str] = []

    class DummyDatasetDict:
        def save_to_disk(self, path: str) -> None:
            calls.append(path)

    class DummyDatasetsModule:
        @staticmethod
        def load_dataset(name: str):
            assert name == "yusuke196/CoAM"
            return DummyDatasetDict()

    monkeypatch.setattr(
        "scripts.mwe_eval.datasets.coam._load_dataset_module",
        lambda: DummyDatasetsModule,
    )

    output = coam.download(tmp_path, force=False)
    assert output == tmp_path / "coam" / "hf_saved"
    assert calls == [str(output)]

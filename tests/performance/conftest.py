from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def reference_corpus() -> list[str]:
    corpus: list[str] = []
    this_folder = Path(__file__).parent
    txt_files = sorted((this_folder / "reference_corpus").glob("*.TXT"))
    for txt_file in txt_files:
        corpus.append(txt_file.read_text())
    return corpus

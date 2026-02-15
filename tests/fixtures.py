from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_corpus():
    """We'll use the corpus included in the original implementation,
    which is "[...] a combination of the Santa Barbara
    Corpus of Spoken American English and the spoken
    component of the ICE Canada corpus"

    > Du Bois, John W., Wallace L. Chafe, Charles Meyers, Sandra A. Thompson, Nii Martey, and Robert Englebretson (2005). Santa Barbara corpus of spoken American English. Philadelphia: Linguistic Data Consortium.

    > Newman, John and Georgie Columbus (2010). The International Corpus of English – Canada. Edmonton, Alberta: University of Alberta.

    Note that these are dialogue corpuses, so each line is often referred to as a `turn`. They've also been pre-processed with lowercasing, and punctuation replaced with alphanumeric substitutions (`_` --> `undrscr`).
    """
    corpus: list[str] = []
    this_folder = Path(__file__).parent
    txt_files = sorted((this_folder / Path("sample_corpus/")).glob("*.TXT"))
    for txt_file in txt_files:
        corpus.append(txt_file.read_text())
    return corpus

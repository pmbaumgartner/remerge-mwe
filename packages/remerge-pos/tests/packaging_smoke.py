"""Exercise an installed remerge-pos distribution with an explicit model."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from remerge_pos import Tagger
from remerge_pos.training import Config, train, write_artifact


DATA = (
    (("The", "DET"), ("watch", "NOUN")),
    (("I", "PRON"), ("watch", "VERB")),
    (("bright", "ADJ"), ("lights", "NOUN")),
    (("lights", "NOUN"), ("shine", "VERB")),
)


def main() -> None:
    with TemporaryDirectory(prefix="remerge-pos-smoke-") as directory:
        artifact = Path(directory) / "synthetic.rmsp"
        digest = write_artifact(
            train(DATA, Config(epochs=2, feature_cutoff=1, feature_buckets=4096), 7),
            artifact,
        )
        tagger = Tagger.load(artifact)
        assert tagger.artifact_sha256 == digest
        assert tagger.tag((("The", "watch"), ("I", "watch"))) == (
            ("DET", "NOUN"),
            ("PRON", "NOUN"),
        )
    print("explicit-model inference smoke passed")


if __name__ == "__main__":
    main()

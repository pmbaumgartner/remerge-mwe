#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python - <<'PY'
import json
from pathlib import Path

from remerge import run


def summarize_winners(winners):
    return [
        {
            "merged_word": list(winner.merged_lexeme.word),
            "merge_token_count": winner.merge_token_count,
        }
        for winner in winners
    ]


sample_corpus = [
    txt.read_text() for txt in sorted(Path("tests/sample_corpus").glob("*.TXT"))
]

parity = {
    "log_likelihood": summarize_winners(
        run(sample_corpus, 5, method="log_likelihood", min_count=0)
    ),
    "frequency": summarize_winners(run(sample_corpus, 5, method="frequency", min_count=0)),
    "npmi": summarize_winners(run(sample_corpus, 5, method="npmi", min_count=25)),
    "tie_deterministic": {},
}

tie_corpus = ["a b", "c d"]
for method in ["frequency", "log_likelihood", "npmi"]:
    winner = run(tie_corpus, 1, method=method)[0]
    parity["tie_deterministic"][method] = list(winner.merged_lexeme.word)

output_path = Path("tests/parity_expected.json")
output_path.write_text(json.dumps(parity, indent=2) + "\n")
print(f"Wrote {output_path}")
PY

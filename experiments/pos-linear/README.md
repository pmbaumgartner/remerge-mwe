# Linear POS research experiment

This directory preserves the rejected `RMPOS001` trainer and Rust loader as an
independent research workspace. It is not part of the production Cargo package,
Python package, normal tests, release CI, or artifact checks.

The experiment uses only development data. Do not evaluate the protected final
split without a new Kata outcome and oracle decision.

Run the retained checks independently:

```sh
cargo test --manifest-path experiments/pos-linear/Cargo.toml
uv run --directory experiments/pos-linear pytest -q
```

Run `python experiments/pos-linear/train.py --help` for the deterministic
trainer interface. The frozen dataset revisions, seed, candidate results, and
artifact digests are recorded in `docs/pos_linear_dev_evaluation.md`.

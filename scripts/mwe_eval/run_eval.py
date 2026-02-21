from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any, cast

from remerge import run as remerge_run

from scripts.mwe_eval import datasets
from scripts.mwe_eval.datasets.base import SplitData, count_gold_mentions
from scripts.mwe_eval.match import (
    DecodePolicy,
    build_trie,
    decode_spans,
    find_spans,
    lexicon_from_winners,
    spans_to_types,
    suppress_subspan_types as suppress_subspan_types_fn,
)
from scripts.mwe_eval.metrics import PRFCounts, prf_from_counts, prf_from_sets

_DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "mwe_eval"
).resolve()
_STOPWORD_POLICIES = ("none", "block_stopword_stopword", "block_any_stopword")
_DECODE_POLICIES: tuple[DecodePolicy, ...] = (
    "all",
    "longest_per_start",
    "maximal_non_overlapping",
)
_SEARCH_STRATEGIES = ("greedy", "beam")
_DEFAULT_EVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "up",
        "off",
        "out",
        "over",
        "under",
        "into",
        "onto",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "so",
        "than",
        "then",
        "too",
        "very",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
        "because",
        "while",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "there",
        "here",
        "just",
        "also",
        "'m",
        "'re",
        "'ve",
        "'ll",
        "'d",
        "n't",
    }
)
_PUNCT_ONLY_RE = re.compile(r"^\W+$")


@dataclass(frozen=True, slots=True)
class EvalRunConfig:
    dataset: str
    split: str
    iterations: int
    method: str = "log_likelihood"
    min_count: int = 0
    min_score: float | None = None
    rescore_interval: int = 25
    min_range: int = 1
    range_alpha: float = 0.0
    min_p_ab: float | None = None
    min_p_ba: float | None = None
    min_merge_count: int = 1
    min_winner_score_output: float | None = None
    min_winner_range_output: int = 1
    search_strategy: str = "greedy"
    beam_width: int = 1
    beam_top_m: int = 8
    consensus_runs: tuple[dict[str, Any], ...] | None = None
    consensus_min_run_support: int = 2
    consensus_min_method_support: int = 1
    decode_policy: DecodePolicy = "all"
    suppress_subspan_types: bool = False
    parseme_langs: tuple[str, ...] = ("EN",)
    stopword_policy: str = "none"
    stopwords: tuple[str, ...] | None = None
    block_punct_only: bool = True
    prediction_stopword_policy: str = "none"
    prediction_block_punct_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "iterations": self.iterations,
            "method": self.method,
            "min_count": self.min_count,
            "min_score": self.min_score,
            "rescore_interval": self.rescore_interval,
            "min_range": self.min_range,
            "range_alpha": self.range_alpha,
            "min_p_ab": self.min_p_ab,
            "min_p_ba": self.min_p_ba,
            "min_merge_count": self.min_merge_count,
            "min_winner_score_output": self.min_winner_score_output,
            "min_winner_range_output": self.min_winner_range_output,
            "search_strategy": self.search_strategy,
            "beam_width": self.beam_width,
            "beam_top_m": self.beam_top_m,
            "consensus_runs": list(self.consensus_runs)
            if self.consensus_runs is not None
            else None,
            "consensus_min_run_support": self.consensus_min_run_support,
            "consensus_min_method_support": self.consensus_min_method_support,
            "decode_policy": self.decode_policy,
            "suppress_subspan_types": self.suppress_subspan_types,
            "parseme_langs": list(self.parseme_langs),
            "stopword_policy": self.stopword_policy,
            "stopwords": list(self.stopwords) if self.stopwords is not None else None,
            "block_punct_only": self.block_punct_only,
            "prediction_stopword_policy": self.prediction_stopword_policy,
            "prediction_block_punct_only": self.prediction_block_punct_only,
        }


def _validate_probability(name: str, value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive.")
    return out


def _normalize_consensus_runs(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    explicit_runs = spec.get("consensus_runs")
    method_list = spec.get("consensus_methods")

    runs: list[dict[str, Any]] = []
    if explicit_runs is not None:
        if not isinstance(explicit_runs, list):
            raise ValueError("consensus_runs must be a list[object] when provided.")
        for run in explicit_runs:
            if not isinstance(run, dict):
                raise ValueError("Each consensus_runs entry must be an object.")
            runs.append(dict(run))

    if method_list is not None:
        if not isinstance(method_list, list):
            raise ValueError("consensus_methods must be a list[str] when provided.")
        for method in method_list:
            runs.append({"method": str(method)})

    if not runs:
        return None
    return tuple(runs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MWE evaluation suite.")
    parser.add_argument("--dataset", choices=[*datasets.dataset_names(), "all"])
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--method")
    parser.add_argument("--min-count", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--rescore-interval", type=int)
    parser.add_argument("--min-range", type=int)
    parser.add_argument("--range-alpha", type=float)
    parser.add_argument("--min-p-ab", type=float)
    parser.add_argument("--min-p-ba", type=float)
    parser.add_argument("--min-merge-count", type=int)
    parser.add_argument("--winner-min-score-output", type=float)
    parser.add_argument("--winner-min-range-output", type=int)
    parser.add_argument(
        "--search-strategy",
        choices=_SEARCH_STRATEGIES,
        help="Core search strategy.",
    )
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--beam-top-m", type=int)
    parser.add_argument(
        "--consensus-method",
        dest="consensus_methods",
        action="append",
        default=None,
        help="Consensus convenience: add a run with this method (repeatable).",
    )
    parser.add_argument("--consensus-min-run-support", type=int)
    parser.add_argument("--consensus-min-method-support", type=int)
    parser.add_argument(
        "--decode-policy",
        choices=list(_DECODE_POLICIES),
        help="Span decode policy for mention/type scoring.",
    )
    parser.add_argument(
        "--suppress-subspan-types",
        dest="suppress_subspan_types",
        action="store_true",
        default=None,
        help="Drop predicted type tuples that are contiguous subspans of longer types.",
    )
    parser.add_argument(
        "--no-suppress-subspan-types",
        dest="suppress_subspan_types",
        action="store_false",
        help="Do not suppress subspan predicted types.",
    )
    parser.add_argument(
        "--stopword-policy",
        choices=_STOPWORD_POLICIES,
        help="Engine-level stopword policy.",
    )
    parser.add_argument(
        "--stopword",
        dest="stopwords",
        action="append",
        default=None,
        help="Engine-level stopword token (repeatable).",
    )
    parser.add_argument(
        "--block-punct-only",
        dest="block_punct_only",
        action="store_true",
        default=None,
        help="Engine-level punctuation-only candidate blocking.",
    )
    parser.add_argument(
        "--no-block-punct-only",
        dest="block_punct_only",
        action="store_false",
        help="Disable engine-level punctuation-only candidate blocking.",
    )
    parser.add_argument(
        "--prediction-stopword-policy",
        choices=_STOPWORD_POLICIES,
        help="Eval-time predicted-span stopword filter policy.",
    )
    parser.add_argument(
        "--prediction-block-punct-only",
        dest="prediction_block_punct_only",
        action="store_true",
        default=None,
        help="Eval-time predicted-span punctuation-only filtering.",
    )
    parser.add_argument(
        "--no-prediction-block-punct-only",
        dest="prediction_block_punct_only",
        action="store_false",
        help="Disable eval-time predicted-span punctuation-only filtering.",
    )
    parser.add_argument(
        "--parseme-lang",
        dest="parseme_langs",
        action="append",
        default=None,
        help="PARSEME language code (repeatable). Default: EN.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help=f"Dataset root directory (default: {_DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    parser.add_argument(
        "--config", type=Path, help="Optional JSON config (single run or runs[])."
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download dataset automatically when local files are missing.",
    )
    return parser


def _require_fields(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing: list[str] = []
    for key in ("dataset", "split", "iterations"):
        if getattr(args, key) is None:
            missing.append(f"--{key.replace('_', '-')}")
    if missing:
        parser.error(
            "Missing required arguments when --config is not provided: "
            + ", ".join(missing)
        )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_run_spec(spec: dict[str, Any]) -> EvalRunConfig:
    dataset = str(spec["dataset"])
    split = str(spec["split"])
    iterations = int(spec["iterations"])

    if iterations < 0:
        raise ValueError("iterations must be >= 0.")

    method = str(spec.get("method", "log_likelihood"))
    min_count = int(spec.get("min_count", 0))
    min_score_raw = spec.get("min_score")
    min_score = float(min_score_raw) if min_score_raw is not None else None
    rescore_interval = int(spec.get("rescore_interval", 25))

    min_range = int(spec.get("min_range", 1))
    if min_range < 1:
        raise ValueError("min_range must be >= 1.")

    range_alpha = float(spec.get("range_alpha", 0.0))
    if not math.isfinite(range_alpha):
        raise ValueError("range_alpha must be finite.")
    if range_alpha < 0:
        raise ValueError("range_alpha must be >= 0.")

    min_p_ab = _validate_probability("min_p_ab", spec.get("min_p_ab"))
    min_p_ba = _validate_probability("min_p_ba", spec.get("min_p_ba"))

    min_merge_count = int(spec.get("min_merge_count", 1))
    if min_merge_count < 1:
        raise ValueError("min_merge_count must be >= 1.")

    min_winner_score_output_raw = spec.get("min_winner_score_output")
    min_winner_score_output = (
        float(min_winner_score_output_raw)
        if min_winner_score_output_raw is not None
        else None
    )
    if min_winner_score_output is not None and not math.isfinite(
        min_winner_score_output
    ):
        raise ValueError("min_winner_score_output must be finite.")

    min_winner_range_output = int(spec.get("min_winner_range_output", 1))
    if min_winner_range_output < 1:
        raise ValueError("min_winner_range_output must be >= 1.")

    search_strategy = str(spec.get("search_strategy", "greedy"))
    if search_strategy not in _SEARCH_STRATEGIES:
        raise ValueError(
            "search_strategy must be one of: "
            + ", ".join(repr(value) for value in _SEARCH_STRATEGIES)
        )

    beam_width = int(spec.get("beam_width", 1))
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")

    beam_top_m = int(spec.get("beam_top_m", 8))
    if beam_top_m < 1:
        raise ValueError("beam_top_m must be >= 1.")

    consensus_runs = _normalize_consensus_runs(spec)
    consensus_min_run_support = int(spec.get("consensus_min_run_support", 2))
    if consensus_min_run_support < 1:
        raise ValueError("consensus_min_run_support must be >= 1.")

    consensus_min_method_support = int(spec.get("consensus_min_method_support", 1))
    if consensus_min_method_support < 1:
        raise ValueError("consensus_min_method_support must be >= 1.")

    decode_policy_raw = str(spec.get("decode_policy", "all"))
    if decode_policy_raw not in _DECODE_POLICIES:
        raise ValueError(
            "decode_policy must be one of: "
            + ", ".join(repr(value) for value in _DECODE_POLICIES)
        )
    decode_policy = cast(DecodePolicy, decode_policy_raw)

    suppress_subspan_types = bool(spec.get("suppress_subspan_types", False))

    parseme_langs_raw = spec.get("parseme_langs", ["EN"])
    parseme_langs = tuple(str(lang).upper() for lang in parseme_langs_raw)

    stopword_policy = str(spec.get("stopword_policy", "none"))
    if stopword_policy not in _STOPWORD_POLICIES:
        raise ValueError(
            "stopword_policy must be one of: "
            + ", ".join(repr(value) for value in _STOPWORD_POLICIES)
        )

    stopwords_raw = spec.get("stopwords")
    stopwords: tuple[str, ...] | None
    if stopwords_raw is None:
        stopwords = None
    elif isinstance(stopwords_raw, list):
        stopwords = tuple(str(token) for token in stopwords_raw)
    else:
        raise ValueError("stopwords must be a list[str] when provided.")

    block_punct_only = bool(spec.get("block_punct_only", True))

    prediction_stopword_policy = str(spec.get("prediction_stopword_policy", "none"))
    if prediction_stopword_policy not in _STOPWORD_POLICIES:
        raise ValueError(
            "prediction_stopword_policy must be one of: "
            + ", ".join(repr(value) for value in _STOPWORD_POLICIES)
        )

    prediction_block_punct_only = bool(spec.get("prediction_block_punct_only", False))

    return EvalRunConfig(
        dataset=dataset,
        split=split,
        iterations=iterations,
        method=method,
        min_count=min_count,
        min_score=min_score,
        rescore_interval=rescore_interval,
        min_range=min_range,
        range_alpha=range_alpha,
        min_p_ab=min_p_ab,
        min_p_ba=min_p_ba,
        min_merge_count=min_merge_count,
        min_winner_score_output=min_winner_score_output,
        min_winner_range_output=min_winner_range_output,
        search_strategy=search_strategy,
        beam_width=beam_width,
        beam_top_m=beam_top_m,
        consensus_runs=consensus_runs,
        consensus_min_run_support=consensus_min_run_support,
        consensus_min_method_support=consensus_min_method_support,
        decode_policy=decode_policy,
        suppress_subspan_types=suppress_subspan_types,
        parseme_langs=parseme_langs,
        stopword_policy=stopword_policy,
        stopwords=stopwords,
        block_punct_only=block_punct_only,
        prediction_stopword_policy=prediction_stopword_policy,
        prediction_block_punct_only=prediction_block_punct_only,
    )


def _apply_cli_overrides(
    run_spec: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    overridden = dict(run_spec)
    for key, arg_name in (
        ("dataset", "dataset"),
        ("split", "split"),
        ("iterations", "iterations"),
        ("method", "method"),
        ("min_count", "min_count"),
        ("min_score", "min_score"),
        ("rescore_interval", "rescore_interval"),
        ("min_range", "min_range"),
        ("range_alpha", "range_alpha"),
        ("min_p_ab", "min_p_ab"),
        ("min_p_ba", "min_p_ba"),
        ("min_merge_count", "min_merge_count"),
        ("min_winner_score_output", "winner_min_score_output"),
        ("min_winner_range_output", "winner_min_range_output"),
        ("search_strategy", "search_strategy"),
        ("beam_width", "beam_width"),
        ("beam_top_m", "beam_top_m"),
        ("consensus_min_run_support", "consensus_min_run_support"),
        ("consensus_min_method_support", "consensus_min_method_support"),
        ("decode_policy", "decode_policy"),
        ("stopword_policy", "stopword_policy"),
        ("prediction_stopword_policy", "prediction_stopword_policy"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overridden[key] = value

    if args.parseme_langs is not None:
        overridden["parseme_langs"] = args.parseme_langs
    if args.stopwords is not None:
        overridden["stopwords"] = args.stopwords
    if args.block_punct_only is not None:
        overridden["block_punct_only"] = args.block_punct_only
    if args.prediction_block_punct_only is not None:
        overridden["prediction_block_punct_only"] = args.prediction_block_punct_only
    if args.suppress_subspan_types is not None:
        overridden["suppress_subspan_types"] = args.suppress_subspan_types
    if args.consensus_methods is not None:
        overridden["consensus_methods"] = args.consensus_methods

    return overridden


def _load_run_configs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[EvalRunConfig]:
    if args.config is None:
        _require_fields(parser, args)
        spec: dict[str, Any] = {
            "dataset": args.dataset,
            "split": args.split,
            "iterations": args.iterations,
        }
        if args.method is not None:
            spec["method"] = args.method
        if args.min_count is not None:
            spec["min_count"] = args.min_count
        if args.min_score is not None:
            spec["min_score"] = args.min_score
        if args.rescore_interval is not None:
            spec["rescore_interval"] = args.rescore_interval
        if args.min_range is not None:
            spec["min_range"] = args.min_range
        if args.range_alpha is not None:
            spec["range_alpha"] = args.range_alpha
        if args.min_p_ab is not None:
            spec["min_p_ab"] = args.min_p_ab
        if args.min_p_ba is not None:
            spec["min_p_ba"] = args.min_p_ba
        if args.min_merge_count is not None:
            spec["min_merge_count"] = args.min_merge_count
        if args.winner_min_score_output is not None:
            spec["min_winner_score_output"] = args.winner_min_score_output
        if args.winner_min_range_output is not None:
            spec["min_winner_range_output"] = args.winner_min_range_output
        if args.search_strategy is not None:
            spec["search_strategy"] = args.search_strategy
        if args.beam_width is not None:
            spec["beam_width"] = args.beam_width
        if args.beam_top_m is not None:
            spec["beam_top_m"] = args.beam_top_m
        if args.consensus_methods is not None:
            spec["consensus_methods"] = args.consensus_methods
        if args.consensus_min_run_support is not None:
            spec["consensus_min_run_support"] = args.consensus_min_run_support
        if args.consensus_min_method_support is not None:
            spec["consensus_min_method_support"] = args.consensus_min_method_support
        if args.decode_policy is not None:
            spec["decode_policy"] = args.decode_policy
        if args.suppress_subspan_types is not None:
            spec["suppress_subspan_types"] = args.suppress_subspan_types
        if args.parseme_langs is not None:
            spec["parseme_langs"] = args.parseme_langs
        if args.stopword_policy is not None:
            spec["stopword_policy"] = args.stopword_policy
        if args.stopwords is not None:
            spec["stopwords"] = args.stopwords
        if args.block_punct_only is not None:
            spec["block_punct_only"] = args.block_punct_only
        if args.prediction_stopword_policy is not None:
            spec["prediction_stopword_policy"] = args.prediction_stopword_policy
        if args.prediction_block_punct_only is not None:
            spec["prediction_block_punct_only"] = args.prediction_block_punct_only
        return [_normalize_run_spec(spec)]

    payload = _load_json(args.config)
    run_specs: list[dict[str, Any]]
    if isinstance(payload, dict) and "runs" in payload:
        base = {k: v for k, v in payload.items() if k != "runs"}
        runs = payload["runs"]
        if not isinstance(runs, list):
            raise ValueError("Config key 'runs' must be a list.")
        run_specs = []
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError("Each run in config 'runs' must be an object.")
            merged = dict(base)
            merged.update(run)
            run_specs.append(merged)
    elif isinstance(payload, dict):
        run_specs = [payload]
    elif isinstance(payload, list):
        run_specs = []
        for run in payload:
            if not isinstance(run, dict):
                raise ValueError("Each config entry must be an object.")
            run_specs.append(run)
    else:
        raise ValueError("Config must be an object, list, or object with runs[].")

    return [_normalize_run_spec(_apply_cli_overrides(spec, args)) for spec in run_specs]


def _load_dataset_split(
    dataset_name: str,
    data_root: Path,
    split: str,
    *,
    parseme_langs: tuple[str, ...],
    download_missing: bool,
) -> SplitData:
    kwargs: dict[str, Any] = {}
    if dataset_name == "parseme12":
        kwargs["parseme_langs"] = list(parseme_langs)

    try:
        return datasets.load_split(dataset_name, data_root, split, **kwargs)
    except FileNotFoundError:
        if not download_missing:
            raise
        datasets.download(dataset_name, data_root, force=False)
        return datasets.load_split(dataset_name, data_root, split, **kwargs)


def _token_is_punct_only(token: str) -> bool:
    return bool(_PUNCT_ONLY_RE.fullmatch(token))


def _is_stopword_token(token: str, stopwords: frozenset[str]) -> bool:
    return token.lower() in stopwords


def _span_passes_prediction_filters(
    tokens: tuple[str, ...],
    *,
    stopword_policy: str,
    block_punct_only: bool,
) -> bool:
    if (
        block_punct_only
        and tokens
        and all(_token_is_punct_only(token) for token in tokens)
    ):
        return False

    if stopword_policy == "none":
        return True

    if stopword_policy == "block_stopword_stopword":
        return not all(
            _is_stopword_token(token, _DEFAULT_EVAL_STOPWORDS)
            or _token_is_punct_only(token)
            for token in tokens
        )

    if stopword_policy == "block_any_stopword":
        return not any(
            _is_stopword_token(token, _DEFAULT_EVAL_STOPWORDS) for token in tokens
        )

    raise ValueError(f"Unsupported prediction stopword policy: {stopword_policy!r}")


def _score_split(
    split_data: SplitData,
    winners: list[Any],
    *,
    prediction_stopword_policy: str,
    prediction_block_punct_only: bool,
    decode_policy: DecodePolicy,
    suppress_subspan_types: bool,
) -> dict[str, Any]:
    predicted_lexicon = lexicon_from_winners(winners)
    trie = build_trie(predicted_lexicon)

    mention_tp = 0
    mention_fp = 0
    mention_fn = 0

    gold_types: set[tuple[str, ...]] = set()
    predicted_types: set[tuple[str, ...]] = set()

    for sentence in split_data.sentences:
        gold_spans = sentence.contiguous_spans()
        predicted_spans_all = find_spans(sentence.tokens, trie)
        predicted_spans_filtered = {
            span
            for span in predicted_spans_all
            if _span_passes_prediction_filters(
                sentence.tokens[span[0] : span[1]],
                stopword_policy=prediction_stopword_policy,
                block_punct_only=prediction_block_punct_only,
            )
        }
        predicted_spans = decode_spans(predicted_spans_filtered, policy=decode_policy)

        mention_tp += len(gold_spans & predicted_spans)
        mention_fp += len(predicted_spans - gold_spans)
        mention_fn += len(gold_spans - predicted_spans)

        gold_types |= sentence.contiguous_types()
        predicted_types |= spans_to_types(sentence.tokens, predicted_spans)

    if suppress_subspan_types:
        predicted_types = suppress_subspan_types_fn(predicted_types)

    mention_prf = prf_from_counts(mention_tp, mention_fp, mention_fn)
    type_prf = prf_from_sets(gold_types, predicted_types)

    gold_total, gold_contiguous, gold_discontinuous = count_gold_mentions(split_data)

    return {
        "counts": {
            "gold_total": gold_total,
            "gold_contiguous": gold_contiguous,
            "gold_discontinuous": gold_discontinuous,
            "predicted_total": mention_tp + mention_fp,
            "matched": mention_tp,
            "type_gold_total": len(gold_types),
            "type_predicted_total": len(predicted_types),
            "type_matched": type_prf.tp,
        },
        "filtering": {
            "prediction_stopword_policy": prediction_stopword_policy,
            "prediction_block_punct_only": prediction_block_punct_only,
            "decode_policy": decode_policy,
            "suppress_subspan_types": suppress_subspan_types,
            "predicted_lexicon_size": len(predicted_lexicon),
        },
        "metrics": {
            "mention": mention_prf.to_dict(),
            "type": type_prf.to_dict(),
        },
    }


def _evaluate_single_dataset(
    dataset_name: str,
    config: EvalRunConfig,
    *,
    data_root: Path,
    download_missing: bool,
) -> dict[str, Any]:
    dataset_start = perf_counter()

    load_start = perf_counter()
    split_data = _load_dataset_split(
        dataset_name,
        data_root,
        config.split,
        parseme_langs=config.parseme_langs,
        download_missing=download_missing,
    )
    load_seconds = perf_counter() - load_start

    corpus = [" ".join(sentence.tokens) for sentence in split_data.sentences]
    stopwords_for_run: list[str] | None
    if config.stopwords is not None:
        stopwords_for_run = list(config.stopwords)
    elif config.stopword_policy != "none":
        stopwords_for_run = sorted(_DEFAULT_EVAL_STOPWORDS)
    else:
        stopwords_for_run = None

    run_start = perf_counter()
    winners = remerge_run(
        corpus,
        config.iterations,
        method=config.method,
        min_count=config.min_count,
        min_score=config.min_score,
        rescore_interval=config.rescore_interval,
        min_range=config.min_range,
        range_alpha=config.range_alpha,
        min_p_ab=config.min_p_ab,
        min_p_ba=config.min_p_ba,
        min_merge_count=config.min_merge_count,
        min_winner_score_output=config.min_winner_score_output,
        min_winner_range_output=config.min_winner_range_output,
        search_strategy=config.search_strategy,
        beam_width=config.beam_width,
        beam_top_m=config.beam_top_m,
        consensus_runs=list(config.consensus_runs)
        if config.consensus_runs is not None
        else None,
        consensus_min_run_support=config.consensus_min_run_support,
        consensus_min_method_support=config.consensus_min_method_support,
        splitter="delimiter",
        line_delimiter=None,
        stopwords=stopwords_for_run,
        stopword_policy=config.stopword_policy,
        block_punct_only=config.block_punct_only,
        on_exhausted="stop",
    )
    run_seconds = perf_counter() - run_start

    scores = _score_split(
        split_data,
        winners,
        prediction_stopword_policy=config.prediction_stopword_policy,
        prediction_block_punct_only=config.prediction_block_punct_only,
        decode_policy=config.decode_policy,
        suppress_subspan_types=config.suppress_subspan_types,
    )

    return {
        "dataset": dataset_name,
        "split": config.split,
        "config": config.to_dict(),
        **scores,
        "winners": {
            "count": len(winners),
            "min_winner_score_output": config.min_winner_score_output,
            "min_winner_range_output": config.min_winner_range_output,
        },
        "timing": {
            "dataset_load_seconds": load_seconds,
            "run_seconds": run_seconds,
            "total_seconds": perf_counter() - dataset_start,
        },
    }


def _aggregate_prf(results: list[dict[str, Any]], level: str) -> PRFCounts:
    tp = sum(int(result["metrics"][level]["tp"]) for result in results)
    fp = sum(int(result["metrics"][level]["fp"]) for result in results)
    fn = sum(int(result["metrics"][level]["fn"]) for result in results)
    return prf_from_counts(tp=tp, fp=fp, fn=fn)


def _aggregate_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "gold_total",
        "gold_contiguous",
        "gold_discontinuous",
        "predicted_total",
        "matched",
        "type_gold_total",
        "type_predicted_total",
        "type_matched",
    )
    return {key: sum(int(result["counts"][key]) for result in results) for key in keys}


def evaluate_run(
    config: EvalRunConfig,
    *,
    data_root: Path,
    download_missing: bool,
) -> dict[str, Any]:
    selected_datasets = datasets.expand_dataset(config.dataset)
    total_start = perf_counter()

    results = [
        _evaluate_single_dataset(
            dataset_name,
            config,
            data_root=data_root,
            download_missing=download_missing,
        )
        for dataset_name in selected_datasets
    ]

    if config.dataset != "all":
        return results[0]

    mention_summary = _aggregate_prf(results, "mention")
    type_summary = _aggregate_prf(results, "type")

    return {
        "dataset": "all",
        "split": config.split,
        "config": config.to_dict(),
        "results": results,
        "summary": {
            "counts": _aggregate_counts(results),
            "metrics": {
                "mention": mention_summary.to_dict(),
                "type": type_summary.to_dict(),
            },
            "timing": {
                "total_seconds": perf_counter() - total_start,
            },
        },
    }


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configs = _load_run_configs(parser, args)
    data_root: Path = args.data_root.resolve()

    run_results = [
        evaluate_run(
            config,
            data_root=data_root,
            download_missing=args.download_missing,
        )
        for config in configs
    ]

    if len(run_results) == 1:
        output_payload: Any = run_results[0]
    else:
        output_payload = {"runs": run_results}

    rendered = _dump_json(output_payload)

    if args.output is not None:
        output_path: Path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote evaluation results to {output_path}")
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

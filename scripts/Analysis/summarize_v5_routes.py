#!/usr/bin/env python3
"""Summarize FedPLoRA-Oneshot v5 route diagnostics from metrics JSON files.

Usage:
    python scripts/Analysis/summarize_v5_routes.py \
        --inputs "artifacts_35c/v4_sft_metrics/*/*.json" \
        --out artifacts_35c/v5_route_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _last_round_records(blob):
    if "per_seed" in blob:
        for rec in blob.get("per_seed", []) or []:
            rounds = rec.get("rounds", []) or []
            if rounds:
                yield rec, rounds[-1]
    else:
        rounds = blob.get("rounds", []) or []
        if rounds:
            yield blob, rounds[-1]


def _safe_float(value, default=float("nan")):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _route_row(path, blob, seed_rec, last_round):
    stats = last_round.get("v5_route_stats") or {}
    if not stats:
        return None

    route_counts = stats.get("route_counts") or {}
    total = sum(int(v) for v in route_counts.values()) if route_counts else 0

    def frac(name):
        if total <= 0:
            return float("nan")
        return float(route_counts.get(name, 0)) / float(total)

    post_align = stats.get("post_align") or {}
    args = seed_rec.get("args", {}) if isinstance(seed_rec, dict) else {}
    agg_type = blob.get("agg_type") or args.get("agg_type", "?")
    seed = seed_rec.get("seed", args.get("seed", "?")) if isinstance(seed_rec, dict) else "?"

    return {
        "source": str(Path(path).name),
        "parent": str(Path(path).parent.name),
        "agg_type": agg_type,
        "seed": seed,
        "scope": stats.get("scope", ""),
        "num_routes": int(stats.get("num_routes", 0) or 0),
        "num_cached_searches": int(stats.get("num_cached_searches", 0) or 0),
        "global_frac": frac("global"),
        "mixed_frac": frac("mixed"),
        "local_frac": frac("local"),
        "mean_eta": _safe_float(stats.get("mean_eta")),
        "min_eta": _safe_float(stats.get("min_eta")),
        "max_eta": _safe_float(stats.get("max_eta")),
        "tie_breaker": stats.get("tie_breaker", ""),
        "tie_margin": _safe_float(stats.get("tie_margin")),
        "search_max_batches": int(stats.get("search_max_batches", 0) or 0),
        "post_align_enabled": bool(post_align.get("enabled", False)),
        "num_align_states": int(post_align.get("num_align_states", 0) or 0),
        "num_clients_aligned": int(post_align.get("num_clients_aligned", 0) or 0),
        "post_align_mean_final_loss": _safe_float(post_align.get("mean_final_loss")),
        "macro_acc": _safe_float(last_round.get("domain_macro_token_accuracy")),
        "worst_acc": _safe_float(last_round.get("worst_domain_token_accuracy")),
        "macro_ppl": _safe_float(last_round.get("domain_macro_perplexity")),
        "worst_ppl": _safe_float(last_round.get("worst_domain_perplexity")),
    }


def _fmt(row):
    out = dict(row)
    for key, value in list(out.items()):
        if isinstance(value, float):
            if math.isnan(value):
                out[key] = "nan"
            elif key.endswith("_frac") or key in {"mean_eta", "min_eta", "max_eta", "macro_acc", "worst_acc"}:
                out[key] = f"{value:.5f}"
            else:
                out[key] = f"{value:.4f}"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON paths or glob patterns.")
    parser.add_argument("--out", default="-", help="CSV output path; '-' prints to stdout.")
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        if not matched and os.path.isfile(pattern):
            matched = [pattern]
        paths.extend(matched)
    paths = sorted(set(paths))

    rows = []
    for path in paths:
        try:
            blob = _load_json(path)
            for seed_rec, last_round in _last_round_records(blob):
                row = _route_row(path, blob, seed_rec, last_round)
                if row is not None:
                    rows.append(row)
        except Exception as exc:
            print(f"[warn] failed on {path}: {exc}")

    fieldnames = [
        "source",
        "parent",
        "agg_type",
        "seed",
        "scope",
        "num_routes",
        "num_cached_searches",
        "global_frac",
        "mixed_frac",
        "local_frac",
        "mean_eta",
        "min_eta",
        "max_eta",
        "tie_breaker",
        "tie_margin",
        "search_max_batches",
        "post_align_enabled",
        "num_align_states",
        "num_clients_aligned",
        "post_align_mean_final_loss",
        "macro_acc",
        "worst_acc",
        "macro_ppl",
        "worst_ppl",
    ]

    out_stream = open(args.out, "w", encoding="utf-8", newline="") if args.out != "-" else None
    try:
        import sys

        target = out_stream if out_stream is not None else sys.stdout
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_fmt(row))
        if out_stream is not None:
            print(f"wrote {len(rows)} rows to {args.out}")
    finally:
        if out_stream is not None:
            out_stream.close()


if __name__ == "__main__":
    main()

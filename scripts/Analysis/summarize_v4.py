#!/usr/bin/env python3
"""Summarize v2 / v3 / v4 result JSONs into a side-by-side comparison table.

Usage:
    python scripts/Analysis/summarize_v4.py \
        --inputs ../Result/*.json artifacts/v4_sft_metrics/*.json \
        --out artifacts/v4_summary.csv

Each JSON is expected to follow either:
  - v2/v3 format: top-level `args`, `rounds`, `communication`
  - v4 multi-seed format: `per_seed: [...]`, `agg_type`, `seeds`

Outputs a CSV with columns:
  agg_type, n_clients, seed(s), eval_batches, macro_acc, macro_acc_std,
  worst_acc, worst_acc_std, hard_avg_loss, hard_avg_loss_std,
  comm_down_MB, comm_up_MB
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from pathlib import Path

HARD_DOMAINS = ("legal", "finance", "medical")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _extract_v4_seed_block(per_seed_record):
    if not per_seed_record.get("rounds"):
        return None
    last = per_seed_record["rounds"][-1]
    domain = last.get("domain_metrics", {})
    losses = [domain[d]["loss"] for d in HARD_DOMAINS if d in domain]
    hard_avg = float(statistics.mean(losses)) if losses else float("nan")
    return {
        "macro_acc": last.get("domain_macro_token_accuracy"),
        "worst_acc": last.get("worst_domain_token_accuracy"),
        "macro_ppl": last.get("domain_macro_perplexity"),
        "worst_ppl": last.get("worst_domain_perplexity"),
        "hard_avg_loss": hard_avg,
        "domain_metrics": domain,
    }


def _summarize_one(path, blob):
    rows = []
    if "per_seed" in blob:
        # v4 multi-seed format
        records = blob.get("per_seed", [])
        agg = blob.get("agg_type") or (records[0].get("args", {}) if records else {}).get("agg_type", "?")
        per_seed_metrics = []
        for rec in records:
            m = _extract_v4_seed_block(rec)
            if m is not None:
                per_seed_metrics.append(m)
        if not per_seed_metrics:
            return rows
        def _mean_std(field):
            vals = [m[field] for m in per_seed_metrics if m.get(field) is not None
                    and not (isinstance(m[field], float) and math.isnan(m[field]))]
            if not vals:
                return None, None
            if len(vals) == 1:
                return float(vals[0]), 0.0
            return float(statistics.mean(vals)), float(statistics.stdev(vals))
        macro_acc, macro_acc_std = _mean_std("macro_acc")
        worst_acc, worst_acc_std = _mean_std("worst_acc")
        hard_avg, hard_avg_std = _mean_std("hard_avg_loss")
        comm = (records[0].get("communication", {}) if records else {})
        n_clients = (records[0].get("args", {}) if records else {}).get("num_clients", "?")
        rows.append({
            "source": Path(path).name,
            "agg_type": agg,
            "n_clients": n_clients,
            "seeds": blob.get("seeds", []),
            "macro_acc": macro_acc, "macro_acc_std": macro_acc_std,
            "worst_acc": worst_acc, "worst_acc_std": worst_acc_std,
            "hard_avg_loss": hard_avg, "hard_avg_loss_std": hard_avg_std,
            "comm_down_MB": comm.get("down_bytes_per_client", 0) / (1024 ** 2),
            "comm_up_MB":   comm.get("up_bytes_per_client", 0) / (1024 ** 2),
        })
    else:
        # v2 / v3 single-seed format
        last = (blob.get("rounds") or [{}])[-1]
        domain = last.get("domain_metrics", {})
        losses = [domain[d]["loss"] for d in HARD_DOMAINS if d in domain]
        hard_avg = float(statistics.mean(losses)) if losses else float("nan")
        comm = blob.get("communication", {})
        rows.append({
            "source": Path(path).name,
            "agg_type": blob.get("args", {}).get("agg_type", "?"),
            "n_clients": blob.get("args", {}).get("num_clients", "?"),
            "seeds": [blob.get("args", {}).get("seed", "?")],
            "macro_acc": last.get("domain_macro_token_accuracy"),
            "macro_acc_std": 0.0,
            "worst_acc": last.get("worst_domain_token_accuracy"),
            "worst_acc_std": 0.0,
            "hard_avg_loss": hard_avg,
            "hard_avg_loss_std": 0.0,
            "comm_down_MB": comm.get("down_bytes_per_client", 0) / (1024 ** 2),
            "comm_up_MB":   comm.get("up_bytes_per_client", 0) / (1024 ** 2),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths or glob patterns matching JSON files to summarize.")
    parser.add_argument("--out", type=str, default="-",
                        help="Output CSV path; '-' prints to stdout.")
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        if not matched and os.path.isfile(pattern):
            matched = [pattern]
        if not matched:
            print(f"[warn] no matches for: {pattern}")
        paths.extend(matched)
    paths = sorted(set(paths))

    rows = []
    for p in paths:
        try:
            rows.extend(_summarize_one(p, _load(p)))
        except Exception as exc:
            print(f"[warn] failed on {p}: {exc}")

    if not rows:
        print("no results")
        return

    fieldnames = ["source", "agg_type", "n_clients", "seeds",
                  "macro_acc", "macro_acc_std",
                  "worst_acc", "worst_acc_std",
                  "hard_avg_loss", "hard_avg_loss_std",
                  "comm_down_MB", "comm_up_MB"]

    def _fmt(row):
        out = dict(row)
        for k in ("macro_acc", "worst_acc"):
            v = out.get(k)
            out[k] = f"{v:.5f}" if isinstance(v, float) else v
        for k in ("macro_acc_std", "worst_acc_std", "hard_avg_loss_std"):
            v = out.get(k)
            out[k] = f"{v:.5f}" if isinstance(v, float) else v
        for k in ("hard_avg_loss",):
            v = out.get(k)
            out[k] = f"{v:.4f}" if isinstance(v, float) else v
        for k in ("comm_down_MB", "comm_up_MB"):
            v = out.get(k)
            out[k] = f"{v:.2f}" if isinstance(v, float) else v
        out["seeds"] = ",".join(str(s) for s in out.get("seeds", [])) if isinstance(out.get("seeds"), list) else out.get("seeds")
        return out

    out_stream = open(args.out, "w") if args.out != "-" else None
    try:
        import sys as _sys
        target = out_stream if out_stream else _sys.stdout
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_fmt(row))
        if out_stream:
            print(f"wrote {len(rows)} rows to {args.out}")
    finally:
        if out_stream:
            out_stream.close()


if __name__ == "__main__":
    main()

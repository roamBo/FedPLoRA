#!/usr/bin/env python3
"""Summarize matched-domain eval-only JSON files across seeds."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt_percent(values: list[float]) -> str:
    mean, std = _mean_std(values)
    return f"{100.0 * mean:.2f}±{100.0 * std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="Matched-domain JSON files or directories")
    parser.add_argument(
        "--require-seeds",
        type=int,
        default=3,
        help="Fail unless every dataset/method group has this many distinct seeds (default: 3)",
    )
    args = parser.parse_args()

    paths: list[pathlib.Path] = []
    for raw in args.roots:
        path = pathlib.Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*_matched_domain.json")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)

    groups: dict[tuple[str, str], list[dict]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rounds = payload.get("rounds") or []
        if not rounds:
            continue
        final = rounds[-1]
        if "in_domain_domain_test_worst_token_accuracy" not in final:
            continue
        run_args = payload.get("args") or {}
        benchmark_dir = pathlib.Path(str(payload.get("benchmark_dir") or ""))
        dataset = benchmark_dir.parent.name or benchmark_dir.name
        agg = str(run_args.get("agg_type") or (payload.get("effective_hparams") or {}).get("agg_type") or "")
        seed = int(run_args.get("seed", -1))
        groups.setdefault((dataset, agg), []).append(
            {
                "seed": seed,
                "macro": float(final["in_domain_domain_test_macro_token_accuracy"]),
                "worst": float(final["in_domain_domain_test_worst_token_accuracy"]),
                "macro_ppl": float(final["in_domain_domain_test_macro_perplexity"]),
                "worst_ppl": float(final["in_domain_domain_test_worst_perplexity"]),
                "path": str(path),
            }
        )

    print("dataset\tagg_type\tn\tseeds\tIn-Domain↑\tWorst In-Domain↑\tIn-Domain PPL↓\tWorst In-Domain PPL↓")
    failures: list[str] = []
    for (dataset, agg), rows in sorted(groups.items()):
        by_seed = {row["seed"]: row for row in rows}
        rows = [by_seed[seed] for seed in sorted(by_seed)]
        if args.require_seeds > 0 and len(rows) != args.require_seeds:
            failures.append(
                f"{dataset}/{agg}: expected {args.require_seeds} seeds, found {sorted(by_seed)}"
            )
        macro_ppl = [row["macro_ppl"] for row in rows]
        worst_ppl = [row["worst_ppl"] for row in rows]
        macro_ppl_mean, macro_ppl_std = _mean_std(macro_ppl)
        worst_ppl_mean, worst_ppl_std = _mean_std(worst_ppl)
        print(
            "\t".join(
                [
                    dataset,
                    agg,
                    str(len(rows)),
                    ",".join(str(row["seed"]) for row in rows),
                    _fmt_percent([row["macro"] for row in rows]),
                    _fmt_percent([row["worst"] for row in rows]),
                    f"{macro_ppl_mean:.3f}±{macro_ppl_std:.3f}",
                    f"{worst_ppl_mean:.3f}±{worst_ppl_std:.3f}",
                ]
            )
        )

    if failures:
        raise SystemExit("\n".join(f"[incomplete] {item}" for item in failures))
    if not groups:
        raise SystemExit("no matched-domain eval JSON files found")


if __name__ == "__main__":
    main()

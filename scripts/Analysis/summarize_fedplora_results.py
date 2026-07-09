#!/usr/bin/env python3
"""Summarize FedPLoRA domain-SFT result JSON files.

This script is intentionally dependency-light (stdlib only) so it can run on a
login node before copying data into notebooks.  It emits a Markdown report with:

  - main final-round ranking;
  - communication accounting;
  - B-geometry / cluster quality;
  - per-domain token accuracy;
  - pairwise deltas against selected baselines.

It also prints the benchmark split observed in every JSON.  If multiple split
paths are mixed, the report marks that loudly instead of silently producing a
paper table across incompatible splits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _f(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _mib(x: Any) -> str:
    try:
        val = float(x) / 1024.0 / 1024.0
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.2f}"


def _method_name(path: Path, data: Dict[str, Any]) -> str:
    parent = path.parent.name
    agg = str((data.get("communication") or {}).get("agg_type") or "")
    if parent and parent not in {"result_logs", "sft_metrics"}:
        return parent
    return agg or path.stem


def _final_round(data: Dict[str, Any]) -> Dict[str, Any]:
    rounds = data.get("rounds") or []
    for item in reversed(rounds):
        if not item.get("eval_skipped"):
            return item
    return rounds[-1] if rounds else {}


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": path, "parse_error": repr(exc)})
            continue
        final = _final_round(data)
        comm = data.get("communication") or {}
        expert = final.get("lora_expert_stats") or {}
        row = {
            "path": path,
            "method": _method_name(path, data),
            "agg": comm.get("agg_type") or (data.get("args") or {}).get("agg_type", ""),
            "benchmark_dir": data.get("benchmark_dir") or (data.get("args") or {}).get("benchmark_dir", ""),
            "rounds": len(data.get("rounds") or []),
            "macro": final.get("domain_macro_token_accuracy"),
            "worst": final.get("worst_domain_token_accuracy"),
            "ppl": final.get("domain_macro_perplexity"),
            "local": final.get("client_local_macro_token_accuracy"),
            "off": final.get("off_domain_macro_token_accuracy"),
            "gap": final.get("personalization_gap_token_accuracy"),
            "indom": final.get("in_domain_domain_test_macro_token_accuracy"),
            "eff_comm": (comm.get("effective_down_bytes_per_client", 0) or 0)
            + (comm.get("effective_up_bytes_per_client", 0) or 0),
            "raw_comm": (comm.get("down_bytes_per_client", 0) or 0)
            + (comm.get("up_bytes_per_client", 0) or 0),
            "raw_eq_eff": comm.get("raw_comm_equals_effective_comm"),
            "nmi": expert.get("domain_nmi"),
            "ari": expert.get("domain_ari"),
            "selected_k": expert.get("selected_k"),
            "cluster_mode": expert.get("cluster_mode"),
            "v11_branch": expert.get("v11_branch"),
            "v12_branch": expert.get("v12_branch"),
            "v12_current_mu": expert.get("v12_current_mu"),
            "v12_mu_policy": expert.get("v12_mu_policy"),
            "v12_pre_mix_domain_nmi": expert.get("v12_pre_mix_domain_nmi"),
            "v12_post_mix_domain_nmi": expert.get("v12_post_mix_domain_nmi"),
            "v12_pre_mix_selected_k": expert.get("v12_pre_mix_selected_k"),
            "v12_post_mix_selected_k": expert.get("v12_post_mix_selected_k"),
            "domain_metrics": final.get("domain_metrics") or {},
        }
        for key in (
            "v10_a_mean_rel_update_norm",
            "v10_a_mean_row_cos_to_ref",
            "v10_a_clipped_row_frac",
            "v11_a_mean_rel_update_norm",
            "v11_a_mean_row_cos_to_ref",
            "v11_a_clipped_row_frac",
            "v11_global_b_mix_mu",
        ):
            row[key] = expert.get(key)
        rows.append(row)
    return rows


def _table(headers: List[str], rows: Iterable[Iterable[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _safe_delta(a: Any, b: Any) -> str:
    try:
        return f"{float(a) - float(b):+.4f}"
    except Exception:
        return ""


def build_report(rows: List[Dict[str, Any]], root: Path, compare: List[str]) -> str:
    parsed = [r for r in rows if "parse_error" not in r]
    errors = [r for r in rows if "parse_error" in r]
    splits = sorted({str(r.get("benchmark_dir", "")) for r in parsed if r.get("benchmark_dir")})

    lines = [
        f"# FedPLoRA result summary",
        "",
        f"- root: `{root}`",
        f"- json_files: {len(rows)}",
        f"- parsed: {len(parsed)}",
        f"- parse_errors: {len(errors)}",
        f"- benchmark_splits: {len(splits)}",
    ]
    for split in splits:
        lines.append(f"  - `{split}`")
    if len(splits) > 1:
        lines.append("")
        lines.append("> WARNING: multiple benchmark splits detected; do not merge these rows into one paper table.")
    lines.append("")

    ranked = sorted(
        parsed,
        key=lambda r: float(r["macro"]) if r.get("macro") is not None else -1.0,
        reverse=True,
    )
    lines.extend(
        [
            "## Main final-round ranking",
            "",
            _table(
                ["method", "agg", "Macro", "Worst", "PPL", "Local", "Off", "Gap", "InDom", "EffMiB", "RawMiB", "Raw=Eff", "NMI", "ARI"],
                [
                    [
                        r["method"],
                        r["agg"],
                        _f(r["macro"]),
                        _f(r["worst"]),
                        _f(r["ppl"], 2),
                        _f(r["local"]),
                        _f(r["off"]),
                        _f(r["gap"]),
                        _f(r["indom"]),
                        _mib(r["eff_comm"]),
                        _mib(r["raw_comm"]),
                        "" if r.get("raw_eq_eff") is None else str(bool(r.get("raw_eq_eff"))),
                        _f(r["nmi"]),
                        _f(r["ari"]),
                    ]
                    for r in ranked
                ],
            ),
            "",
            "## Geometry and A-correction",
            "",
            _table(
                [
                    "method",
                    "branch",
                    "K",
                    "cluster_mode",
                    "NMI",
                    "ARI",
                    "v10_A_rel",
                    "v10_A_row_cos",
                    "v10_clip",
                    "v11/v12_mu",
                    "v11_A_rel",
                    "v12_policy",
                    "v12_preNMI",
                    "v12_postNMI",
                    "v12_preK",
                    "v12_postK",
                ],
                [
                    [
                        r["method"],
                        r.get("v12_branch") or r.get("v11_branch") or "",
                        r.get("selected_k", ""),
                        r.get("cluster_mode", ""),
                        _f(r.get("nmi")),
                        _f(r.get("ari")),
                        _f(r.get("v10_a_mean_rel_update_norm")),
                        _f(r.get("v10_a_mean_row_cos_to_ref")),
                        _f(r.get("v10_a_clipped_row_frac")),
                        _f(r.get("v12_current_mu") if r.get("v12_current_mu") is not None else r.get("v11_global_b_mix_mu")),
                        _f(r.get("v11_a_mean_rel_update_norm")),
                        r.get("v12_mu_policy", ""),
                        _f(r.get("v12_pre_mix_domain_nmi")),
                        _f(r.get("v12_post_mix_domain_nmi")),
                        r.get("v12_pre_mix_selected_k", ""),
                        r.get("v12_post_mix_selected_k", ""),
                    ]
                    for r in ranked
                    if r.get("nmi") is not None
                    or r.get("v10_a_mean_rel_update_norm") is not None
                    or r.get("v11_a_mean_rel_update_norm") is not None
                    or r.get("v12_post_mix_domain_nmi") is not None
                ],
            ),
            "",
        ]
    )

    domains = sorted(
        {
            d
            for r in parsed
            for d, metric in (r.get("domain_metrics") or {}).items()
            if isinstance(metric, dict)
        }
    )
    if domains:
        lines.extend(
            [
                "## Per-domain token accuracy",
                "",
                _table(
                    ["method"] + domains,
                    [
                        [r["method"]]
                        + [
                            _f(((r.get("domain_metrics") or {}).get(d) or {}).get("token_accuracy"))
                            for d in domains
                        ]
                        for r in ranked
                    ],
                ),
                "",
            ]
        )

    if compare:
        by_method = {r["method"]: r for r in parsed}
        bases = [by_method[name] for name in compare if name in by_method]
        if bases:
            lines.extend(["## Pairwise deltas", ""])
            delta_rows = []
            for r in ranked:
                for base in bases:
                    if r["method"] == base["method"]:
                        continue
                    delta_rows.append(
                        [
                            r["method"],
                            base["method"],
                            _safe_delta(r.get("macro"), base.get("macro")),
                            _safe_delta(r.get("local"), base.get("local")),
                            _safe_delta(r.get("gap"), base.get("gap")),
                            _safe_delta(r.get("eff_comm"), base.get("eff_comm")),
                        ]
                    )
            lines.extend(
                [
                    _table(["method", "minus", "ΔMacro", "ΔLocal", "ΔGap", "ΔEffBytes"], delta_rows),
                    "",
                ]
            )

    if errors:
        lines.extend(["## Parse errors", ""])
        for err in errors:
            lines.append(f"- `{err['path']}`: {err['parse_error']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Result root containing result_logs/**/*.json")
    ap.add_argument("--output", type=Path, default=None, help="Optional Markdown output path")
    ap.add_argument(
        "--compare",
        default="baseline_normal_dir05,baseline_ecolora_dir05,fedplora_v8",
        help="Comma-separated method directory names for pairwise delta table.",
    )
    args = ap.parse_args()
    rows = _load_rows(args.root)
    compare = [x.strip() for x in str(args.compare or "").split(",") if x.strip()]
    report = build_report(rows, args.root, compare)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

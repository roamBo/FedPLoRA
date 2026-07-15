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
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _args(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = data.get("config") or {}
    raw_args = data.get("args") or {}
    out: Dict[str, Any] = {}
    if isinstance(raw_args, dict):
        out.update(raw_args)
    if isinstance(cfg, dict):
        out.update(cfg)
    return out


def _is_personalized_eval(data: Dict[str, Any]) -> bool:
    return isinstance(data.get("results"), dict) and (
        "strict_held_out" in data or "eval_objective" in data or "protocol_tag" in data
    )


def _is_smoke_row(path: Path, row: Dict[str, Any]) -> bool:
    ptxt = str(path).lower()
    if "smoke" in ptxt or "debug" in ptxt:
        return True
    try:
        if int(row.get("train_max_steps_per_client") or 0) == 1:
            return True
    except Exception:
        pass
    try:
        if int(row.get("max_steps") or 0) == 1:
            return True
    except Exception:
        pass
    return False


def _model_tag(value: Any) -> str:
    text = str(value or "")
    return Path(text).name if text else ""


def _training_protocol_tag(data: Dict[str, Any], row: Dict[str, Any]) -> str:
    args = _args(data)
    eval_final_only = bool(args.get("eval_final_only", False))
    return (
        f"sft:rounds={row.get('rounds', '')}:eval_max_batches={row.get('eval_max_batches', '')}"
        f":eval_final_only={eval_final_only}:train_steps={row.get('train_max_steps_per_client', '')}"
        f":train_cap={row.get('max_train_samples_per_client', '')}"
    )


def _final_round(data: Dict[str, Any]) -> Dict[str, Any]:
    rounds = data.get("rounds") or []
    for item in reversed(rounds):
        if not item.get("eval_skipped"):
            return item
    return rounds[-1] if rounds else {}


def _protocol_signature(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("result_kind", ""),
        row.get("protocol_tag", ""),
        row.get("rounds", ""),
        row.get("eval_max_batches", ""),
        row.get("train_max_steps_per_client", ""),
        row.get("max_steps", ""),
        row.get("max_train_samples_per_client", ""),
        row.get("model_tag", ""),
    )


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": path, "parse_error": repr(exc)})
            continue
        if not isinstance(data, dict):
            continue
        if not (data.get("rounds") or _is_personalized_eval(data)):
            # Ignore auxiliary JSON files such as benchmark fingerprints,
            # manifests, and raw dataset metadata when a broad root is passed.
            continue
        args = _args(data)
        is_personalized = _is_personalized_eval(data)
        final = _final_round(data)
        comm = data.get("communication") or {}
        expert = final.get("lora_expert_stats") or {}
        bench_dir = data.get("benchmark_dir") or args.get("benchmark_dir", "")
        bench_fp = data.get("benchmark_fingerprint") or (data.get("checkpoint_meta") or {}).get("benchmark_fingerprint") or {}
        result_kind = "personalized_eval" if is_personalized else "sft"
        method = _method_name(path, data)
        if is_personalized and (path.parent.name == "result_logs" or not method):
            method = path.stem
        row = {
            "path": path,
            "method": method,
            "result_kind": result_kind,
            "agg": comm.get("agg_type") or args.get("agg_type", ""),
            "benchmark_dir": bench_dir,
            "split_tag": Path(str(bench_dir)).name if bench_dir else "",
            "benchmark_fingerprint": bench_fp,
            "benchmark_sha256": bench_fp.get("combined_sha256") if isinstance(bench_fp, dict) else None,
            "effective_hparams": data.get("effective_hparams") or (data.get("checkpoint_meta") or {}).get("effective_hparams") or {},
            "rounds": len(data.get("rounds") or []),
            "eval_max_batches": args.get("eval_max_batches", ""),
            "train_max_steps_per_client": args.get("train_max_steps_per_client", ""),
            "max_train_samples_per_client": args.get("max_train_samples_per_client", ""),
            "max_steps": args.get("max_steps", ""),
            "model": args.get("model", ""),
            "model_tag": _model_tag(args.get("model", "")),
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
            "down_comm": comm.get("down_bytes_per_client"),
            "up_comm": comm.get("up_bytes_per_client"),
            "eff_down_comm": comm.get("effective_down_bytes_per_client"),
            "eff_up_comm": comm.get("effective_up_bytes_per_client"),
            "downlink_policy": comm.get("downlink_policy", ""),
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
        if is_personalized:
            results = data.get("results") or {}
            primary_scheme = "coldstart_geom" if "coldstart_geom" in results else (
                "coldstart" if "coldstart" in results else "global"
            )
            primary = results.get(primary_scheme) or {}
            strict = data.get("strict_held_out") or {}
            row.update(
                {
                    "primary_scheme": primary_scheme,
                    "protocol_tag": data.get("protocol_tag") or "personalized_eval",
                    "macro": primary.get("macro_acc"),
                    "worst": primary.get("worst_acc"),
                    "local": primary.get("macro_acc"),
                    "gap": (
                        float(primary.get("macro_acc")) - float((results.get("global") or {}).get("macro_acc"))
                        if primary.get("macro_acc") is not None and (results.get("global") or {}).get("macro_acc") is not None
                        else None
                    ),
                    "eval_objective": data.get("eval_objective", ""),
                    "strict_held_out_enabled": bool(strict.get("enabled", False)),
                    "held_out_policy": strict.get("selection_policy", ""),
                    "held_out_offset": strict.get("selection_offset", ""),
                    "held_out_clients": strict.get("held_out_clients", []),
                    "geom_route_oracle_match_rate": strict.get("geom_route_oracle_match_rate"),
                    "geom_route_mean_margin": ((strict.get("geom_route_summary") or {}).get("mean_margin")),
                    "geom_route_min_margin": ((strict.get("geom_route_summary") or {}).get("min_margin")),
                    "domain_metrics": {
                        d: {"token_accuracy": v}
                        for d, v in (primary.get("per_domain_acc") or {}).items()
                    },
                }
            )
        else:
            row["protocol_tag"] = _training_protocol_tag(data, row)
            row["primary_scheme"] = ""
            row["eval_objective"] = "domain_macro_sft"
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
        row["is_smoke"] = _is_smoke_row(path, row)
        rows.append(row)
    return rows


def _fingerprint_conflicts(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_split: Dict[str, set] = {}
    for r in rows:
        tag = str(r.get("split_tag") or "")
        sha = str(r.get("benchmark_sha256") or "")
        if not tag or not sha:
            continue
        by_split.setdefault(tag, set()).add(sha)
    return {
        tag: sorted(vals)
        for tag, vals in sorted(by_split.items())
        if len(vals) > 1
    }


def _protocol_conflicts(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_split: Dict[str, set] = {}
    for r in rows:
        tag = str(r.get("split_tag") or "")
        if not tag:
            continue
        by_split.setdefault(tag, set()).add("|".join(str(x) for x in _protocol_signature(r)))
    return {
        tag: sorted(vals)
        for tag, vals in sorted(by_split.items())
        if len(vals) > 1
    }


def _missing_fingerprints(rows: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in rows:
        if not r.get("benchmark_sha256"):
            out.append(str(r.get("path", "")))
    return out


def _filter_rows(rows: List[Dict[str, Any]], *, exclude_smoke: bool, kind: str) -> List[Dict[str, Any]]:
    parsed = [r for r in rows if "parse_error" not in r]
    if exclude_smoke:
        parsed = [r for r in parsed if not r.get("is_smoke")]
    if kind:
        parsed = [r for r in parsed if str(r.get("result_kind", "")) == kind]
    return parsed


def _group_protocol_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            str(r.get("result_kind", "")),
            str(r.get("split_tag", "")),
            str(r.get("protocol_tag", "")),
            str(r.get("model_tag", "")),
        )
        groups.setdefault(key, []).append(r)
    return groups


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
    fp_conflicts = _fingerprint_conflicts(parsed)
    protocol_conflicts = _protocol_conflicts(parsed)
    missing_fp = _missing_fingerprints(parsed)
    protocols = sorted({"|".join(str(x) for x in _protocol_signature(r)) for r in parsed})
    kinds = sorted({str(r.get("result_kind", "")) for r in parsed})
    smoke_count = sum(1 for r in parsed if r.get("is_smoke"))

    lines = [
        f"# FedPLoRA result summary",
        "",
        f"- root: `{root}`",
        f"- json_files: {len(rows)}",
        f"- parsed: {len(parsed)}",
        f"- parse_errors: {len(errors)}",
        f"- benchmark_splits: {len(splits)}",
        f"- benchmark_fingerprint_conflicts_by_split_tag: {len(fp_conflicts)}",
        f"- missing_benchmark_fingerprints: {len(missing_fp)}",
        f"- result_kinds: {', '.join(kinds) if kinds else ''}",
        f"- smoke_or_debug_rows: {smoke_count}",
        f"- protocol_groups: {len(protocols)}",
    ]
    for split in splits:
        lines.append(f"  - `{split}`")
    if len(splits) > 1:
        lines.append("")
        lines.append("> WARNING: multiple benchmark splits detected; do not merge these rows into one paper table.")
    if fp_conflicts:
        lines.append("")
        lines.append("> DANGER: same split tag has multiple benchmark fingerprints; paired tables are unsafe until fixed.")
        for tag, vals in fp_conflicts.items():
            lines.append(f"  - {tag}: {', '.join(v[:16] for v in vals)}")
    if missing_fp:
        lines.append("")
        lines.append("> WARNING: some rows have no benchmark fingerprint; strict paired claims should exclude or regenerate them.")
        for path in missing_fp[:20]:
            lines.append(f"  - `{path}`")
        if len(missing_fp) > 20:
            lines.append(f"  - ... {len(missing_fp) - 20} more")
    if protocol_conflicts:
        lines.append("")
        lines.append("> WARNING: multiple protocol signatures under the same split tag; use --strict_protocol for paper-table generation.")
        for tag, vals in protocol_conflicts.items():
            lines.append(f"  - {tag}: {len(vals)} protocol groups")
    lines.append("")

    if parsed:
        lines.extend(
            [
                "## Protocol groups",
                "",
                _table(
                    [
                        "kind",
                        "split",
                        "protocol",
                        "model",
                        "rows",
                        "smoke",
                        "fingerprinted",
                    ],
                    [
                        [
                            kind,
                            split,
                            protocol,
                            model,
                            str(len(group_rows)),
                            str(sum(1 for r in group_rows if r.get("is_smoke"))),
                            str(sum(1 for r in group_rows if r.get("benchmark_sha256"))),
                        ]
                        for (kind, split, protocol, model), group_rows in sorted(
                            _group_protocol_rows(parsed).items(),
                            key=lambda kv: kv[0],
                        )
                    ],
                ),
                "",
            ]
        )

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
                    if r.get("result_kind") == "sft"
                ],
            ),
            "",
            "## Personalized / strict-heldout ranking",
            "",
            _table(
                [
                    "method",
                    "scheme",
                    "split",
                    "Macro",
                    "Worst",
                    "ΔvsGlobal",
                    "objective",
                    "heldout",
                    "policy",
                    "offset",
                    "route_match",
                    "route_margin",
                ],
                [
                    [
                        r["method"],
                        r.get("primary_scheme", ""),
                        r.get("split_tag", ""),
                        _f(r["macro"]),
                        _f(r["worst"]),
                        _f(r["gap"]),
                        r.get("eval_objective", ""),
                        str(bool(r.get("strict_held_out_enabled"))),
                        r.get("held_out_policy", ""),
                        r.get("held_out_offset", ""),
                        _f(r.get("geom_route_oracle_match_rate")),
                        _f(r.get("geom_route_mean_margin")),
                    ]
                    for r in ranked
                    if r.get("result_kind") == "personalized_eval"
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

    lines.extend(
        [
            "## Communication accounting",
            "",
            _table(
                [
                    "method",
                    "agg",
                    "kind",
                    "split",
                    "protocol",
                    "RawMiB",
                    "EffMiB",
                    "DownMiB",
                    "UpMiB",
                    "EffDownMiB",
                    "EffUpMiB",
                    "Raw=Eff",
                    "policy",
                ],
                [
                    [
                        r["method"],
                        r["agg"],
                        r.get("result_kind", ""),
                        r.get("split_tag", ""),
                        r.get("protocol_tag", ""),
                        _mib(r.get("raw_comm")),
                        _mib(r.get("eff_comm")),
                        _mib(r.get("down_comm")),
                        _mib(r.get("up_comm")),
                        _mib(r.get("eff_down_comm")),
                        _mib(r.get("eff_up_comm")),
                        "" if r.get("raw_eq_eff") is None else str(bool(r.get("raw_eq_eff"))),
                        r.get("downlink_policy", ""),
                    ]
                    for r in ranked
                    if r.get("raw_comm") or r.get("eff_comm")
                ],
            ),
            "",
        ]
    )

    if parsed:
        lines.extend(
            [
                "## Benchmark fingerprints",
                "",
                _table(
                    ["method", "kind", "split", "protocol", "sha256", "train_rows", "clients", "path"],
                    [
                        [
                            r["method"],
                            r.get("result_kind", ""),
                            r.get("split_tag", ""),
                            r.get("protocol_tag", ""),
                            str(r.get("benchmark_sha256") or "")[:16],
                            (((r.get("benchmark_fingerprint") or {}).get("split_counts") or {}).get("train", "")),
                            (((r.get("benchmark_fingerprint") or {}).get("clients") or {}).get("num_clients", "")),
                            r.get("benchmark_dir", ""),
                        ]
                        for r in ranked
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
                    ["method", "kind", "split"] + domains,
                    [
                        [r["method"], r.get("result_kind", ""), r.get("split_tag", "")]
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

            paired_rows = []
            by_method_split = {}
            for r in parsed:
                by_method_split[(r["method"], r.get("split_tag", ""))] = r
            for r in sorted(parsed, key=lambda x: (x.get("method", ""), x.get("split_tag", ""))):
                for base in bases:
                    if r["method"] == base["method"]:
                        continue
                    b = by_method_split.get((base["method"], r.get("split_tag", "")))
                    if not b:
                        continue
                    if (
                        r.get("benchmark_sha256")
                        and b.get("benchmark_sha256")
                        and r.get("benchmark_sha256") != b.get("benchmark_sha256")
                    ):
                        safe = "NO:fingerprint"
                    else:
                        safe = "YES"
                    paired_rows.append(
                        [
                            r.get("split_tag", ""),
                            r["method"],
                            base["method"],
                            safe,
                            _safe_delta(r.get("macro"), b.get("macro")),
                            _safe_delta(r.get("local"), b.get("local")),
                            _safe_delta(r.get("gap"), b.get("gap")),
                            _safe_delta(r.get("raw_comm"), b.get("raw_comm")),
                        ]
                    )
            if paired_rows:
                lines.extend(
                    [
                        "## Paired split deltas",
                        "",
                        _table(
                            ["split", "method", "minus", "safe", "ΔMacro", "ΔLocal", "ΔGap", "ΔRawBytes"],
                            paired_rows,
                        ),
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
        "--kind",
        choices=["", "sft", "personalized_eval"],
        default="",
        help="Optional result kind filter. Empty keeps both SFT and personalized-eval JSON files.",
    )
    ap.add_argument(
        "--exclude_smoke",
        action="store_true",
        help="Exclude paths or configs that look like smoke/debug runs.",
    )
    ap.add_argument(
        "--compare",
        default="baseline_normal_dir05,baseline_ecolora_dir05,fedplora_v8",
        help="Comma-separated method directory names for pairwise delta table.",
    )
    ap.add_argument(
        "--strict_fingerprint",
        action="store_true",
        help="Exit non-zero if the same split tag appears with multiple benchmark fingerprints.",
    )
    ap.add_argument(
        "--require_fingerprint",
        action="store_true",
        help="Exit non-zero if any parsed row lacks benchmark_fingerprint.combined_sha256.",
    )
    ap.add_argument(
        "--strict_protocol",
        action="store_true",
        help="Exit non-zero if one split tag contains mixed protocol signatures.",
    )
    args = ap.parse_args()
    raw_rows = _load_rows(args.root)
    parsed_rows = _filter_rows(raw_rows, exclude_smoke=bool(args.exclude_smoke), kind=str(args.kind or ""))
    parse_errors = [r for r in raw_rows if "parse_error" in r]
    rows = parsed_rows + parse_errors
    if args.strict_fingerprint:
        conflicts = _fingerprint_conflicts(parsed_rows)
        if conflicts:
            print("[summary][error] benchmark fingerprint conflicts detected:", conflicts)
            raise SystemExit(2)
    if args.require_fingerprint:
        missing = _missing_fingerprints(parsed_rows)
        if missing:
            print("[summary][error] rows missing benchmark fingerprints:")
            for path in missing:
                print(f"  - {path}")
            raise SystemExit(2)
    if args.strict_protocol:
        conflicts = _protocol_conflicts(parsed_rows)
        if conflicts:
            print("[summary][error] protocol conflicts detected:", conflicts)
            raise SystemExit(2)
    compare = [x.strip() for x in str(args.compare or "").split(",") if x.strip()]
    report = build_report(rows, args.root, compare)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

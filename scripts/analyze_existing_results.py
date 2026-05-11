#!/usr/bin/env python3
"""Offline result analysis for FedPLoRA domain-SFT logs.

The script only parses existing logs/metrics and writes derived tables. It does
not load models, datasets, or rerun evaluation.
"""

import argparse
import ast
import csv
import json
import math
import glob
import re
from datetime import datetime
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
KV_RE = re.compile(rf"([A-Za-z][A-Za-z0-9_]*)=({FLOAT_RE})")
DOMAIN_LOSS_RE = re.compile(rf"([A-Za-z0-9_-]+)_loss=({FLOAT_RE})")
DOMAIN_ACC_RE = re.compile(rf"([A-Za-z0-9_-]+)_tok_acc=({FLOAT_RE})")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

SCALAR_LOSS_PREFIXES = {
    "domain_macro",
    "best_domain_macro",
    "worst_domain",
    "best_worst_domain",
    "client_local_macro",
    "in_domain_domain_test_macro",
    "off_domain_macro",
}

SCALAR_ACC_PREFIXES = {
    "domain_macro",
    "best_domain_macro",
    "worst_domain",
    "best_worst_domain",
}

HARD_DOMAINS = ("finance", "legal", "medical")
CAPABILITY_DOMAINS = ("code", "math")


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value, ndigits=4):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(value):
        return "-"
    return f"{value:.{ndigits}f}"


def fmt_gb(value):
    return fmt(value, 3)


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def parse_args_line(line):
    marker = "[log] args:"
    if marker not in line:
        return {}
    raw = line.split(marker, 1)[1].strip()
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def parse_setup_line(line):
    if "[setup]" not in line or "comm_down_bytes_per_client" not in line:
        return {}
    out = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", line):
        if key in {"agg_type"}:
            out[key] = value
        elif key in {"num_clients", "comm_down_bytes_per_client", "comm_up_bytes_per_client"}:
            try:
                out[key] = int(value)
            except ValueError:
                pass
    return out


def parse_eval_line(line):
    if "[eval] round=" not in line or "domain_macro_loss=" not in line:
        return None
    values = {k: to_float(v) for k, v in KV_RE.findall(line)}
    if "round" not in values:
        match = re.search(r"round=(\d+)", line)
        if match:
            values["round"] = float(match.group(1))
    row = {
        "round": int(values.get("round", 0)),
        "domain_macro_loss": values.get("domain_macro_loss", math.nan),
        "best_domain_macro_loss": values.get("best_domain_macro_loss", math.nan),
        "worst_domain_loss": values.get("worst_domain_loss", math.nan),
        "best_worst_domain_loss": values.get("best_worst_domain_loss", math.nan),
        "domain_macro_token_accuracy": values.get("domain_macro_tok_acc", math.nan),
        "best_domain_macro_token_accuracy": values.get("best_domain_macro_tok_acc", math.nan),
        "worst_domain_token_accuracy": values.get("worst_domain_tok_acc", math.nan),
        "best_worst_domain_token_accuracy": values.get("best_worst_domain_tok_acc", math.nan),
        "domain_losses": {},
        "domain_token_accuracy": {},
    }
    for name, value in DOMAIN_LOSS_RE.findall(line):
        if name not in SCALAR_LOSS_PREFIXES:
            row["domain_losses"][name] = to_float(value)
    for name, value in DOMAIN_ACC_RE.findall(line):
        if name not in SCALAR_ACC_PREFIXES:
            row["domain_token_accuracy"][name] = to_float(value)
    return row


def parse_personalization_line(line):
    if "[eval] personalization" not in line:
        return None
    values = {k: to_float(v) for k, v in KV_RE.findall(line)}
    match = re.search(r"round=(\d+)", line)
    round_idx = int(match.group(1)) if match else int(values.get("round", 0))
    return {
        "round": round_idx,
        "client_local_macro_loss": values.get("client_local_macro_loss", math.nan),
        "in_domain_domain_test_macro_loss": values.get(
            "in_domain_domain_test_macro_loss", math.nan
        ),
        "off_domain_macro_loss": values.get("off_domain_macro_loss", math.nan),
    }


def parse_per_domain_acc_line(line):
    if "[eval] per-domain token_accuracy:" not in line:
        return None
    return {name: to_float(value) for name, value in DOMAIN_ACC_RE.findall(line)}


def parse_conflict_line(line):
    if "[fedplora-oneshot] conflict" not in line:
        return None
    out = {}
    for key, value in KV_RE.findall(line):
        out[f"conflict_{key}"] = to_float(value)
    return out


def parse_log(path):
    run_args = {}
    setup = {}
    rounds = []
    latest_conflict = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.replace("\r", "\n"))
            for part in line.splitlines():
                if not part:
                    continue
                if "[log] args:" in part:
                    run_args.update(parse_args_line(part))
                if "comm_down_bytes_per_client" in part:
                    setup.update(parse_setup_line(part))
                conflict = parse_conflict_line(part)
                if conflict:
                    latest_conflict = conflict
                eval_row = parse_eval_line(part)
                if eval_row:
                    if latest_conflict:
                        eval_row.update(latest_conflict)
                        latest_conflict = {}
                    rounds.append(eval_row)
                    continue
                personalization = parse_personalization_line(part)
                if personalization and rounds:
                    target_round = personalization.pop("round")
                    for row in reversed(rounds):
                        if row["round"] == target_round:
                            row.update(personalization)
                            break
                    continue
                accs = parse_per_domain_acc_line(part)
                if accs and rounds:
                    rounds[-1]["domain_token_accuracy"].update(accs)

    method = setup.get("agg_type") or run_args.get("agg_type") or infer_method_from_name(path.name)
    num_clients = setup.get("num_clients") or int(run_args.get("num_clients") or 0)
    if num_clients <= 0:
        num_clients = infer_clients_from_name(path.name)

    for row in rounds:
        row.update(
            {
                "source_file": str(path),
                "source_type": "log",
                "method": method,
                "model": run_args.get("model", ""),
                "benchmark_dir": run_args.get("benchmark_dir", ""),
                "seed": run_args.get("seed", ""),
                "requested_rounds": run_args.get("rounds", ""),
                "local_epochs": run_args.get("local_epochs", ""),
                "lr": run_args.get("lr", ""),
                "lora_r": run_args.get("lora_r", ""),
                "num_clients": num_clients,
                "down_bytes_per_client": setup.get("comm_down_bytes_per_client", math.nan),
                "up_bytes_per_client": setup.get("comm_up_bytes_per_client", math.nan),
            }
        )
    return rounds


def parse_metrics_json(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    args = payload.get("args", {})
    comm = payload.get("communication", {})
    method = comm.get("agg_type") or args.get("agg_type") or infer_method_from_name(path.name)
    num_clients = int(args.get("num_clients") or 0)
    rows = []
    for item in payload.get("rounds", []):
        domain_metrics = item.get("domain_metrics", {}) or {}
        domain_losses = {}
        domain_acc = {}
        for domain, metrics in domain_metrics.items():
            if isinstance(metrics, dict):
                domain_losses[domain] = to_float(metrics.get("loss", math.nan))
                domain_acc[domain] = to_float(metrics.get("token_accuracy", math.nan))
            else:
                domain_losses[domain] = to_float(metrics)
        row = {
            "source_file": str(path),
            "source_type": "json",
            "method": method,
            "model": args.get("model", ""),
            "benchmark_dir": payload.get("benchmark_dir", args.get("benchmark_dir", "")),
            "seed": args.get("seed", ""),
            "requested_rounds": args.get("rounds", ""),
            "local_epochs": args.get("local_epochs", ""),
            "lr": args.get("lr", ""),
            "lora_r": args.get("lora_r", ""),
            "num_clients": num_clients,
            "down_bytes_per_client": comm.get("down_bytes_per_client", math.nan),
            "up_bytes_per_client": comm.get("up_bytes_per_client", math.nan),
            "round": int(item.get("round", 0)),
            "domain_macro_loss": to_float(item.get("domain_macro_loss", math.nan)),
            "best_domain_macro_loss": to_float(item.get("best_domain_macro_loss", math.nan)),
            "worst_domain_loss": to_float(item.get("worst_domain_loss", math.nan)),
            "best_worst_domain_loss": to_float(item.get("best_worst_domain_loss", math.nan)),
            "domain_macro_token_accuracy": to_float(
                item.get("domain_macro_token_accuracy", math.nan)
            ),
            "best_domain_macro_token_accuracy": to_float(
                item.get("best_domain_macro_token_accuracy", math.nan)
            ),
            "worst_domain_token_accuracy": to_float(
                item.get("worst_domain_token_accuracy", math.nan)
            ),
            "best_worst_domain_token_accuracy": to_float(
                item.get("best_worst_domain_token_accuracy", math.nan)
            ),
            "domain_losses": domain_losses,
            "domain_token_accuracy": domain_acc,
            "client_local_macro_loss": to_float(
                item.get("client_local_macro_loss", math.nan)
            ),
            "in_domain_domain_test_macro_loss": to_float(
                item.get("in_domain_domain_test_macro_loss", math.nan)
            ),
            "off_domain_macro_loss": to_float(item.get("off_domain_macro_loss", math.nan)),
        }
        conflict = item.get("fedplora_oneshot_conflict", {}) or {}
        for key, value in conflict.items():
            row[f"conflict_{key}"] = to_float(value)
        rows.append(row)
    return rows


def infer_method_from_name(name):
    match = re.search(r"sft_run_([^_]+(?:-[^_]+)?)_", name)
    if match:
        return match.group(1)
    return Path(name).stem.split("_")[0]


def infer_clients_from_name(name):
    match = re.search(r"_c(\d+)_", name)
    return int(match.group(1)) if match else 0


def mean(values):
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            clean.append(value)
    values = clean
    return sum(values) / len(values) if values else math.nan


def std(values):
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            clean.append(value)
    values = clean
    if not values:
        return math.nan
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def add_derived_round_metrics(rows):
    for row in rows:
        domain_losses = row.get("domain_losses", {}) or {}
        if domain_losses:
            losses = list(domain_losses.values())
            if math.isnan(row.get("domain_macro_loss", math.nan)):
                row["domain_macro_loss"] = mean(losses)
            if math.isnan(row.get("worst_domain_loss", math.nan)):
                row["worst_domain_loss"] = max(losses)
            row["domain_loss_std"] = std(losses)
            row["domain_loss_gap"] = max(losses) - min(losses)
            row["domain_loss_cv"] = row["domain_loss_std"] / max(
                abs(row["domain_macro_loss"]), 1e-12
            )
            row["hard_domain_macro_loss"] = mean(
                domain_losses.get(d, math.nan) for d in HARD_DOMAINS
            )
            row["capability_domain_macro_loss"] = mean(
                domain_losses.get(d, math.nan) for d in CAPABILITY_DOMAINS
            )
            row["hardest_domain"] = max(domain_losses, key=domain_losses.get)
            row["easiest_domain"] = min(domain_losses, key=domain_losses.get)
        else:
            row["domain_loss_std"] = math.nan
            row["domain_loss_gap"] = math.nan
            row["domain_loss_cv"] = math.nan
            row["hard_domain_macro_loss"] = math.nan
            row["capability_domain_macro_loss"] = math.nan
            row["hardest_domain"] = ""
            row["easiest_domain"] = ""

        down = to_float(row.get("down_bytes_per_client", math.nan))
        up = to_float(row.get("up_bytes_per_client", math.nan))
        clients = int(row.get("num_clients") or 0)
        round_idx = int(row.get("round") or 0)
        if not math.isnan(down) and not math.isnan(up) and clients > 0:
            row["cumulative_down_gb"] = down * clients * round_idx / 1e9
            row["cumulative_up_gb"] = up * clients * round_idx / 1e9
            row["cumulative_total_comm_gb"] = (down + up) * clients * round_idx / 1e9
            row["cumulative_total_comm_gib"] = (down + up) * clients * round_idx / (1024**3)
        else:
            row["cumulative_down_gb"] = math.nan
            row["cumulative_up_gb"] = math.nan
            row["cumulative_total_comm_gb"] = math.nan
            row["cumulative_total_comm_gib"] = math.nan
    return rows


def summarize_runs(rows):
    by_run = {}
    for row in rows:
        key = (row["source_file"], row["method"])
        by_run.setdefault(key, []).append(row)

    summary = []
    for (_source_file, _method), run_rows in by_run.items():
        run_rows = sorted(run_rows, key=lambda x: int(x.get("round") or 0))
        final_row = run_rows[-1]
        best_macro_row = min(run_rows, key=lambda x: x.get("domain_macro_loss", math.inf))
        best_worst_row = min(run_rows, key=lambda x: x.get("worst_domain_loss", math.inf))
        best_hard_row = min(
            run_rows, key=lambda x: x.get("hard_domain_macro_loss", math.inf)
        )
        out = {
            "method": final_row["method"],
            "source_file": final_row["source_file"],
            "source_type": final_row["source_type"],
            "model": final_row.get("model", ""),
            "benchmark_dir": final_row.get("benchmark_dir", ""),
            "seed": final_row.get("seed", ""),
            "num_clients": final_row.get("num_clients", ""),
            "requested_rounds": final_row.get("requested_rounds", ""),
            "actual_rounds": final_row.get("round", ""),
            "best_macro_round": best_macro_row.get("round", ""),
            "best_worst_round": best_worst_row.get("round", ""),
            "best_hard_round": best_hard_row.get("round", ""),
            "final_domain_macro_loss": final_row.get("domain_macro_loss", math.nan),
            "best_domain_macro_loss": best_macro_row.get("domain_macro_loss", math.nan),
            "final_worst_domain_loss": final_row.get("worst_domain_loss", math.nan),
            "best_worst_domain_loss": best_worst_row.get("worst_domain_loss", math.nan),
            "final_domain_loss_std": final_row.get("domain_loss_std", math.nan),
            "final_domain_loss_gap": final_row.get("domain_loss_gap", math.nan),
            "final_domain_loss_cv": final_row.get("domain_loss_cv", math.nan),
            "final_hard_domain_macro_loss": final_row.get(
                "hard_domain_macro_loss", math.nan
            ),
            "best_hard_domain_macro_loss": best_hard_row.get(
                "hard_domain_macro_loss", math.nan
            ),
            "final_capability_domain_macro_loss": final_row.get(
                "capability_domain_macro_loss", math.nan
            ),
            "best_capability_domain_macro_loss": best_macro_row.get(
                "capability_domain_macro_loss", math.nan
            ),
            "hardest_domain": final_row.get("hardest_domain", ""),
            "easiest_domain": final_row.get("easiest_domain", ""),
            "final_domain_macro_token_accuracy": final_row.get(
                "domain_macro_token_accuracy", math.nan
            ),
            "final_worst_domain_token_accuracy": final_row.get(
                "worst_domain_token_accuracy", math.nan
            ),
            "client_local_macro_loss": final_row.get("client_local_macro_loss", math.nan),
            "in_domain_domain_test_macro_loss": final_row.get(
                "in_domain_domain_test_macro_loss", math.nan
            ),
            "off_domain_macro_loss": final_row.get("off_domain_macro_loss", math.nan),
            "down_bytes_per_client": final_row.get("down_bytes_per_client", math.nan),
            "up_bytes_per_client": final_row.get("up_bytes_per_client", math.nan),
            "final_total_comm_gb": final_row.get("cumulative_total_comm_gb", math.nan),
            "final_total_comm_gib": final_row.get("cumulative_total_comm_gib", math.nan),
            "best_macro_total_comm_gb": best_macro_row.get(
                "cumulative_total_comm_gb", math.nan
            ),
            "best_hard_total_comm_gb": best_hard_row.get(
                "cumulative_total_comm_gb", math.nan
            ),
            "macro_degradation_final_minus_best": final_row.get(
                "domain_macro_loss", math.nan
            )
            - best_macro_row.get("domain_macro_loss", math.nan),
            "worst_degradation_final_minus_best": final_row.get(
                "worst_domain_loss", math.nan
            )
            - best_worst_row.get("worst_domain_loss", math.nan),
            "hard_degradation_final_minus_best": final_row.get(
                "hard_domain_macro_loss", math.nan
            )
            - best_hard_row.get("hard_domain_macro_loss", math.nan),
            "conflict_mean": final_row.get("conflict_mean", math.nan),
            "conflict_max": final_row.get("conflict_max", math.nan),
            "conflict_high_row_frac": final_row.get("conflict_high_row_frac", math.nan),
            "conflict_init_gate": final_row.get("conflict_init_gate", math.nan),
        }
        for domain, value in sorted((final_row.get("domain_losses", {}) or {}).items()):
            out[f"loss_{domain}"] = value
        for domain, value in sorted((final_row.get("domain_token_accuracy", {}) or {}).items()):
            out[f"tok_acc_{domain}"] = value
        summary.append(out)

    add_ranks(summary)
    return summary


def add_ranks(summary):
    def rank_key(row, key):
        value = row.get(key, math.nan)
        return value if not math.isnan(to_float(value)) else math.inf

    for metric, rank_name in [
        ("best_domain_macro_loss", "rank_macro"),
        ("best_worst_domain_loss", "rank_worst"),
        ("best_hard_domain_macro_loss", "rank_hard_domains"),
    ]:
        ordered = sorted(summary, key=lambda row: rank_key(row, metric))
        for idx, row in enumerate(ordered, start=1):
            row[rank_name] = idx
    for row in summary:
        ranks = [row.get("rank_macro"), row.get("rank_worst"), row.get("rank_hard_domains")]
        row["rank_average"] = mean(ranks)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames and key not in {"domain_losses", "domain_token_accuracy"}:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_csv_value(row.get(k, "")) for k in fieldnames})


def serialize_csv_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    return value


def serialize_json_value(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def json_clean(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {key: json_clean(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_clean(value) for value in obj]
    return obj


def write_markdown(path, summary, rows, main_method):
    summary_sorted = sorted(
        summary,
        key=lambda x: x.get("rank_average", math.inf)
        if not math.isnan(to_float(x.get("rank_average", math.nan)))
        else math.inf,
    )
    main = next((row for row in summary_sorted if row.get("method") == main_method), None)

    lines = []
    lines.append("# FedPLoRA 离线结果分析")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 解析记录：{len(rows)} 个 round，{len(summary)} 个 run")
    lines.append("- 说明：该文件仅基于已有日志/JSON 派生指标，不重新训练、不重新评测。")
    lines.append("")
    lines.append("## 指标是否够用")
    lines.append("")
    lines.append("- 现有 `domain_macro_loss`、`worst_domain_loss`、per-domain loss 与通信量，足够支撑阶段性主表、跨域冲突分析和通信-性能 Pareto。")
    lines.append("- 若用于顶会/顶刊最终版，仍建议补充多 seed、生成式任务指标、个性化本域/跨域指标和 token accuracy；若旧日志没有这些字段，脚本会留空，不会伪造。")
    lines.append("- 当前已有日志中的 `fedplora-oneshot` 属于旧版 FedP+YOCO 聚合结果；新增 conflict-gated FedPLoRA-Oneshot 需要后续至少做小规模 smoke 或主实验复核。")
    lines.append("")
    lines.append("## 主结果汇总")
    lines.append("")
    lines.append("| 方法 | 轮数 | Best Macro↓ | Best Worst↓ | Best Hard↓ | Final Macro↓ | 退化↓ | 通信GB↓ | Rank | Hardest |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_sorted:
        lines.append(
            "| {method} | {rounds} | {macro} | {worst} | {hard} | {final_macro} | {degrade} | {comm} | {rank} | {hardest} |".format(
                method=row.get("method", ""),
                rounds=row.get("actual_rounds", ""),
                macro=fmt(row.get("best_domain_macro_loss")),
                worst=fmt(row.get("best_worst_domain_loss")),
                hard=fmt(row.get("best_hard_domain_macro_loss")),
                final_macro=fmt(row.get("final_domain_macro_loss")),
                degrade=fmt(row.get("macro_degradation_final_minus_best")),
                comm=fmt_gb(row.get("final_total_comm_gb")),
                rank=fmt(row.get("rank_average"), 2),
                hardest=row.get("hardest_domain", ""),
            )
        )
    lines.append("")

    if main:
        lines.append(f"## 与 `{main_method}` 对比")
        lines.append("")
        lines.append("| Baseline | Macro 相对降幅↑ | Worst 相对降幅↑ | Hard Avg 相对降幅↑ | 通信倍率 |")
        lines.append("|---|---:|---:|---:|---:|")
        for base in summary_sorted:
            if base is main:
                continue
            lines.append(
                "| {method} | {macro} | {worst} | {hard} | {comm} |".format(
                    method=base.get("method", ""),
                    macro=relative_loss_reduction(main, base, "best_domain_macro_loss"),
                    worst=relative_loss_reduction(main, base, "best_worst_domain_loss"),
                    hard=relative_loss_reduction(
                        main, base, "best_hard_domain_macro_loss"
                    ),
                    comm=comm_ratio(main, base),
                )
            )
        lines.append("")

    lines.append("## Per-Domain Loss")
    lines.append("")
    domains = sorted(
        {
            key.replace("loss_", "", 1)
            for row in summary
            for key in row.keys()
            if key.startswith("loss_")
        }
    )
    if domains:
        lines.append("| 方法 | " + " | ".join(domains) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(domains)) + "|")
        for row in summary_sorted:
            cells = [row.get("method", "")]
            cells.extend(fmt(row.get(f"loss_{domain}")) for domain in domains)
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("未在日志中解析到 per-domain loss。")
    lines.append("")

    lines.append("## 可直接写入论文的离线派生指标")
    lines.append("")
    lines.append("- 跨域平均：`best_domain_macro_loss` / `final_domain_macro_loss`。")
    lines.append("- 鲁棒性：`best_worst_domain_loss`、`final_domain_loss_gap`、`final_domain_loss_std`、`final_domain_loss_cv`。")
    lines.append("- 高风险领域：`finance/legal/medical` 的 `final_hard_domain_macro_loss`。")
    lines.append("- 能力领域：`code/math` 的 `final_capability_domain_macro_loss`。")
    lines.append("- 通信效率：`final_total_comm_gb`、`best_macro_total_comm_gb`，由日志中的上下行字节、客户端数和实际轮数计算。")
    lines.append("- 退化分析：多轮方法可看 `macro_degradation_final_minus_best`，避免只报最后一轮掩盖过拟合。")
    lines.append("")
    lines.append("## 文件")
    lines.append("")
    lines.append("- `analysis_rounds.csv`：每轮指标。")
    lines.append("- `analysis_summary.csv`：每个 run 的最终/最优/通信/鲁棒性指标。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_name("analysis_summary.json").write_text(
        json.dumps(
            json_clean(summary_sorted),
            indent=2,
            ensure_ascii=False,
            default=serialize_json_value,
        ),
        encoding="utf-8",
    )


def relative_loss_reduction(main, base, key):
    main_v = to_float(main.get(key, math.nan))
    base_v = to_float(base.get(key, math.nan))
    if math.isnan(main_v) or math.isnan(base_v) or base_v == 0:
        return "-"
    return f"{(base_v - main_v) / base_v * 100.0:.2f}%"


def comm_ratio(main, base):
    main_v = to_float(main.get("final_total_comm_gb", math.nan))
    base_v = to_float(base.get("final_total_comm_gb", math.nan))
    if math.isnan(main_v) or math.isnan(base_v) or base_v == 0:
        return "-"
    return f"{main_v / base_v:.2f}x"


def main():
    parser = argparse.ArgumentParser(
        description="Parse existing FedPLoRA logs and derive paper-ready metrics."
    )
    parser.add_argument("--result_dir", default="result", help="Directory containing *.log")
    parser.add_argument(
        "--log_glob", default="*.log", help="Log glob under --result_dir"
    )
    parser.add_argument(
        "--include_json",
        action="store_true",
        help="Also parse metrics JSON files matched by --json_glob.",
    )
    parser.add_argument(
        "--json_glob",
        default="artifacts*c/sft_metrics/*.json",
        help="Metrics JSON glob, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output_dir",
        default="result",
        help="Where to write analysis_summary.csv/md and analysis_rounds.csv.",
    )
    parser.add_argument(
        "--main_method",
        default="fedplora-oneshot",
        help="Method used as the main method in pairwise Markdown comparison.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log_path in sorted(result_dir.glob(args.log_glob)):
        rows.extend(parse_log(log_path))

    if args.include_json:
        json_paths = sorted(glob.glob(args.json_glob))
        for json_path in json_paths:
            rows.extend(parse_metrics_json(Path(json_path)))

    rows = add_derived_round_metrics(rows)
    summary = summarize_runs(rows)

    write_csv(output_dir / "analysis_rounds.csv", flatten_round_rows(rows))
    write_csv(output_dir / "analysis_summary.csv", summary)
    write_markdown(output_dir / "analysis_summary.md", summary, rows, args.main_method)

    print(f"[analysis] parsed_rounds={len(rows)} runs={len(summary)}")
    print(f"[analysis] wrote {output_dir / 'analysis_rounds.csv'}")
    print(f"[analysis] wrote {output_dir / 'analysis_summary.csv'}")
    print(f"[analysis] wrote {output_dir / 'analysis_summary.md'}")


def flatten_round_rows(rows):
    flat = []
    for row in rows:
        out = {k: v for k, v in row.items() if k not in {"domain_losses", "domain_token_accuracy"}}
        for domain, value in sorted((row.get("domain_losses", {}) or {}).items()):
            out[f"loss_{domain}"] = value
        for domain, value in sorted((row.get("domain_token_accuracy", {}) or {}).items()):
            out[f"tok_acc_{domain}"] = value
        flat.append(out)
    return flat


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize v14 unlearning-dividend eval-only JSONs into GO/NO-GO tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean


def _last_round(payload: dict) -> dict:
    rounds = payload.get("rounds") or []
    return rounds[-1] if rounds else {}


def _tok_acc_map(round_payload: dict) -> dict[str, float]:
    out = {}
    for domain, values in (round_payload.get("domain_metrics") or {}).items():
        if isinstance(values, dict) and "token_accuracy" in values:
            out[str(domain)] = float(values["token_accuracy"])
    return out


def _macro_except(values: dict[str, float], forget_domain: str | None) -> float:
    items = [
        float(value)
        for domain, value in values.items()
        if forget_domain is None or str(domain) != str(forget_domain)
    ]
    return float(mean(items)) if items else float("nan")


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _record_from_eval(path: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("checkpoint_meta") or {}
    phase = meta.get("unlearning_phase0") or {}
    if not phase:
        return None
    round_payload = _last_round(payload)
    domain_acc = _tok_acc_map(round_payload)
    forget = phase.get("forget_domain")
    removed = domain_acc.get(str(forget), float("nan")) if forget else float("nan")
    remaining = _macro_except(domain_acc, str(forget)) if forget else _macro_except(domain_acc, None)
    return {
        "path": str(path),
        "tag": str(phase.get("tag", path.parent.name)),
        "arm": str(phase.get("arm", "")),
        "forget_domain": None if forget in (None, "") else str(forget),
        "removed_domain_token_accuracy": removed,
        "remaining_domains_macro_token_accuracy": remaining,
        "all_domain_macro_token_accuracy": _safe_float(round_payload.get("domain_macro_token_accuracy")),
        "worst_domain_token_accuracy": _safe_float(round_payload.get("worst_domain_token_accuracy")),
        "client_local_macro_token_accuracy": _safe_float(round_payload.get("client_local_macro_token_accuracy")),
        "in_domain_domain_test_macro_token_accuracy": _safe_float(round_payload.get("in_domain_domain_test_macro_token_accuracy")),
        "off_domain_macro_token_accuracy": _safe_float(round_payload.get("off_domain_macro_token_accuracy")),
        "domain_acc": domain_acc,
        "meta": phase,
    }


def _derive_forget_record(global_record: dict, forget_domain: str) -> dict:
    domain_acc = global_record.get("domain_acc") or {}
    rec = dict(global_record)
    rec["forget_domain"] = str(forget_domain)
    rec["removed_domain_token_accuracy"] = domain_acc.get(str(forget_domain), float("nan"))
    rec["remaining_domains_macro_token_accuracy"] = _macro_except(domain_acc, str(forget_domain))
    rec["derived_from_global_arm"] = True
    return rec


def _best(records: list[dict], predicate) -> dict | None:
    candidates = [
        row
        for row in records
        if predicate(row)
        and math.isfinite(float(row.get("remaining_domains_macro_token_accuracy", float("nan"))))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["remaining_domains_macro_token_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json_glob", action="append", required=True, help="Glob for fed_train_sft eval-only result JSONs. Can be passed multiple times.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--projection_arm", default="proj_auto")
    parser.add_argument("--min_success_domains", type=int, default=4)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.eval_json_glob:
        paths.extend(Path(p).resolve() for p in glob.glob(pattern, recursive=True))
    paths = sorted(set(paths))
    records = []
    skipped = 0
    for path in paths:
        try:
            rec = _record_from_eval(path)
        except Exception as exc:
            skipped += 1
            print(f"[unlearning-summary][warn] skip {path}: {exc}", flush=True)
            continue
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    if not records:
        raise SystemExit("[unlearning-summary][error] no v14 unlearning eval JSONs found")

    global_records = {row["arm"]: row for row in records if row.get("forget_domain") is None}
    forget_domains = sorted({row["forget_domain"] for row in records if row.get("forget_domain")})
    detailed = []
    for forget in forget_domains:
        for row in records:
            if row.get("forget_domain") == forget:
                detailed.append(row)
        for arm in ("base", "pool_all", "routed_domain"):
            if arm in global_records:
                detailed.append(_derive_forget_record(global_records[arm], forget))

    go_rows = []
    for forget in forget_domains:
        scoped = [row for row in detailed if row.get("forget_domain") == forget]
        pool_all = _best(scoped, lambda row: row.get("arm") == "pool_all")
        proj = _best(scoped, lambda row: row.get("arm") == args.projection_arm)
        task = _best(scoped, lambda row: str(row.get("arm", "")).startswith("task_arith"))
        random = _best(scoped, lambda row: str(row.get("arm", "")).startswith("random_proj"))
        pool_loo = _best(scoped, lambda row: row.get("arm") == "pool_loo")
        if not pool_all or not proj:
            continue
        proj_rem = float(proj["remaining_domains_macro_token_accuracy"])
        pool_rem = float(pool_all["remaining_domains_macro_token_accuracy"])
        task_rem = float(task["remaining_domains_macro_token_accuracy"]) if task else float("nan")
        random_rem = float(random["remaining_domains_macro_token_accuracy"]) if random else float("nan")
        proj_removed = float(proj["removed_domain_token_accuracy"])
        pool_removed = float(pool_all["removed_domain_token_accuracy"])
        conditions = {
            "proj_gt_pool_all_remaining": bool(proj_rem > pool_rem),
            "proj_gt_task_arith_remaining": bool(not task or proj_rem > task_rem),
            "proj_gt_random_proj_remaining": bool(not random or proj_rem > random_rem),
            "proj_lt_pool_all_removed": bool(proj_removed < pool_removed),
        }
        go_rows.append(
            {
                "forget_domain": forget,
                "projection_arm": args.projection_arm,
                "proj_remaining": proj_rem,
                "pool_all_remaining": pool_rem,
                "best_task_arith_remaining": task_rem,
                "best_random_proj_remaining": random_rem,
                "pool_loo_remaining": (
                    float(pool_loo["remaining_domains_macro_token_accuracy"]) if pool_loo else float("nan")
                ),
                "proj_removed": proj_removed,
                "pool_all_removed": pool_removed,
                "delta_proj_minus_pool_all": proj_rem - pool_rem,
                "delta_proj_minus_task_arith": proj_rem - task_rem if math.isfinite(task_rem) else float("nan"),
                "delta_proj_minus_random_proj": proj_rem - random_rem if math.isfinite(random_rem) else float("nan"),
                "conditions": conditions,
                "go_for_domain": all(conditions.values()),
            }
        )

    success_count = sum(1 for row in go_rows if row["go_for_domain"])
    verdict = {
        "go": bool(success_count >= int(args.min_success_domains)),
        "success_domains": int(success_count),
        "total_domains": int(len(go_rows)),
        "min_success_domains": int(args.min_success_domains),
        "projection_arm": args.projection_arm,
        "interpretation": (
            "GO: projection surgery beats the fixed baselines on enough domains."
            if success_count >= int(args.min_success_domains)
            else "NO-GO: do not spend more seeds/FT budget until the failed conditions are understood."
        ),
    }

    csv_path = Path(args.output_csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "forget_domain",
                "tag",
                "arm",
                "removed_domain_token_accuracy",
                "remaining_domains_macro_token_accuracy",
                "all_domain_macro_token_accuracy",
                "worst_domain_token_accuracy",
                "client_local_macro_token_accuracy",
                "in_domain_domain_test_macro_token_accuracy",
                "off_domain_macro_token_accuracy",
                "path",
            ],
        )
        writer.writeheader()
        for row in sorted(detailed, key=lambda r: (str(r.get("forget_domain")), str(r.get("arm")), str(r.get("tag")))):
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    out_json = Path(args.output_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_eval_jsons": len(records),
                "num_skipped": skipped,
                "verdict": verdict,
                "go_rows": go_rows,
                "detailed_rows": detailed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[unlearning-summary][done] go={verdict['go']} "
        f"success={success_count}/{len(go_rows)} csv={csv_path} json={out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()

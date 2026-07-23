#!/usr/bin/env python3
"""Audit and resolve final FedPLoRA checkpoint bundles without guessing paths."""

import argparse
import json
from pathlib import Path


def _records(roots):
    records = []
    for root_text in roots:
        root = Path(root_text).expanduser().resolve()
        if not root.exists():
            continue
        for meta_path in root.rglob("run_checkpoint_meta.json"):
            bundle = meta_path.parent
            ok_path = bundle / "checkpoint_ok.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ok = json.loads(ok_path.read_text(encoding="utf-8")) if ok_path.is_file() else {}
            except Exception as exc:
                records.append({"bundle": str(bundle), "valid": False, "error": repr(exc)})
                continue
            phase = str(ok.get("checkpoint_phase", ok.get("phase", "")))
            valid = bool(ok.get("ok", False)) and phase == "final"
            records.append({
                "bundle": str(bundle),
                "valid": valid,
                "checkpoint_phase": phase,
                "agg_type": meta.get("agg_type"),
                "seed": meta.get("seed"),
                "model": meta.get("model"),
                "benchmark_dir": meta.get("benchmark_dir"),
                "train_rounds": meta.get("train_rounds"),
                "train_local_epochs": meta.get("train_local_epochs"),
                "benchmark_fingerprint": meta.get("benchmark_fingerprint"),
                "has_full_clients": (bundle / "full_clients.pt").is_file(),
                "has_global_shared": (bundle / "global_shared.pt").is_file(),
                "has_clients_dir": (bundle / "clients").is_dir(),
            })
    return sorted(records, key=lambda row: row["bundle"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--agg_type", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_contains", default="")
    parser.add_argument("--benchmark_contains", default="")
    args = parser.parse_args()
    records = _records(args.roots)
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[checkpoint-manifest] wrote {len(records)} records -> {out}")
    if not args.resolve:
        valid = sum(bool(row.get("valid")) for row in records)
        print(f"[checkpoint-manifest] discovered={len(records)} valid_final={valid}")
        return
    selected = [row for row in records if row.get("valid")]
    if args.agg_type:
        selected = [row for row in selected if row.get("agg_type") == args.agg_type]
    if args.seed is not None:
        selected = [row for row in selected if int(row.get("seed", -1)) == args.seed]
    if args.model_contains:
        selected = [row for row in selected if args.model_contains in str(row.get("model", ""))]
    if args.benchmark_contains:
        selected = [row for row in selected if args.benchmark_contains in str(row.get("benchmark_dir", ""))]
    if len(selected) != 1:
        details = "\n".join(row["bundle"] for row in selected[:20])
        raise SystemExit(
            f"[checkpoint-manifest][error] expected exactly one match, found {len(selected)}\n{details}"
        )
    print(selected[0]["bundle"])


if __name__ == "__main__":
    main()

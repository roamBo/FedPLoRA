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
<<<<<<< HEAD
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--resolve", action="store_true")
=======
    parser.add_argument("--roots", nargs="*", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--list_matches", action="store_true")
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
    parser.add_argument("--agg_type", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_contains", default="")
    parser.add_argument("--benchmark_contains", default="")
    parser.add_argument("--bundle_contains", default="")
    parser.add_argument(
        "--from_result_json",
        default="",
        help="Resolve checkpoint dir from a formal result JSON save_run_checkpoint_dir field.",
    )
    args = parser.parse_args()
    if args.from_result_json:
        result_path = Path(args.from_result_json).expanduser().resolve()
        if not result_path.is_file():
            raise SystemExit(f"[checkpoint-manifest][error] missing result JSON: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run_args = payload.get("args") or {}
        effective = payload.get("effective_hparams") or {}
        ckpt_text = run_args.get("save_run_checkpoint_dir") or effective.get("save_run_checkpoint_dir")
        if not ckpt_text:
            raise SystemExit(
                f"[checkpoint-manifest][error] no save_run_checkpoint_dir in {result_path}"
            )
        bundle = Path(str(ckpt_text)).expanduser().resolve()
        meta_path = bundle / "run_checkpoint_meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"[checkpoint-manifest][error] missing {meta_path}")
        ok_path = bundle / "checkpoint_ok.json"
        if ok_path.is_file():
            ok = json.loads(ok_path.read_text(encoding="utf-8"))
            phase = str(ok.get("checkpoint_phase", ok.get("phase", "")))
            if not (bool(ok.get("ok", False)) and phase == "final"):
                raise SystemExit(
                    f"[checkpoint-manifest][error] checkpoint not final-ok: {bundle} phase={phase!r}"
                )
        print(str(bundle))
        return
<<<<<<< HEAD
=======
    if not args.roots:
        raise SystemExit(
            "[checkpoint-manifest][error] --roots is required unless --from_result_json is set"
        )
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
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
    if args.bundle_contains:
        selected = [row for row in selected if args.bundle_contains in str(row.get("bundle", ""))]
<<<<<<< HEAD
=======
    if args.list_matches:
        for row in selected:
            print(row["bundle"])
        print(f"[checkpoint-manifest] list_matches count={len(selected)}", flush=True)
        return
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
    if len(selected) != 1:
        details = "\n".join(row["bundle"] for row in selected[:20])
        raise SystemExit(
            f"[checkpoint-manifest][error] expected exactly one match, found {len(selected)}\n{details}"
        )
    print(selected[0]["bundle"])


if __name__ == "__main__":
    main()

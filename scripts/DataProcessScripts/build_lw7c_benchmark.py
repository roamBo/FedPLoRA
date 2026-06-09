#!/usr/bin/env python3
"""
Build LW7c benchmark: 7 clients (1/domain), each with ~1/5 of a 35c single-client shard.

Method A (default): derive from checked-in domain_benchmark_35c by keeping the first
client per domain (same train count as one 35c client, NOT 5 clients merged).

Method B: rebuild from raw JSONL with --from_jsonl and --per_client_data_fraction 0.2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_from_35c(src_dir: str, out_dir: str, seed: int = 42) -> dict:
    src_split = os.path.join(src_dir, f"seed_{seed}")
    out_split = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(out_split, exist_ok=True)

    with open(os.path.join(src_split, "clients.json"), "r", encoding="utf-8") as f:
        src_clients = json.load(f)

    first_by_domain = {}
    for c in sorted(src_clients, key=lambda x: int(x["client_id"])):
        dom = c["domain"]
        if dom not in first_by_domain:
            first_by_domain[dom] = int(c["client_id"])

    keep_old_ids = set(first_by_domain.values())
    old_to_new = {old: new for new, old in enumerate(sorted(keep_old_ids))}

    def _remap(rows):
        out = []
        for row in rows:
            cid = int(row.get("client_id", -1))
            if cid not in keep_old_ids:
                continue
            row = dict(row)
            row["client_id"] = old_to_new[cid]
            out.append(row)
        return out

    train = _remap(_load_jsonl(os.path.join(src_split, "train.jsonl")))
    val = _remap(_load_jsonl(os.path.join(src_split, "val.jsonl")))
    test_local = _remap(_load_jsonl(os.path.join(src_split, "test_local.jsonl")))
    test_domain = _load_jsonl(os.path.join(src_split, "test_domain.jsonl"))
    test_global = _load_jsonl(os.path.join(src_split, "test_global.jsonl"))

    _write_jsonl(os.path.join(out_split, "train.jsonl"), train)
    _write_jsonl(os.path.join(out_split, "val.jsonl"), val)
    _write_jsonl(os.path.join(out_split, "test_local.jsonl"), test_local)
    _write_jsonl(os.path.join(out_split, "test_domain.jsonl"), test_domain)
    _write_jsonl(os.path.join(out_split, "test_global.jsonl"), test_global)

    new_clients = []
    for old_id, new_id in sorted(old_to_new.items(), key=lambda x: x[1]):
        src = next(c for c in src_clients if int(c["client_id"]) == old_id)
        new_clients.append(
            {
                "client_id": new_id,
                "domain": src["domain"],
                "n_train": src["n_train"],
                "n_val": src["n_val"],
                "n_local_test": src["n_local_test"],
            }
        )

    with open(os.path.join(out_split, "clients.json"), "w", encoding="utf-8") as f:
        json.dump(new_clients, f, indent=2, ensure_ascii=False)

    with open(os.path.join(src_split, "domain_stats.json"), "r", encoding="utf-8") as f:
        src_stats = json.load(f)
    domain_stats = {}
    for c in new_clients:
        dom = c["domain"]
        if dom not in domain_stats:
            domain_stats[dom] = dict(src_stats.get(dom, {}))
            domain_stats[dom]["n_clients"] = 1
    with open(os.path.join(out_split, "domain_stats.json"), "w", encoding="utf-8") as f:
        json.dump(domain_stats, f, indent=2, ensure_ascii=False)

    manifest = {
        "benchmark_tag": "LW7c",
        "derived_from": os.path.abspath(src_split),
        "num_clients": len(new_clients),
        "per_client_data": "one 35c client shard per domain (~1/5 of domain train pool)",
        "seed": seed,
    }
    with open(os.path.join(out_split, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {"split_dir": out_split, "clients": new_clients, "manifest": manifest}


def main():
    p = argparse.ArgumentParser(description="Build domain_benchmark_LW7c (7c, 1/5 data/client).")
    p.add_argument(
        "--src_35c",
        type=str,
        default="data/domain_benchmark_35c",
        help="Source 35-client benchmark (default: checked-in 35c).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/domain_benchmark_LW7c",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--from_jsonl",
        type=str,
        default="",
        help="If set, rebuild from raw JSONL instead of deriving from 35c.",
    )
    p.add_argument("--per_client_data_fraction", type=float, default=0.2)
    p.add_argument("--min_samples_per_client", type=int, default=10)
    args = p.parse_args()

    if args.from_jsonl:
        from utilities.data_utils import build_domain_benchmark_from_jsonl  # noqa: WPS433

        info = build_domain_benchmark_from_jsonl(
            input_path=args.from_jsonl,
            output_dir=args.output_dir,
            num_clients_per_domain=1,
            min_samples_per_client=args.min_samples_per_client,
            per_client_data_fraction=args.per_client_data_fraction,
            seed=args.seed,
        )
        split_dir = info["split_dir"]
        with open(os.path.join(split_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "benchmark_tag": "LW7c",
                    "built_from_jsonl": os.path.abspath(args.from_jsonl),
                    "per_client_data_fraction": args.per_client_data_fraction,
                    "seed": args.seed,
                },
                f,
                indent=2,
            )
        clients_path = os.path.join(split_dir, "clients.json")
        with open(clients_path, "r", encoding="utf-8") as f:
            clients = json.load(f)
    else:
        info = build_from_35c(args.src_35c, args.output_dir, seed=args.seed)
        split_dir = info["split_dir"]
        clients = info["clients"]

    print(f"[ok] LW7c split_dir={split_dir}")
    print(f"[ok] num_clients={len(clients)}")
    for c in clients:
        print(
            f"  client {c['client_id']} domain={c['domain']} "
            f"train={c['n_train']} val={c['n_val']} local_test={c['n_local_test']}"
        )


if __name__ == "__main__":
    main()

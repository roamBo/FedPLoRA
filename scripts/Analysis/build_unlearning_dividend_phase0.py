#!/usr/bin/env python3
"""Build v14 unlearning-dividend Phase-0 synthetic checkpoints.

Typical use on the server:

  python scripts/Analysis/build_unlearning_dividend_phase0.py \
    --checkpoint_dir /data2/.../trained_models/.../fedplora_v13a_os... \
    --clients_json /data2/.../data/domain_benchmark_35c_dir05/seed_42/clients.json \
    --benchmark_dir /data2/.../data/domain_benchmark_35c_dir05/seed_42 \
    --model /data2/minghao/model/SmolLM2-135M \
    --output_dir /data2/minghao/result/FedPLoRA/unlearning_20260730/phase0_d1_seed42 \
    --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.v14 import (  # noqa: E402
    build_phase0_arms,
    load_client_rows,
    load_source_states,
    save_phase0_checkpoints,
)


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _split_floats(text: str) -> list[float]:
    return [float(item) for item in _split_csv(text)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize LoRA-B unlearning-dividend Phase-0 arms as synthetic eval-only checkpoints."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint_dir", type=str, default="", help="Final run checkpoint containing global_shared.pt + clients/ or full_clients.pt.")
    src.add_argument("--state_dir", type=str, default="", help="Raw client_*.pt directory. Requires --shared_state if states contain B only.")
    parser.add_argument("--shared_state", type=str, default="", help="Optional global_shared.pt when using --state_dir.")
    parser.add_argument("--clients_json", type=str, required=True)
    parser.add_argument("--benchmark_dir", type=str, default="", help="Stored into synthetic checkpoint metadata; defaults to clients_json parent.")
    parser.add_argument("--model", type=str, default="", help="Stored into synthetic checkpoint metadata; eval command can still override.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--forget_domains", type=str, default="all", help="'all' or comma-separated domain names.")
    parser.add_argument("--weights", type=str, default="sample", choices=["sample", "uniform"])
    parser.add_argument("--energy_tau", type=float, default=0.90)
    parser.add_argument("--projection_ranks", type=str, default="auto", help="Comma list: auto,1,2,4 ...")
    parser.add_argument("--task_arith_lambdas", type=str, default="0.5,1.0")
    parser.add_argument("--random_trials", type=int, default=1)
    parser.add_argument("--no_routed", action="store_true", help="Do not emit the routed-domain upper-reference checkpoint.")
    parser.add_argument("--expected_clients", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--rslora", action="store_true")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_symlink_clients", action="store_true", help="Copy replicated client states instead of using relative symlinks.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build in memory and print the manifest summary without writing checkpoints.")
    args = parser.parse_args()

    client_rows = load_client_rows(args.clients_json)
    if args.expected_clients and len(client_rows) != int(args.expected_clients):
        raise SystemExit(
            f"[unlearning-build][error] expected {args.expected_clients} clients in clients.json, found {len(client_rows)}"
        )
    source = load_source_states(
        checkpoint_dir=args.checkpoint_dir or None,
        state_dir=args.state_dir or None,
        shared_state_path=args.shared_state or None,
        client_rows=client_rows,
    )
    if args.expected_clients and len(source.client_ids) != int(args.expected_clients):
        raise SystemExit(
            f"[unlearning-build][error] expected {args.expected_clients} source states, found {len(source.client_ids)}"
        )

    all_domains = sorted({str(row["domain"]) for row in client_rows})
    forget_domains = all_domains if args.forget_domains.strip().lower() == "all" else _split_csv(args.forget_domains)
    benchmark_dir = args.benchmark_dir or str(Path(args.clients_json).expanduser().resolve().parent)
    result = build_phase0_arms(
        source,
        client_rows,
        forget_domains=forget_domains,
        weight_mode=args.weights,
        energy_tau=float(args.energy_tau),
        projection_ranks=_split_csv(args.projection_ranks),
        task_arith_lambdas=_split_floats(args.task_arith_lambdas),
        random_trials=int(args.random_trials),
        include_routed=not bool(args.no_routed),
        seed=int(args.seed),
    )

    print(
        "[unlearning-build] "
        f"source={source.source_path} clients={len(source.client_ids)} "
        f"domains={result.summary['domain_client_counts']} arms={len(result.arms)}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(result.summary, ensure_ascii=False, indent=2), flush=True)
        return

    manifest = save_phase0_checkpoints(
        result,
        output_dir=args.output_dir,
        source=source,
        benchmark_dir=benchmark_dir,
        model=args.model,
        seed=int(args.seed),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        rslora=bool(args.rslora),
        target_modules=args.target_modules,
        torch_dtype=args.torch_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        gradient_checkpointing=not bool(args.no_gradient_checkpointing),
        symlink_replicated_clients=not bool(args.no_symlink_clients),
        force=bool(args.force),
    )
    manifest_path = Path(args.output_dir).expanduser().resolve() / "phase0_manifest.json"
    print(
        f"[unlearning-build][ok] checkpoints={len(manifest['checkpoints'])} manifest={manifest_path}",
        flush=True,
    )
    for row in manifest["checkpoints"][:10]:
        print(
            f"[unlearning-build][ckpt] tag={row['tag']} arm={row['arm']} "
            f"forget={row['forget_domain']} dir={row['checkpoint_dir']}",
            flush=True,
        )
    if len(manifest["checkpoints"]) > 10:
        print(f"[unlearning-build] ... {len(manifest['checkpoints']) - 10} more checkpoints", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

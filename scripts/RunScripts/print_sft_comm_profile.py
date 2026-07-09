#!/usr/bin/env python3
"""
Print per-round upload/download byte estimates for domain SFT agg types (same rule as training log [setup] line).

Loads the PEFT causal LM once; requires a local or HF model path (can be slow on first run).

Usage (repo root):
  python scripts/RunScripts/print_sft_comm_profile.py --model /path/to/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utilities.models import create_peft_causal_lm_model  # noqa: E402
from utilities.utils import estimate_round_communication_bytes, get_trainable_param_names  # noqa: E402


DEFAULT_AGGS = [
    "normal",
    "ffa",
    "flora",
    "flexlora",
    "feddat",
    "fedplora-oneshot",
    "fedplora_v3_lite",
    "fedplora_v3_cluster",
    "fedplora_v3_rpca",
    "v6_dcr_global",
    "v6_dcr_domain",
    "yoco",
    "fedsa_lora",
    "fedalt",
    "fedplora_v7",
    "fedplora_v7_bonly",
    "fedplora_v8",
    "fedplora_v8_ab",
    "fedplora_v8_warma",
    "fedplora_v8_periodic",
    "fedplora_v11c_gmix",
    "fedplora_v12a_sched_gmix",
    "fedplora_v12b_nmi_guard_gmix",
    "fedlease",
    "hilora",
    "ecolora",
    "hydralora",
]


def main():
    p = argparse.ArgumentParser(description="SFT communication profile (bytes per client per round)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--rslora", action="store_true")
    p.add_argument("--torch_dtype", type=str, default="bfloat16")
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--ecolora_keep_ratio",
        type=float,
        default=0.25,
        help="EcoLoRA sparse/segment upload keep ratio used for communication estimates.",
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--v8_a_warmup_rounds", type=int, default=1)
    p.add_argument("--v8_a_refresh_interval", type=int, default=5)
    p.add_argument(
        "--v8_cache_shared_a_downlink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, v8 effective downlink assumes cached shared A and routed B-only non-refresh rounds.",
    )
    p.add_argument(
        "--agg_types",
        type=str,
        default=",".join(DEFAULT_AGGS),
        help="Comma-separated agg_type list",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object to stdout")
    args_ns = p.parse_args()

    class _A:
        pass

    a = _A()
    a.model = args_ns.model
    a.lora_r = args_ns.lora_r
    a.lora_alpha = args_ns.lora_alpha
    a.lora_dropout = args_ns.lora_dropout
    a.rslora = args_ns.rslora
    a.torch_dtype = args_ns.torch_dtype
    a.target_modules = args_ns.target_modules
    a.trust_remote_code = args_ns.trust_remote_code
    a.gradient_checkpointing = args_ns.gradient_checkpointing

    model = create_peft_causal_lm_model(a)
    sd = model.state_dict()
    trainable = get_trainable_param_names(model)
    aggs = [x.strip() for x in args_ns.agg_types.split(",") if x.strip()]

    rows = []
    for agg in aggs:
        info = estimate_round_communication_bytes(
            sd,
            agg,
            trainable_param_names=trainable,
            ecolora_keep_ratio=args_ns.ecolora_keep_ratio,
            rounds=args_ns.rounds,
            v8_a_warmup_rounds=args_ns.v8_a_warmup_rounds,
            v8_a_refresh_interval=args_ns.v8_a_refresh_interval,
            v8_cache_shared_a_downlink=args_ns.v8_cache_shared_a_downlink,
        )
        rows.append(
            {
                "agg_type": agg,
                "down_bytes_per_client": int(info["down_bytes_per_client"]),
                "up_bytes_per_client": int(info["up_bytes_per_client"]),
                "effective_down_bytes_per_client": int(
                    info["effective_down_bytes_per_client"]
                ),
                "effective_up_bytes_per_client": int(
                    info["effective_up_bytes_per_client"]
                ),
                "downlink_policy": str(info.get("downlink_policy", "raw")),
                "v8_a_refresh_rounds": int(info.get("v8_a_refresh_rounds", 0)),
            }
        )

    if args_ns.json:
        print(json.dumps({"model": args_ns.model, "methods": rows}, indent=2))
        return

    print(f"[comm_profile] model={args_ns.model}")
    print(
        f"{'agg_type':<24} {'down_B':>14} {'up_B':>14} "
        f"{'eff_down_B':>14} {'eff_up_B':>14} {'refresh':>8}"
    )
    for r in rows:
        d = r["down_bytes_per_client"]
        u = r["up_bytes_per_client"]
        ed = r["effective_down_bytes_per_client"]
        eu = r["effective_up_bytes_per_client"]
        print(
            f"{r['agg_type']:<24} {d:>14} {u:>14} {ed:>14} {eu:>14} "
            f"{r['v8_a_refresh_rounds']:>8}"
        )
    print(
        "[note] All methods count LoRA (+ heads) only; frozen backbone is never in link budget. "
        "FedPLoRA / oneshot / DCR / fedsa: A+heads (+ row stats uplink); fedalt: personalized A+B; "
        "ffa, fedplora_v7_bonly, and fedplora_v8: B+heads uplink; "
        "fedplora_v8_ab uses A+B+heads as the A+B ablation; "
        "fedplora_v8_warma/periodic amortize A refreshes in effective_* columns; "
        "ecolora: keep_ratio*(A+B)+heads uplink; "
        "normal/flora/flexlora/feddat/yoco/fedlease/hilora/hydralora/fedplora_v7: A+B+heads."
    )


if __name__ == "__main__":
    main()

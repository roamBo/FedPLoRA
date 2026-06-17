"""
diag_b_swap.py — 同域 B 互换诊断：判定"域内 B 可否共享/去噪"还是"客户端独有"。

范数公平的测法（不做平均，避免近正交平均的范数坍塌）：
  对每个客户端 i（域 d），在 i 自己域的 held-out 测试上评测：
    local      : (A_i, B_i)                         本地基线
    swap_intra : (A_i, B_j)  j 为同域其它客户端       同域 B 能否服务我
    swap_inter : (A_i, B_k)  k 为跨域客户端           跨域 B（应明显更差，作上界对照）
  比较 loss / token_acc 的增量。

判读：
  Δ_intra ≈ 0  且 Δ_intra ≪ Δ_inter  -> 同域 B 可互换 -> 域内共享/去噪可行 -> 新方法成立。
  Δ_intra 接近 Δ_inter                -> B 客户端独有 -> 连域内都不能共享 -> B 必须私有(退回 FedSA)。

用法（仓库根目录）：
  python scripts/Analysis/diag_b_swap.py \
    --model /path/to/SmolLM2-135M \
    --benchmark_dir data/domain_benchmark_35c/seed_42 \
    --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
    --max_seq_length 512 --max_steps 0 \
    --eval_max_batches 20 --n_peers 4 --n_cross 2 \
    --out artifacts_LW/diag_b_swap_smol_35c.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities.data_utils import (  # noqa: E402
    load_domain_sft_benchmark,
    create_domain_client_dataloaders,
    create_domain_eval_dataloader,
    group_rows_by_domain,
)
from utilities.models import create_peft_causal_lm_model, init_fedplora_adapters  # noqa: E402
from utilities.utils import is_lora_a_param_name, is_lora_b_param_name  # noqa: E402


def _resolve_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


def _snapshot(model, pred):
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items() if pred(k)}


def _reset_adapters(model, A0):
    sd = model.state_dict()
    for k, v in sd.items():
        if is_lora_a_param_name(k) and k in A0:
            sd[k] = A0[k].to(device=v.device, dtype=v.dtype)
        elif is_lora_b_param_name(k):
            sd[k] = torch.zeros_like(v)
    model.load_state_dict(sd)


def _install_AB(model, A_dict, B_dict):
    """Load a given (A, B) combo into the model's lora_A / lora_B weights."""
    sd = model.state_dict()
    for k, v in A_dict.items():
        if k in sd:
            sd[k] = v.to(device=sd[k].device, dtype=sd[k].dtype)
    for k, v in B_dict.items():
        if k in sd:
            sd[k] = v.to(device=sd[k].device, dtype=sd[k].dtype)
    model.load_state_dict(sd)


def _train_one_client(model, loader, args, device):
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    model.train()
    amp_dtype = _resolve_dtype(args.torch_dtype)
    use_amp = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    step = 0
    for _ in range(max(1, int(args.local_epochs))):
        for batch in loader:
            if args.max_steps and step >= args.max_steps:
                return
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            step += 1


@torch.no_grad()
def _eval_loss_acc(model, loader, device, max_batches):
    model.eval()
    total_loss, steps, correct, valid = 0.0, 0, 0, 0
    for batch in loader:
        if max_batches and steps >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += float(out.loss.detach().cpu().item())
        logits = out.logits[..., :-1, :]
        labels = batch["labels"][..., 1:]
        pred = logits.argmax(dim=-1)
        mask = labels.ne(-100)
        if mask.any():
            correct += int((pred[mask] == labels[mask]).sum().cpu())
            valid += int(mask.sum().cpu())
        steps += 1
    return total_loss / max(steps, 1), correct / max(valid, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--rslora", action="store_true")
    ap.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=0)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_clients", type=int, default=0)
    ap.add_argument("--max_train_samples_per_client", type=int, default=0)
    ap.add_argument("--eval_max_batches", type=int, default=20)
    ap.add_argument("--n_peers", type=int, default=4, help="每客户端采样的同域 peer 数")
    ap.add_argument("--n_cross", type=int, default=2, help="每客户端采样的跨域 client 数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="diag_b_swap.json")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False,
                                        trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bench = load_domain_sft_benchmark(args.benchmark_dir)
    domain_of = {int(c["client_id"]): str(c["domain"]) for c in bench["clients"]}
    client_ids, loaders = create_domain_client_dataloaders(bench["train"], tok, args)
    if args.max_clients and len(client_ids) > args.max_clients:
        client_ids, loaders = client_ids[:args.max_clients], loaders[:args.max_clients]

    by_dom = defaultdict(list)
    for cid in client_ids:
        by_dom[domain_of.get(cid, "?")].append(cid)
    if max(len(v) for v in by_dom.values()) < 2:
        print("[warn] 每域客户端数<2，互换诊断无意义，请用 14c/21c/35c。", flush=True)

    # per-domain eval loaders (held-out domain test)
    test_by_dom = group_rows_by_domain(bench["test_domain"])
    eval_loaders = {d: create_domain_eval_dataloader(rows, tok, args)
                    for d, rows in test_by_dom.items()}

    model = create_peft_causal_lm_model(args)
    init_fedplora_adapters(model)
    A0 = _snapshot(model, is_lora_a_param_name)

    # --- capture trained A_i, B_i per client ---
    A_by, B_by = {}, {}
    for n, cid in enumerate(client_ids):
        _reset_adapters(model, A0)
        _train_one_client(model, loaders[n], args, device)
        sd = model.state_dict()
        A_by[cid] = {k: sd[k].detach().cpu().clone() for k in A0}
        B_by[cid] = {k.replace("lora_A", "lora_B"): sd[k.replace("lora_A", "lora_B")].detach().cpu().clone()
                     for k in A0 if k.replace("lora_A", "lora_B") in sd}
        print(f"[diag] captured {cid} ({domain_of.get(cid,'?')}) [{n+1}/{len(client_ids)}]", flush=True)

    # --- swap eval ---
    per_client = []
    for cid in client_ids:
        d = domain_of.get(cid, "?")
        dl = eval_loaders.get(d)
        if dl is None:
            continue
        # local
        _install_AB(model, A_by[cid], B_by[cid])
        l_loc, a_loc = _eval_loss_acc(model, dl, device, args.eval_max_batches)
        # same-domain peers
        peers = [c for c in by_dom[d] if c != cid]
        rng.shuffle(peers)
        peers = peers[:args.n_peers]
        intra = []
        for j in peers:
            _install_AB(model, A_by[cid], B_by[j])
            lj, aj = _eval_loss_acc(model, dl, device, args.eval_max_batches)
            intra.append((lj, aj))
        # cross-domain
        cross = [c for c in client_ids if domain_of.get(c) != d]
        rng.shuffle(cross)
        cross = cross[:args.n_cross]
        inter = []
        for k in cross:
            _install_AB(model, A_by[cid], B_by[k])
            lk, ak = _eval_loss_acc(model, dl, device, args.eval_max_batches)
            inter.append((lk, ak))
        rec = {
            "client": int(cid), "domain": d,
            "loss_local": l_loc, "acc_local": a_loc,
            "loss_swap_intra": float(np.mean([x[0] for x in intra])) if intra else float("nan"),
            "acc_swap_intra": float(np.mean([x[1] for x in intra])) if intra else float("nan"),
            "loss_swap_inter": float(np.mean([x[0] for x in inter])) if inter else float("nan"),
            "acc_swap_inter": float(np.mean([x[1] for x in inter])) if inter else float("nan"),
        }
        per_client.append(rec)
        print(f"[swap] c{cid}({d}) local={l_loc:.4f} intra={rec['loss_swap_intra']:.4f} "
              f"inter={rec['loss_swap_inter']:.4f}", flush=True)

    # --- aggregate ---
    def col(key):
        return np.array([r[key] for r in per_client if not math.isnan(r[key])])
    l_loc = col("loss_local"); l_in = col("loss_swap_intra"); l_ix = col("loss_swap_inter")
    a_loc = col("acc_local"); a_in = col("acc_swap_intra"); a_ix = col("acc_swap_inter")
    d_intra = float(np.mean(l_in - l_loc))
    d_inter = float(np.mean(l_ix - l_loc))
    summary = {
        "n_clients_eval": len(per_client),
        "loss_local_mean": float(np.mean(l_loc)),
        "loss_swap_intra_mean": float(np.mean(l_in)),
        "loss_swap_inter_mean": float(np.mean(l_ix)),
        "delta_loss_intra": d_intra,          # 同域 B 互换的 loss 增量
        "delta_loss_inter": d_inter,          # 跨域 B 互换的 loss 增量（上界对照）
        "delta_intra_pct_of_local": float(d_intra / max(np.mean(l_loc), 1e-9) * 100),
        "intra_over_inter_ratio": float(d_intra / d_inter) if abs(d_inter) > 1e-9 else float("nan"),
        "acc_local_mean": float(np.mean(a_loc)),
        "acc_swap_intra_mean": float(np.mean(a_in)),
        "acc_swap_inter_mean": float(np.mean(a_ix)),
    }
    report = {"config": {k: getattr(args, k) for k in
                         ["model", "benchmark_dir", "target_modules", "max_steps",
                          "eval_max_batches", "n_peers", "n_cross"]},
              "summary": summary, "per_client": per_client}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n============ 同域 B 互换诊断结果 ============")
    print(f"loss  local={summary['loss_local_mean']:.4f}  "
          f"swap_intra={summary['loss_swap_intra_mean']:.4f}  "
          f"swap_inter={summary['loss_swap_inter_mean']:.4f}")
    print(f"acc   local={summary['acc_local_mean']:.4f}  "
          f"swap_intra={summary['acc_swap_intra_mean']:.4f}  "
          f"swap_inter={summary['acc_swap_inter_mean']:.4f}")
    print(f"Δloss 同域互换={d_intra:+.4f} ({summary['delta_intra_pct_of_local']:+.1f}% of local)  "
          f"跨域互换={d_inter:+.4f}")
    print(f"intra/inter 比值={summary['intra_over_inter_ratio']:.3f}")
    print("--------------------------------------------")
    ratio = summary["intra_over_inter_ratio"]
    pct = summary["delta_intra_pct_of_local"]
    if pct < 5 and ratio < 0.3:
        print("[判读] 同域 B 互换几乎无损(<5%)且远小于跨域 -> 域内 B 可共享/去噪 -> 新方法成立(域内B共识、跨域隔离)。")
    elif ratio > 0.7:
        print("[判读] 同域 B 互换损失接近跨域 -> B 客户端独有，连域内都不能共享 -> B 必须私有(退回 FedSA)。")
    else:
        print(f"[判读] 中间地带(Δintra={pct:.1f}%, intra/inter={ratio:.2f}) -> 域内 B 部分可共享；"
              "共识(去噪)可能有用但有损，需进一步用训练实验确认。")
    print(f"\n[diag] 已保存: {args.out}", flush=True)


if __name__ == "__main__":
    main()

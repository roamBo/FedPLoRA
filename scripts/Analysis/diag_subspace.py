"""
diag_subspace.py — DCR(v6) 前提验证诊断（在 GPU 服务器上运行）

回答三件事，决定 DCR 是"有机制"还是"空壳"：
  D1 gauge 跨度：同域客户端 A_i 的「逐元素差异」vs「子空间主夹角差异」。
                 若逐元素大但子空间小 -> gauge 在搅局，DCR 有理；
                 若两者都小 -> A_i 近乎相同，DCR 动机弱。
  D2 更新幅度：‖A_i - A0‖/‖A0‖。若普遍很小 -> 问题在「1-epoch 更新太小」regime，换聚合无用。
  D3 域内 vs 域间主夹角 + 随机 null：域间是否显著大于域内且偏离随机？
                 显著 -> 有跨域几何信号；重叠 -> 没有可利用的跨域结构。

用法（仓库根目录）：
  python scripts/Analysis/diag_subspace.py \
    --model /path/to/Meta-Llama-3.1-8B \
    --benchmark_dir data/domain_benchmark_35c/seed_42 \
    --lora_r 8 --lora_alpha 16 --batch_size 2 --max_seq_length 1024 \
    --target_modules q_proj,v_proj --max_steps 60 \
    --out artifacts_35c/diag_subspace.json

提示：
  - 需每域 ≥2 客户端（用 35c/21c/14c，不要用 7c/LW7c）。
  - 快筛先用 --target_modules q_proj,v_proj --max_steps 60 --max_seq_length 1024；
    定稿再全模块全 epoch（--max_steps 0）。
  - --anchor_lambda 0（默认，测原始 gauge 自由度）vs 1e-4（复现真实 oneshot 训练，gauge 被锚定缩小）；
    两个都跑可对比 anchor 是否人为压低了 gauge。
"""

import argparse
import json
import math
import os
import sys
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch

# --- repo imports (run from repo root) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities.data_utils import (  # noqa: E402
    load_domain_sft_benchmark,
    create_domain_client_dataloaders,
)
from utilities.models import create_peft_causal_lm_model, init_fedplora_adapters  # noqa: E402
from utilities.utils import is_lora_a_param_name  # noqa: E402


# --------------------------------------------------------------------------- #
# model / training helpers
# --------------------------------------------------------------------------- #
def _resolve_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


def _snapshot_A(model):
    return {k: v.detach().cpu().float().clone()
            for k, v in model.state_dict().items() if is_lora_a_param_name(k)}


def _reset_adapters(model, A0):
    """Set lora_A back to A0, zero lora_B — fresh shared init for the next client."""
    sd = model.state_dict()
    for k, v in sd.items():
        if is_lora_a_param_name(k) and k in A0:
            sd[k] = A0[k].to(device=v.device, dtype=v.dtype)
        elif "lora_B" in k and k.endswith("default.weight"):
            sd[k] = torch.zeros_like(v)
    model.load_state_dict(sd)


def _anchor_loss(model, A0, lam):
    if lam <= 0:
        return None
    eps = 1e-8
    terms = []
    for name, p in model.named_parameters():
        if "lora_A" not in name or not name.endswith("default.weight") or not p.requires_grad:
            continue
        ref = A0.get(name)
        if ref is None:
            continue
        ref = ref.to(device=p.device, dtype=torch.float32)
        if tuple(ref.shape) != tuple(p.shape):
            continue
        Ad = p.float() / p.float().norm(dim=1, keepdim=True).clamp_min(eps)
        Rd = ref / ref.norm(dim=1, keepdim=True).clamp_min(eps)
        terms.append((1.0 - (Ad * Rd).sum(dim=1).abs().clamp(max=1.0)).mean())
    if not terms:
        return None
    return lam * torch.stack(terms).mean()


def _train_one_client(model, loader, args, A0, device):
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
                    out = model(**batch)
                    loss = out.loss
            else:
                out = model(**batch)
                loss = out.loss
            extra = _anchor_loss(model, A0, float(args.anchor_lambda))
            if extra is not None:
                loss = loss + extra.to(loss.dtype)
            loss.backward()
            optim.step()
            optim.zero_grad()
            step += 1


# --------------------------------------------------------------------------- #
# subspace math
# --------------------------------------------------------------------------- #
def _rowspace_basis(A):
    """Orthonormal basis of the row space of A (r×d) as (r×d) with orthonormal rows."""
    A = A.float()
    # thin SVD: A = U S Vh ; Vh rows (r×d) are orthonormal and span the row space
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = int((S > S.max().clamp_min(1e-12) * 1e-6).sum().item()) if S.numel() else 0
    keep = max(keep, 1)
    return Vh[:keep]


def _principal_angles_deg(Q1, Q2):
    """Principal angles (degrees) between row spaces spanned by Q1, Q2 (orthonormal rows)."""
    M = Q1 @ Q2.transpose(0, 1)
    sv = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    ang = torch.arccos(sv) * (180.0 / math.pi)
    return ang  # ascending sv -> descending angle; vector length = min(r1, r2)


def _rel_elem_diff(Ai, Aj):
    ni = Ai.float().norm()
    nj = Aj.float().norm()
    return float((Ai.float() - Aj.float()).norm() / (0.5 * (ni + nj)).clamp_min(1e-12))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--rslora", action="store_true")
    ap.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                    help="子集可加速；定稿用全 7 模块")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=60,
                    help="每客户端训练步数上限（0=整个 epoch）")
    ap.add_argument("--anchor_lambda", type=float, default=0.0,
                    help="0=测原始 gauge 自由度；1e-4=复现真实 oneshot（gauge 被锚定缩小）")
    ap.add_argument("--max_clients", type=int, default=0, help="0=全部")
    ap.add_argument("--max_layers", type=int, default=0, help="0=全部 lora_A 层；>0 随机抽样该数量层")
    ap.add_argument("--max_train_samples_per_client", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_null", type=int, default=200, help="随机子空间 null 对数")
    ap.add_argument("--out", type=str, default="diag_subspace.json")
    ap.add_argument("--save_figs", action="store_true", help="需要 matplotlib")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False,
                                        trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bench = load_domain_sft_benchmark(args.benchmark_dir)
    domain_of = {int(c["client_id"]): str(c["domain"]) for c in bench["clients"]}
    client_ids, loaders = create_domain_client_dataloaders(bench["train"], tok, args)
    if args.max_clients and len(client_ids) > args.max_clients:
        client_ids = client_ids[:args.max_clients]
        loaders = loaders[:args.max_clients]

    # per-domain client count check
    dom_count = defaultdict(int)
    for cid in client_ids:
        dom_count[domain_of.get(cid, "?")] += 1
    if max(dom_count.values()) < 2:
        print("[warn] 每域客户端数 < 2，D1/D3 域内诊断无意义。请用 14c/21c/35c。", flush=True)

    print(f"[diag] model={args.model} clients={len(client_ids)} "
          f"target_modules={args.target_modules} max_steps={args.max_steps} "
          f"anchor_lambda={args.anchor_lambda}", flush=True)

    model = create_peft_causal_lm_model(args)
    init_fedplora_adapters(model)
    A0 = _snapshot_A(model)

    # choose which lora_A keys (layers) to analyze
    keys = list(A0.keys())
    if args.max_layers and len(keys) > args.max_layers:
        rng = np.random.default_rng(args.seed)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), args.max_layers, replace=False))]
    print(f"[diag] analyzing {len(keys)} lora_A layers", flush=True)

    # --- capture trained A_i per client ---
    A_by_client = {}
    for n, cid in enumerate(client_ids):
        _reset_adapters(model, A0)
        _train_one_client(model, loaders[n], args, A0, device)
        sd = model.state_dict()
        A_by_client[cid] = {k: sd[k].detach().cpu().float().clone() for k in keys}
        print(f"[diag] captured client {cid} ({domain_of.get(cid,'?')}) [{n+1}/{len(client_ids)}]",
              flush=True)

    # precompute row-space bases and A0 bases
    Q_by_client = {cid: {k: _rowspace_basis(A_by_client[cid][k]) for k in keys} for cid in client_ids}
    A0_basis = {k: _rowspace_basis(A0[k]) for k in keys}

    # ---------------- D2: update magnitude ----------------
    d2 = []
    for cid in client_ids:
        for k in keys:
            rel = float((A_by_client[cid][k] - A0[k]).norm() / A0[k].norm().clamp_min(1e-12))
            d2.append(rel)
    d2 = np.array(d2)

    # ---------------- D1: gauge spread (same-domain pairs) ----------------
    by_dom = defaultdict(list)
    for cid in client_ids:
        by_dom[domain_of.get(cid, "?")].append(cid)

    d1_elem, d1_sub = [], []     # same-domain
    for dom, cids in by_dom.items():
        for a in range(len(cids)):
            for b in range(a + 1, len(cids)):
                ci, cj = cids[a], cids[b]
                for k in keys:
                    d1_elem.append(_rel_elem_diff(A_by_client[ci][k], A_by_client[cj][k]))
                    ang = _principal_angles_deg(Q_by_client[ci][k], Q_by_client[cj][k])
                    d1_sub.append(float(ang.mean().item()))
    d1_elem = np.array(d1_elem) if d1_elem else np.array([np.nan])
    d1_sub = np.array(d1_sub) if d1_sub else np.array([np.nan])

    # ---------------- D3: intra vs inter vs null principal angles ----------------
    intra, inter = [], []
    cids_all = client_ids
    for i in range(len(cids_all)):
        for j in range(i + 1, len(cids_all)):
            ci, cj = cids_all[i], cids_all[j]
            same = domain_of.get(ci) == domain_of.get(cj)
            angs = []
            for k in keys:
                a = _principal_angles_deg(Q_by_client[ci][k], Q_by_client[cj][k])
                angs.append(float(a.mean().item()))
            (intra if same else inter).append(float(np.mean(angs)))
    intra = np.array(intra) if intra else np.array([np.nan])
    inter = np.array(inter) if inter else np.array([np.nan])

    # random null: two random (r,d) row spaces
    r = A0[keys[0]].shape[0]
    d = A0[keys[0]].shape[1]
    rng = torch.Generator().manual_seed(args.seed)
    null = []
    for _ in range(int(args.n_null)):
        Q1 = _rowspace_basis(torch.randn(r, d, generator=rng))
        Q2 = _rowspace_basis(torch.randn(r, d, generator=rng))
        null.append(float(_principal_angles_deg(Q1, Q2).mean().item()))
    null = np.array(null)

    def stat(x):
        return {"mean": float(np.nanmean(x)), "median": float(np.nanmedian(x)),
                "p10": float(np.nanpercentile(x, 10)), "p90": float(np.nanpercentile(x, 90)),
                "n": int(np.sum(~np.isnan(x)))}

    report = {
        "config": {k: getattr(args, k) for k in
                   ["model", "benchmark_dir", "lora_r", "target_modules",
                    "max_steps", "local_epochs", "anchor_lambda", "max_clients", "max_layers"]},
        "num_clients": len(client_ids),
        "per_domain_clients": dict(dom_count),
        "D2_update_magnitude_rel": stat(d2),
        "D1_same_domain_elementwise_reldiff": stat(d1_elem),
        "D1_same_domain_subspace_meanangle_deg": stat(d1_sub),
        "D3_intra_domain_meanangle_deg": stat(intra),
        "D3_inter_domain_meanangle_deg": stat(inter),
        "D3_random_null_meanangle_deg": stat(null),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---------------- 判读 ----------------
    print("\n================ DCR 前提诊断结果 ================", flush=True)
    print(f"[D2] 更新幅度 ‖A_i-A0‖/‖A0‖: median={report['D2_update_magnitude_rel']['median']:.4f} "
          f"(p10={report['D2_update_magnitude_rel']['p10']:.4f}, p90={report['D2_update_magnitude_rel']['p90']:.4f})")
    print(f"[D1] 同域 逐元素差异:   median={report['D1_same_domain_elementwise_reldiff']['median']:.4f}")
    print(f"[D1] 同域 子空间主夹角: median={report['D1_same_domain_subspace_meanangle_deg']['median']:.2f}°")
    print(f"[D3] 域内主夹角 median={report['D3_intra_domain_meanangle_deg']['median']:.2f}°  "
          f"域间={report['D3_inter_domain_meanangle_deg']['median']:.2f}°  "
          f"随机null={report['D3_random_null_meanangle_deg']['median']:.2f}°")
    print("-------------------------------------------------")
    # 自动给出倾向性判读（阈值仅供参考）
    upd = report["D2_update_magnitude_rel"]["median"]
    intra_m = report["D3_intra_domain_meanangle_deg"]["median"]
    inter_m = report["D3_inter_domain_meanangle_deg"]["median"]
    null_m = report["D3_random_null_meanangle_deg"]["median"]
    if upd < 0.03:
        print("[判读] D2 更新幅度很小(<3%) -> 风险2偏成立：可能是 1-epoch regime 问题，建议加 epoch 再评估。")
    sep = (null_m - inter_m)  # 域间比随机小多少
    gap = (inter_m - intra_m)  # 域间比域内大多少
    if gap > 3 and (null_m - inter_m) > 3:
        print(f"[判读] D3 域间({inter_m:.1f}°) 显著 > 域内({intra_m:.1f}°) 且 < 随机({null_m:.1f}°) "
              "-> 存在可利用的跨域几何信号，DCR 机制有据。")
    else:
        print(f"[判读] D3 域内/域间/随机区分不明显(gap={gap:.1f}°) -> 跨域几何信号弱，DCR 机制存疑。")
    elem_m = report["D1_same_domain_elementwise_reldiff"]["median"]
    sub_m = report["D1_same_domain_subspace_meanangle_deg"]["median"]
    print(f"[判读] D1 同域 逐元素={elem_m:.3f} vs 子空间夹角={sub_m:.1f}° —— "
          "若逐元素明显大而子空间夹角小，说明 gauge 在搅局(DCR 有理)；若都很小则 A_i 近乎相同(动机弱)。")
    print(f"\n[diag] 详细结果已保存: {args.out}", flush=True)

    if args.save_figs:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].hist(d2, bins=40); ax[0].set_title("D2 update magnitude ‖A_i-A0‖/‖A0‖")
            ax[0].set_xlabel("relative update")
            ax[1].hist(intra, bins=30, alpha=0.6, label="intra-domain")
            ax[1].hist(inter, bins=30, alpha=0.6, label="inter-domain")
            ax[1].hist(null, bins=30, alpha=0.6, label="random null")
            ax[1].set_title("D3 mean principal angle (deg)"); ax[1].legend()
            figpath = os.path.splitext(args.out)[0] + ".png"
            plt.tight_layout(); plt.savefig(figpath, dpi=120)
            print(f"[diag] 图已保存: {figpath}", flush=True)
        except Exception as e:
            print(f"[diag][warn] 画图失败: {e!r}", flush=True)


if __name__ == "__main__":
    main()

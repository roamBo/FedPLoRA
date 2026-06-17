"""
diag_subspace_AB.py — 对 A(行空间) 与 B(列空间) 同时做"域内 vs 域间 vs 随机"主夹角诊断。

目的：定量回答"跨域域特异信号在 A 还是 B"。
  对每个客户端捕获训练后的 A_i (r×d) 和 B_i (d_out×r)。
  A: 行空间主夹角；B: 列空间主夹角。
  报告域内/域间/随机 null，以及域信号占比 ratio=(inter-intra)/(null-intra)。
  若 B 的 ratio ≫ A 的 ratio -> 域特异信号在 B（共享 B 会抹掉域信息）。

用法（仓库根目录）：
  python scripts/Analysis/diag_subspace_AB.py \
    --model /path/to/SmolLM2-135M \
    --benchmark_dir data/domain_benchmark_35c/seed_42 \
    --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
    --max_seq_length 512 --max_steps 0 \
    --out artifacts_LW/diag_AB_smol_35c.json --save_figs
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

from utilities.data_utils import load_domain_sft_benchmark, create_domain_client_dataloaders  # noqa: E402
from utilities.models import create_peft_causal_lm_model, init_fedplora_adapters  # noqa: E402
from utilities.utils import is_lora_a_param_name, is_lora_b_param_name  # noqa: E402


def _resolve_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


def _snapshot(model, pred):
    return {k: v.detach().cpu().float().clone()
            for k, v in model.state_dict().items() if pred(k)}


def _reset_adapters(model, A0):
    sd = model.state_dict()
    for k, v in sd.items():
        if is_lora_a_param_name(k) and k in A0:
            sd[k] = A0[k].to(device=v.device, dtype=v.dtype)
        elif is_lora_b_param_name(k):
            sd[k] = torch.zeros_like(v)
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


def _orthobasis_rows(M):
    """Return orthonormal-row basis (k×n) spanning the ROW space of M (m×n)."""
    M = M.float()
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)
    keep = max(int((S > S.max().clamp_min(1e-12) * 1e-6).sum().item()) if S.numel() else 1, 1)
    return Vh[:keep]


def _colspace_basis_rows(B):
    """Column space of B (d_out×r) as orthonormal ROWS (k×d_out) = U^T."""
    B = B.float()
    U, S, _ = torch.linalg.svd(B, full_matrices=False)   # U: (d_out, r)
    keep = max(int((S > S.max().clamp_min(1e-12) * 1e-6).sum().item()) if S.numel() else 1, 1)
    return U[:, :keep].transpose(0, 1)                   # (k, d_out)


def _pa_mean_deg(Q1, Q2):
    sv = torch.linalg.svdvals(Q1 @ Q2.transpose(0, 1)).clamp(0.0, 1.0)
    return float((torch.arccos(sv) * (180.0 / math.pi)).mean().item())


def _intra_inter_null(Q_by_client, keys, client_ids, domain_of, shape, seed, n_null):
    intra, inter = [], []
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            ci, cj = client_ids[i], client_ids[j]
            same = domain_of.get(ci) == domain_of.get(cj)
            ang = float(np.mean([_pa_mean_deg(Q_by_client[ci][k], Q_by_client[cj][k]) for k in keys]))
            (intra if same else inter).append(ang)
    r, n = shape
    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_null):
        Q1 = _orthobasis_rows(torch.randn(r, n, generator=g))
        Q2 = _orthobasis_rows(torch.randn(r, n, generator=g))
        null.append(_pa_mean_deg(Q1, Q2))
    return np.array(intra or [np.nan]), np.array(inter or [np.nan]), np.array(null)


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
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_clients", type=int, default=0)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--max_train_samples_per_client", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_null", type=int, default=200)
    ap.add_argument("--out", type=str, default="diag_AB.json")
    ap.add_argument("--save_figs", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
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

    dom_count = defaultdict(int)
    for cid in client_ids:
        dom_count[domain_of.get(cid, "?")] += 1
    if max(dom_count.values()) < 2:
        print("[warn] 每域客户端数<2，域内诊断无意义，请用 14c/21c/35c。", flush=True)

    model = create_peft_causal_lm_model(args)
    init_fedplora_adapters(model)
    A0 = _snapshot(model, is_lora_a_param_name)
    a_keys = list(A0.keys())
    b_keys = [k.replace("lora_A", "lora_B") for k in a_keys]
    if args.max_layers and len(a_keys) > args.max_layers:
        rng = np.random.default_rng(args.seed)
        sel = sorted(rng.choice(len(a_keys), args.max_layers, replace=False))
        a_keys = [a_keys[i] for i in sel]; b_keys = [b_keys[i] for i in sel]
    print(f"[diag] clients={len(client_ids)} layers={len(a_keys)} "
          f"target_modules={args.target_modules} max_steps={args.max_steps}", flush=True)

    QA, QB = {}, {}
    for n, cid in enumerate(client_ids):
        _reset_adapters(model, A0)
        _train_one_client(model, loaders[n], args, device)
        sd = model.state_dict()
        QA[cid] = {k: _orthobasis_rows(sd[k].detach().cpu().float()) for k in a_keys}
        QB[cid] = {bk: _colspace_basis_rows(sd[bk].detach().cpu().float())
                   for bk in b_keys if bk in sd}
        print(f"[diag] captured {cid} ({domain_of.get(cid,'?')}) [{n+1}/{len(client_ids)}]", flush=True)

    rA, dA = A0[a_keys[0]].shape
    dOut = model.state_dict()[b_keys[0]].shape[0]
    a_intra, a_inter, a_null = _intra_inter_null(QA, a_keys, client_ids, domain_of, (rA, dA), args.seed, args.n_null)
    b_intra, b_inter, b_null = _intra_inter_null(QB, b_keys, client_ids, domain_of, (rA, dOut), args.seed + 1, args.n_null)

    def ratio(intra, inter, null):
        denom = (np.nanmedian(null) - np.nanmedian(intra))
        return float((np.nanmedian(inter) - np.nanmedian(intra)) / denom) if abs(denom) > 1e-9 else float("nan")

    rep = {
        "config": {k: getattr(args, k) for k in ["model", "benchmark_dir", "target_modules", "max_steps", "local_epochs"]},
        "A_rowspace": {"intra": float(np.nanmedian(a_intra)), "inter": float(np.nanmedian(a_inter)),
                       "null": float(np.nanmedian(a_null)), "domain_signal_ratio": ratio(a_intra, a_inter, a_null)},
        "B_colspace": {"intra": float(np.nanmedian(b_intra)), "inter": float(np.nanmedian(b_inter)),
                       "null": float(np.nanmedian(b_null)), "domain_signal_ratio": ratio(b_intra, b_inter, b_null)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("\n========= A vs B 域信号诊断 (主夹角中位数，度) =========")
    print(f"{'':10s}{'域内':>8s}{'域间':>8s}{'随机null':>10s}{'域信号占比':>12s}")
    for tag, blk in (("A 行空间", rep["A_rowspace"]), ("B 列空间", rep["B_colspace"])):
        print(f"{tag:10s}{blk['intra']:8.2f}{blk['inter']:8.2f}{blk['null']:10.2f}{blk['domain_signal_ratio']*100:11.1f}%")
    print("--------------------------------------------------------")
    rA_ = rep["A_rowspace"]["domain_signal_ratio"]; rB_ = rep["B_colspace"]["domain_signal_ratio"]
    if rB_ > 2 * rA_ and rB_ > 0.15:
        print(f"[判读] B 域信号占比({rB_*100:.0f}%) ≫ A({rA_*100:.0f}%) -> 域特异信号主要在 B。")
        print("       含义：朴素平均 B 会抹掉域信息；'共享 B' 需谨慎设计，不能直接 FedAvg。")
        print("       但这也说明：方法重心应转向 B 侧（你的真实卖点：跨域冲突在 B）。")
    elif rB_ <= rA_:
        print(f"[判读] B 域信号占比({rB_*100:.0f}%) ≤ A({rA_*100:.0f}%) -> 域信号不在 B；需重新定位冲突来源。")
    else:
        print(f"[判读] A({rA_*100:.0f}%) 与 B({rB_*100:.0f}%) 都偏小或接近 -> 跨域域特异结构整体弱。")
    print(f"\n[diag] 已保存: {args.out}", flush=True)

    if args.save_figs:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for a, intra, inter, null, t in ((ax[0], a_intra, a_inter, a_null, "A row-space"),
                                             (ax[1], b_intra, b_inter, b_null, "B col-space")):
                a.hist(intra, bins=25, alpha=0.6, label="intra")
                a.hist(inter, bins=25, alpha=0.6, label="inter")
                a.hist(null, bins=25, alpha=0.6, label="null")
                a.set_title(f"{t} mean principal angle (deg)"); a.legend()
            figpath = os.path.splitext(args.out)[0] + ".png"
            plt.tight_layout(); plt.savefig(figpath, dpi=120)
            print(f"[diag] 图: {figpath}", flush=True)
        except Exception as e:
            print(f"[diag][warn] 画图失败: {e!r}", flush=True)


if __name__ == "__main__":
    main()

"""
eval_personalized.py — per-domain 个性化评测 + v7 方案对比。

与现有 domain-macro 的区别：**每个客户端只在自己域的测试集上评测**（个性化目标），
而非在全部 7 域上测再平均（通才目标）。后者奖励通才、罚专才，掩盖跨域个性化方法的价值。

捕获每客户端训练后的 (A_i,B_i) 后，对比 4 套聚合方案，全部用 per-domain 评测：
  local  : (A_i, B_i)                          各自本地，个性化上界
  fedsa  : (A_global, B_i)                      共享A，本地B（FedSA/FedPLoRA 范式）
  global : (A_global, B_global)                全局平均 A+B（FedAvg 通才）
  v7     : (A_global, B_domain)                v7：全局A + 按域池化B（跨域隔离）
  v11c   : (A_global, μB_global+(1-μ)B_domain) global-routed B mixing
  select : 按每客户端 validation loss 从候选菜单选择部署状态

输出：每方案的 per-domain 个性化 macro_acc / worst_acc / 各域 acc，便于判定
"域特异 B" 在个性化目标下是否反超通才。

用法（仓库根目录）：
  python scripts/Analysis/eval_personalized.py \
    --model /path/to/SmolLM2-135M \
    --benchmark_dir data/domain_benchmark_35c/seed_42 \
    --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
    --max_seq_length 512 --max_steps 0 --eval_max_batches 30 \
    --v7_b_mode mean --out artifacts_LW/eval_personalized_smol_35c.json
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities.data_utils import (  # noqa: E402
    load_domain_sft_benchmark, create_domain_client_dataloaders,
    create_domain_eval_dataloader, group_rows_by_domain, group_rows_by_client,
    DomainSFTDataset,
)
from utilities.benchmark_fingerprint import compute_benchmark_fingerprint  # noqa: E402
from utilities.models import create_peft_causal_lm_model, init_fedplora_adapters  # noqa: E402
from utilities.utils import is_lora_a_param_name, is_lora_b_param_name  # noqa: E402


# ---- v7 聚合（内联，使本脚本单文件可跑、无需 methods/v7 包）----
def aggregate_global_A(A_by_client, a_keys):
    """A_global = 跨所有客户端按层平均（A 域通用）。"""
    cids = list(A_by_client.keys())
    return {k: torch.stack([A_by_client[c][k].float() for c in cids], 0).mean(0) for k in a_keys}


def _consolidate_B(B_list, mode):
    """同域若干 B_i (d_out×r) 合成域级 B_d (d_out×r)。"""
    r = B_list[0].shape[1]
    if mode == "rep":
        return B_list[0].float().clone()
    if mode == "mean":
        return torch.stack([B.float() for B in B_list], 0).mean(0)
    if mode == "svd":
        stacked = torch.cat([B.float() for B in B_list], dim=1)         # (d_out, n*r)
        U, S, _ = torch.linalg.svd(stacked, full_matrices=False)
        r_eff = min(r, S.numel())
        Bd = U[:, :r_eff] * S[:r_eff].unsqueeze(0)
        target = torch.stack([B.float() for B in B_list], 0).norm(dim=(1, 2)).mean()
        Bd = Bd * (target / Bd.norm().clamp_min(1e-12))
        if r_eff < r:
            Bd = torch.cat([Bd, torch.zeros(Bd.shape[0], r - r_eff)], dim=1)
        return Bd
    raise ValueError(f"unknown v7_b_mode: {mode}")


def aggregate_per_domain_B(B_by_client, domain_of, b_keys, mode="mean"):
    """返回 {domain: {b_key: B_d}}，每域一套（跨域隔离）。"""
    by_dom = defaultdict(list)
    for cid in B_by_client:
        by_dom[domain_of.get(cid, "?")].append(cid)
    out = {}
    for dom, cids in by_dom.items():
        out[dom] = {}
        for bk in b_keys:
            B_list = [B_by_client[c][bk] for c in cids if bk in B_by_client[c]]
            if B_list:
                out[dom][bk] = _consolidate_B(B_list, mode)
    return out


def build_v7_client_state(A_global, B_per_domain, domain):
    """组装某域客户端下发权重：全局 A + 本域 B。"""
    state = {k: v.clone() for k, v in A_global.items()}
    state.update({k: v.clone() for k, v in B_per_domain.get(domain, {}).items()})
    return state


def _resolve_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


def _snapshot(model, pred):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if pred(k)}


def _reset_adapters(model, A0):
    sd = model.state_dict()
    for k, v in sd.items():
        if is_lora_a_param_name(k) and k in A0:
            sd[k] = A0[k].to(device=v.device, dtype=v.dtype)
        elif is_lora_b_param_name(k):
            sd[k] = torch.zeros_like(v)
    model.load_state_dict(sd)


def _install(model, state):
    sd = model.state_dict()
    for k, v in state.items():
        if k in sd:
            sd[k] = v.to(device=sd[k].device, dtype=sd[k].dtype)
    model.load_state_dict(sd)


def _train_one_client(model, loader, args, device):
    model.to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
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
            loss.backward(); optim.step(); optim.zero_grad(); step += 1


def _parse_int_list(text):
    out = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _resolve_held_out_clients(spec, client_ids, domain_of, *, policy="first", offset=0, seed=42):
    spec = str(spec or "").strip()
    if not spec:
        return []
    if spec.lower() in {"auto", "auto_one_per_domain", "one_per_domain"}:
        by_dom = defaultdict(list)
        for cid in client_ids:
            by_dom[domain_of.get(int(cid), "?")].append(int(cid))
        out = []
        policy = str(policy or "first").strip().lower()
        offset = int(offset or 0)
        rng = np.random.default_rng(int(seed))
        for _dom, cids in sorted(by_dom.items()):
            cids = sorted(int(x) for x in cids)
            if not cids:
                continue
            if policy in {"first", "offset", "offset_mod"}:
                out.append(cids[offset % len(cids)])
            elif policy in {"last"}:
                out.append(cids[-1])
            elif policy in {"random", "random_one_per_domain"}:
                out.append(int(rng.choice(cids)))
            else:
                raise ValueError(
                    f"unknown held_out_policy={policy!r}; "
                    "use first, offset, last, or random"
                )
        return out
    return sorted(set(_parse_int_list(spec)))


def _make_train_loader_for_rows(rows, tokenizer, args, *, cap=0, seed=42, client_id=0):
    rows = list(rows or [])
    if cap and cap > 0 and len(rows) > cap:
        rng = np.random.default_rng(int(seed) + int(client_id) * 9973 + int(cap))
        idx = sorted(rng.choice(len(rows), size=int(cap), replace=False).tolist())
        rows = [rows[i] for i in idx]
    ds = DomainSFTDataset(rows, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True)


def _state_nbytes(state, pred=None):
    """Exact tensor payload size (serialization/container overhead excluded)."""
    total = 0
    for key, value in (state or {}).items():
        if torch.is_tensor(value) and (pred is None or pred(key)):
            total += int(value.numel()) * int(value.element_size())
    return int(total)


def _flat_named_state(state, keys):
    values = [state[k].float().reshape(-1) for k in keys if k in state]
    return torch.cat(values) if values else torch.empty(0)


def _flat_cosine_score(query, candidate, keys):
    q = _flat_named_state(query, keys)
    c = _flat_named_state(candidate, keys)
    if q.numel() == 0 or q.numel() != c.numel():
        return None
    return float(F.cosine_similarity(q, c, dim=0).item())


def _relative_l2_score(query, candidate, keys):
    q = _flat_named_state(query, keys)
    c = _flat_named_state(candidate, keys)
    if q.numel() == 0 or q.numel() != c.numel():
        return None
    denom = float(q.norm().item() + c.norm().item())
    return -float((q - c).norm().item()) / max(denom, 1e-12)


def _subspace_score(query, candidate, keys):
    """Mean canonical correlation between per-layer B column spaces."""
    def _basis(value):
        u, singular, _ = torch.linalg.svd(value.float(), full_matrices=False)
        if singular.numel() == 0 or float(singular[0].item()) <= 0.0:
            return None
        tol = torch.finfo(singular.dtype).eps * max(value.shape) * singular[0]
        rank = int((singular > tol).sum().item())
        return u[:, :rank] if rank > 0 else None

    layer_scores = []
    for key in keys:
        if key not in query or key not in candidate:
            continue
        q = query[key].float()
        c = candidate[key].float()
        if q.ndim != 2 or c.ndim != 2 or q.shape[0] != c.shape[0]:
            continue
        q_basis = _basis(q)
        c_basis = _basis(c)
        if q_basis is None or c_basis is None:
            continue
        singular = torch.linalg.svdvals(q_basis.T @ c_basis).clamp(0.0, 1.0)
        if singular.numel():
            layer_scores.append(float(singular.mean().item()))
    return float(np.mean(layer_scores)) if layer_scores else None


def _delta_w_cosine_score(query, candidate, a_keys, b_keys, fallback_a):
    """Cosine between BA updates without materializing the dense matrices."""
    inner = qnorm2 = cnorm2 = 0.0
    for ak, bk in zip(a_keys, b_keys):
        if bk not in query or bk not in candidate:
            continue
        qa = query.get(ak, fallback_a.get(ak))
        ca = candidate.get(ak, fallback_a.get(ak))
        if qa is None or ca is None:
            continue
        qb, cb = query[bk].float(), candidate[bk].float()
        qa, ca = qa.float(), ca.float()
        if qb.ndim != 2 or cb.ndim != 2 or qa.ndim != 2 or ca.ndim != 2:
            continue
        if qb.shape[0] != cb.shape[0] or qa.shape[1] != ca.shape[1]:
            continue
        inner += float(torch.sum(qa * ((qb.T @ cb) @ ca)).item())
        qnorm2 += float(torch.sum(qa * ((qb.T @ qb) @ qa)).item())
        cnorm2 += float(torch.sum(ca * ((cb.T @ cb) @ ca)).item())
    denom = math.sqrt(max(qnorm2, 0.0) * max(cnorm2, 0.0))
    return inner / max(denom, 1e-12) if denom > 0 else None


def _sync_device(device):
    if getattr(device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _eval(model, loader, device, max_batches):
    """NaN-safe per-domain eval: 手动 fp32 交叉熵，跳过无有效标签的 batch。"""
    model.eval()
    tot_loss, n_tok, correct, steps = 0.0, 0, 0, 0
    for batch in loader:
        if max_batches and steps >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float()
        labels = batch["labels"]
        sl = logits[..., :-1, :].reshape(-1, logits.size(-1))
        sy = labels[..., 1:].reshape(-1)
        mask = sy.ne(-100)
        if not bool(mask.any()):
            continue
        loss = F.cross_entropy(sl[mask], sy[mask], reduction="sum")
        if torch.isfinite(loss):
            tot_loss += float(loss.item())
            n_tok += int(mask.sum().item())
            correct += int((sl[mask].argmax(-1) == sy[mask]).sum().item())
        steps += 1
    if n_tok == 0:
        return float("nan"), float("nan")
    return tot_loss / n_tok, correct / n_tok


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
    ap.add_argument("--eval_max_batches", type=int, default=30)
    ap.add_argument("--v7_b_mode", type=str, default="mean", choices=["mean", "rep", "svd"])
    ap.add_argument("--v11c_mu", type=float, default=0.4,
                    help="v11c/v12-style global-routed B mixing weight for eval-only schemes.")
    ap.add_argument("--schemes", type=str, default="local,fedsa,global,v7")
    ap.add_argument("--select_candidates", type=str, default="local,v7,global,v11c",
                    help="Comma-separated candidate schemes for select; evaluated on each client's val split.")
    ap.add_argument("--cold_start", action="store_true",
                    help="额外评测冷启动：base(B=0 无适配下界) + coldstart(同域留一池化B，模拟新客户端零样本)")
    ap.add_argument("--held_out_clients", type=str, default="",
                    help="Strict held-out protocol: comma-separated client ids, or auto_one_per_domain. "
                    "Held-out clients are excluded from A/B aggregation and evaluated after training the remaining clients.")
    ap.add_argument("--held_out_policy", type=str, default="first",
                    choices=["first", "offset", "offset_mod", "last", "random", "random_one_per_domain"],
                    help="Selection policy used when --held_out_clients=auto_one_per_domain.")
    ap.add_argument("--held_out_offset", type=int, default=0,
                    help="Per-domain sorted-client offset for --held_out_policy=offset/first. "
                    "offset=1 selects the second client in each domain.")
    ap.add_argument("--held_out_eval_all", action="store_true",
                    help="With --held_out_clients, evaluate all clients instead of only held-out clients.")
    ap.add_argument("--few_shot_caps", type=str, default="5,10",
                    help="Comma-separated caps for held-out local few-shot upper-bound schemes, e.g. 5,10.")
    ap.add_argument("--held_out_route_probe_samples", type=int, default=10,
                    help="Few-shot samples used only to infer a held-out client's B-geometry route for coldstart_geom. 0 disables.")
    ap.add_argument(
        "--held_out_route_metrics",
        type=str,
        default="",
        help="Optional comma-separated router audit. Supported: flat_b_cosine,subspace,"
        "relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle. "
        "Empty preserves the legacy coldstart_geom-only behavior.",
    )
    ap.add_argument(
        "--onboarding_accounting",
        action="store_true",
        help="Record probe training/router wall time and exact tensor payload bytes.",
    )
    ap.add_argument("--eval_on_local", action="store_true",
                    help="每客户端用自己的 test_local（不同子分布）评测，而非共享 test_domain；"
                    "配合内容型非IID(--subtopic kmeans) 做 bulletproof 冷启动验证")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="eval_personalized.json")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False,
                                        trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bench = load_domain_sft_benchmark(args.benchmark_dir)
    benchmark_fingerprint = compute_benchmark_fingerprint(args.benchmark_dir, bench)
    domain_of = {int(c["client_id"]): str(c["domain"]) for c in bench["clients"]}
    client_ids, loaders = create_domain_client_dataloaders(bench["train"], tok, args)
    if args.max_clients and len(client_ids) > args.max_clients:
        client_ids, loaders = client_ids[:args.max_clients], loaders[:args.max_clients]
    train_rows_by_client = group_rows_by_client(bench["train"])
    test_local_rows_by_client = group_rows_by_client(bench.get("test_local", []) or [])
    val_rows_by_client = group_rows_by_client(bench.get("val", []) or [])
    held_out_clients = _resolve_held_out_clients(
        args.held_out_clients,
        client_ids,
        domain_of,
        policy=getattr(args, "held_out_policy", "first"),
        offset=int(getattr(args, "held_out_offset", 0) or 0),
        seed=int(args.seed),
    )
    held_out_set = set(int(x) for x in held_out_clients)
    if held_out_set:
        unknown = sorted(held_out_set - {int(x) for x in client_ids})
        if unknown:
            raise ValueError(f"held_out_clients not present in benchmark: {unknown}")
        print(
            f"[heldout] strict protocol ON held_out_clients={held_out_clients} "
            f"domains={{"
            + ", ".join(f"{cid}:{domain_of.get(cid, '?')}" for cid in held_out_clients)
            + "}}",
            flush=True,
        )
    train_pairs = [
        (int(cid), loader)
        for cid, loader in zip(client_ids, loaders)
        if int(cid) not in held_out_set
    ]
    eval_client_ids = (
        [int(cid) for cid in client_ids]
        if (not held_out_set or bool(getattr(args, "held_out_eval_all", False)))
        else list(held_out_clients)
    )

    test_by_dom = group_rows_by_domain(bench["test_domain"])
    eval_loaders = {d: create_domain_eval_dataloader(rows, tok, args) for d, rows in test_by_dom.items()}

    # 可选：每客户端用自己的 test_local（不同子分布），bulletproof 冷启动/onboarding 验证。
    local_eval_loaders = {}
    if bool(getattr(args, "eval_on_local", False)):
        for cid, rows in test_local_rows_by_client.items():
            if rows:
                local_eval_loaders[int(cid)] = create_domain_eval_dataloader(rows, tok, args)
        n_missing = sum(1 for c in client_ids if int(c) not in local_eval_loaders)
        print(f"[eval] eval_on_local=ON: {len(local_eval_loaders)} 个客户端有本地测试集"
              f"（{n_missing} 个缺失将回退域测试）", flush=True)

    val_eval_loaders = {}
    for cid, rows in val_rows_by_client.items():
        if rows:
            val_eval_loaders[int(cid)] = create_domain_eval_dataloader(rows, tok, args)

    model = create_peft_causal_lm_model(args)
    init_fedplora_adapters(model)
    A0 = _snapshot(model, is_lora_a_param_name)
    a_keys = list(A0.keys())
    b_keys = [k.replace("lora_A", "lora_B") for k in a_keys]

    # --- capture per-client A_i, B_i ---
    # In strict held-out mode, held-out clients are excluded here.  Their A/B
    # never contribute to A_global, B_global, or domain B pools.
    A_by, B_by = {}, {}
    for n, (cid, loader) in enumerate(train_pairs):
        _reset_adapters(model, A0)
        _train_one_client(model, loader, args, device)
        sd = model.state_dict()
        A_by[cid] = {k: sd[k].detach().cpu().clone() for k in a_keys}
        B_by[cid] = {bk: sd[bk].detach().cpu().clone() for bk in b_keys if bk in sd}
        print(f"[cap] {cid} ({domain_of.get(cid,'?')}) [{n+1}/{len(train_pairs)}]", flush=True)

    # --- precompute aggregates ---
    A_global = aggregate_global_A(A_by, a_keys)
    train_client_ids = [int(cid) for cid, _loader in train_pairs]
    B_global = {bk: torch.stack([B_by[c][bk].float() for c in train_client_ids if bk in B_by[c]], 0).mean(0)
                for bk in b_keys}
    B_per_domain = aggregate_per_domain_B(B_by, domain_of, b_keys, mode=args.v7_b_mode)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    few_shot_caps = [x for x in _parse_int_list(getattr(args, "few_shot_caps", "")) if x > 0]
    if held_out_set and args.schemes == "local,fedsa,global,v7":
        # Safer strict-heldout default: avoid reporting local/fedsa as if the
        # held-out client had joined training.  local_fewshot* is added below.
        schemes = ["base", "global", "coldstart", "coldstart_geom", "v11c_coldstart"]
    if held_out_set:
        for cap in few_shot_caps:
            name = f"local_fewshot{cap}"
            if name not in schemes:
                schemes.append(name)
    if bool(getattr(args, "cold_start", False)):
        # 冷启动：模拟"新客户端无本地数据"。base=无适配(B=0) 下界；
        # coldstart=用同域其它客户端池化的 B（留一法），这是新机构能零样本拿到的。
        for s in ("base", "coldstart", "v11c_coldstart"):
            if s not in schemes:
                schemes.append(s)
    if held_out_set and int(getattr(args, "held_out_route_probe_samples", 0) or 0) > 0:
        if "coldstart_geom" not in schemes:
            schemes.append("coldstart_geom")

    def _domain_B_loo(domain, exclude_cid):
        """同域 B 池化，排除 exclude_cid（留一法，模拟新客户端）。"""
        # Strict held-out clients are intentionally absent from B_by.  Pool only
        # over clients that actually participated in training; otherwise
        # v11c_coldstart/select_without_local can touch a held-out cid and raise
        # KeyError before writing the evaluation JSON.
        cids = [c for c in train_client_ids if domain_of.get(c) == domain and c != exclude_cid]
        out = {}
        for bk in b_keys:
            Bs = []
            for c in cids:
                state = B_by.get(c)
                if state is not None and bk in state:
                    Bs.append(state[bk])
            if Bs:
                out[bk] = _consolidate_B(Bs, args.v7_b_mode)
        return out

    def _global_B_loo(exclude_cid):
        """Global B pool excluding one client, used for true cold-start simulation."""
        cids = [c for c in train_client_ids if c != exclude_cid]
        out = {}
        for bk in b_keys:
            Bs = []
            for c in cids:
                state = B_by.get(c)
                if state is not None and bk in state:
                    Bs.append(state[bk])
            if Bs:
                out[bk] = torch.stack([B.float() for B in Bs], 0).mean(0)
        return out

    def _mix_b(global_b, routed_b):
        mu = min(1.0, max(0.0, float(getattr(args, "v11c_mu", 0.4) or 0.0)))
        out = {}
        for bk in b_keys:
            if bk in global_b and bk in routed_b:
                out[bk] = (
                    mu * global_b[bk].float() + (1.0 - mu) * routed_b[bk].float()
                ).detach().cpu()
        return out

    def _zero_B_state(cid=None):
        out = {}
        for bk in b_keys:
            ref = None
            if cid in B_by and bk in B_by[cid]:
                ref = B_by[cid][bk]
            elif bk in B_global:
                ref = B_global[bk]
            else:
                for st in B_by.values():
                    if bk in st:
                        ref = st[bk]
                        break
            if ref is not None:
                out[bk] = torch.zeros_like(ref)
        return out

    fewshot_states = defaultdict(dict)
    route_probe_states = {}
    fewshot_train_seconds = defaultdict(dict)
    route_probe_train_seconds = {}
    route_probe_sample_count = {}
    if held_out_set and few_shot_caps:
        for cap in few_shot_caps:
            for cid in held_out_clients:
                rows = train_rows_by_client.get(int(cid), [])
                if not rows:
                    continue
                _reset_adapters(model, A0)
                loader = _make_train_loader_for_rows(
                    rows,
                    tok,
                    args,
                    cap=cap,
                    seed=int(args.seed),
                    client_id=int(cid),
                )
                _sync_device(device)
                train_started = time.perf_counter()
                _train_one_client(model, loader, args, device)
                _sync_device(device)
                fewshot_train_seconds[int(cap)][int(cid)] = float(
                    time.perf_counter() - train_started
                )
                sd = model.state_dict()
                fewshot_states[int(cap)][int(cid)] = {
                    **{k: sd[k].detach().cpu().clone() for k in a_keys},
                    **{bk: sd[bk].detach().cpu().clone() for bk in b_keys if bk in sd},
                }
                print(f"[heldout-fewshot] cid={cid} cap={cap}", flush=True)

    route_probe_cap = int(getattr(args, "held_out_route_probe_samples", 0) or 0)
    if held_out_set and route_probe_cap > 0:
        for cid in held_out_clients:
            if cid in fewshot_states.get(route_probe_cap, {}):
                route_probe_states[int(cid)] = fewshot_states[route_probe_cap][int(cid)]
                route_probe_train_seconds[int(cid)] = float(
                    fewshot_train_seconds[route_probe_cap].get(int(cid), 0.0)
                )
                route_probe_sample_count[int(cid)] = int(
                    min(route_probe_cap, len(train_rows_by_client.get(int(cid), []) or []))
                )
                continue
            rows = train_rows_by_client.get(int(cid), [])
            if not rows:
                continue
            _reset_adapters(model, A0)
            loader = _make_train_loader_for_rows(
                rows,
                tok,
                args,
                cap=route_probe_cap,
                seed=int(args.seed),
                client_id=int(cid),
            )
            _sync_device(device)
            train_started = time.perf_counter()
            _train_one_client(model, loader, args, device)
            _sync_device(device)
            route_probe_train_seconds[int(cid)] = float(
                time.perf_counter() - train_started
            )
            route_probe_sample_count[int(cid)] = int(min(route_probe_cap, len(rows)))
            sd = model.state_dict()
            route_probe_states[int(cid)] = {
                **{k: sd[k].detach().cpu().clone() for k in a_keys if k in sd},
                **{bk: sd[bk].detach().cpu().clone() for bk in b_keys if bk in sd},
            }

    def _flat_b(state):
        vals = [state[bk].float().reshape(-1) for bk in b_keys if bk in state]
        return torch.cat(vals) if vals else torch.empty(0)

    geom_route_domain_by_client = {}
    geom_route_scores = {}
    geom_route_top2_by_client = {}
    geom_route_margin_by_client = {}
    geom_route_oracle_match_by_client = {}
    if held_out_set and route_probe_states:
        domain_vectors = {}
        for dom, st in B_per_domain.items():
            vec = _flat_b(st)
            if vec.numel() > 0:
                domain_vectors[dom] = vec
        for cid, st in route_probe_states.items():
            q = _flat_b(st)
            scores = []
            for dom, vec in domain_vectors.items():
                if q.numel() != vec.numel() or q.numel() == 0:
                    continue
                score = float(F.cosine_similarity(q, vec, dim=0).item())
                scores.append((score, dom))
            scores.sort(key=lambda x: x[0], reverse=True)
            if scores:
                best = scores[0]
                second = scores[1] if len(scores) > 1 else None
                geom_route_domain_by_client[int(cid)] = best[1]
                geom_route_scores[int(cid)] = best[0]
                geom_route_top2_by_client[int(cid)] = [
                    {"domain": dom, "score": float(score)}
                    for score, dom in scores[:2]
                ]
                geom_route_margin_by_client[int(cid)] = (
                    float(best[0] - second[0]) if second is not None else None
                )
                geom_route_oracle_match_by_client[int(cid)] = (
                    str(best[1]) == str(domain_of.get(int(cid), "?"))
                )
        print(
            f"[heldout-route] geom_route_domain_by_client={geom_route_domain_by_client}",
            flush=True,
        )

    supported_route_metrics = {
        "flat_b_cosine", "subspace", "relative_l2", "delta_w_cosine",
        "nearest_client_subspace", "largest_domain", "random", "oracle",
    }
    requested_route_metrics = [
        item.strip()
        for item in str(getattr(args, "held_out_route_metrics", "") or "").split(",")
        if item.strip()
    ]
    unknown_route_metrics = sorted(set(requested_route_metrics) - supported_route_metrics)
    if unknown_route_metrics:
        raise ValueError(
            f"unsupported held_out_route_metrics={unknown_route_metrics}; "
            f"supported={sorted(supported_route_metrics)}"
        )
    route_audits = {}

    def _route_metric_score(metric, query, candidate):
        if metric == "flat_b_cosine":
            return _flat_cosine_score(query, candidate, b_keys)
        if metric == "subspace" or metric == "nearest_client_subspace":
            return _subspace_score(query, candidate, b_keys)
        if metric == "relative_l2":
            return _relative_l2_score(query, candidate, b_keys)
        if metric == "delta_w_cosine":
            return _delta_w_cosine_score(
                query, candidate, a_keys, b_keys, A_global
            )
        return None

    for metric in requested_route_metrics:
        routed, score_by_client, top2_by_client = {}, {}, {}
        margin_by_client, match_by_client, seconds_by_client = {}, {}, {}
        for cid, query in sorted(route_probe_states.items()):
            route_started = time.perf_counter()
            if metric == "oracle":
                scores = [(1.0, str(domain_of.get(int(cid), "?")))]
            elif metric == "largest_domain":
                counts = defaultdict(int)
                for train_cid in train_client_ids:
                    counts[str(domain_of.get(int(train_cid), "?"))] += 1
                largest = sorted(counts, key=lambda dom: (-counts[dom], dom))[0] if counts else "?"
                scores = [(1.0 if dom == largest else 0.0, dom) for dom in sorted(counts)]
            elif metric == "random":
                domains = sorted(B_per_domain)
                rng = np.random.default_rng(
                    int(args.seed) * 1000003 + int(cid) * 9176 + 73
                )
                picked = str(rng.choice(domains)) if domains else "?"
                scores = [(1.0 if dom == picked else 0.0, dom) for dom in domains]
            elif metric == "nearest_client_subspace":
                domain_best = {}
                for train_cid, candidate in B_by.items():
                    score = _route_metric_score(metric, query, candidate)
                    dom = str(domain_of.get(int(train_cid), "?"))
                    if score is not None and (
                        dom not in domain_best or score > domain_best[dom]
                    ):
                        domain_best[dom] = float(score)
                scores = [(score, dom) for dom, score in domain_best.items()]
            else:
                scores = []
                for dom, b_state in B_per_domain.items():
                    candidate = {**A_global, **b_state}
                    score = _route_metric_score(metric, query, candidate)
                    if score is not None:
                        scores.append((float(score), str(dom)))
            scores.sort(key=lambda item: (-item[0], item[1]))
            seconds_by_client[int(cid)] = float(time.perf_counter() - route_started)
            if not scores:
                continue
            best = scores[0]
            second = scores[1] if len(scores) > 1 else None
            routed[int(cid)] = best[1]
            score_by_client[int(cid)] = float(best[0])
            top2_by_client[int(cid)] = [
                {"domain": dom, "score": float(score)} for score, dom in scores[:2]
            ]
            margin_by_client[int(cid)] = (
                float(best[0] - second[0]) if second is not None else None
            )
            match_by_client[int(cid)] = (
                str(best[1]) == str(domain_of.get(int(cid), "?"))
            )
        valid_margins = [float(v) for v in margin_by_client.values() if v is not None]
        route_audits[metric] = {
            "route_domain_by_client": routed,
            "score_by_client": score_by_client,
            "top2_by_client": top2_by_client,
            "margin_by_client": margin_by_client,
            "oracle_match_by_client": match_by_client,
            "route_compute_seconds_by_client": seconds_by_client,
            "summary": {
                "num_routed": int(len(routed)),
                "oracle_match_rate": (
                    float(np.mean(list(match_by_client.values())))
                    if match_by_client else None
                ),
                "mean_margin": float(np.mean(valid_margins)) if valid_margins else None,
                "mean_route_compute_seconds": (
                    float(np.mean(list(seconds_by_client.values())))
                    if seconds_by_client else None
                ),
            },
        }
        scheme_name = f"coldstart_route_{metric}"
        if routed and scheme_name not in schemes:
            schemes.append(scheme_name)
        print(
            f"[heldout-route] metric={metric} "
            f"match_rate={route_audits[metric]['summary']['oracle_match_rate']} "
            f"routes={routed}",
            flush=True,
        )

    protocol_tag = "personalized_eval"
    if held_out_set:
        eval_scope = "all" if bool(getattr(args, "held_out_eval_all", False)) else "heldout"
        objective = "localtest" if bool(getattr(args, "eval_on_local", False)) else "domaintest"
        protocol_tag = (
            f"strict_heldout:{objective}:policy={getattr(args, 'held_out_policy', 'first')}"
            f":offset={int(getattr(args, 'held_out_offset', 0) or 0)}"
            f":scope={eval_scope}:route_probe={int(getattr(args, 'held_out_route_probe_samples', 0) or 0)}"
        )
    geom_route_summary = {
        "num_routed": int(len(geom_route_domain_by_client)),
        "oracle_match_rate": (
            float(np.mean([bool(v) for v in geom_route_oracle_match_by_client.values()]))
            if geom_route_oracle_match_by_client else None
        ),
        "mean_margin": (
            float(np.mean([float(v) for v in geom_route_margin_by_client.values() if v is not None]))
            if any(v is not None for v in geom_route_margin_by_client.values()) else None
        ),
        "min_margin": (
            float(np.min([float(v) for v in geom_route_margin_by_client.values() if v is not None]))
            if any(v is not None for v in geom_route_margin_by_client.values()) else None
        ),
    }

    selected_choice_by_client = {}
    selected_choice_without_local_by_client = {}

    def client_state(scheme, cid):
        d = domain_of[cid]
        if scheme == "local":
            if cid not in A_by:
                if few_shot_caps:
                    cap = max(few_shot_caps)
                    if cid in fewshot_states.get(cap, {}):
                        return fewshot_states[cap][cid]
                return {**A_global, **_zero_B_state(cid)}
            return {**A_by[cid], **B_by[cid]}
        if scheme == "fedsa":
            if cid not in B_by:
                return {**A_global, **B_global}
            return {**A_global, **B_by[cid]}
        if scheme == "global":
            return {**A_global, **B_global}
        if scheme == "v7":
            return build_v7_client_state(A_global, B_per_domain, d)
        if scheme == "v11c":
            # v11c/v12-style state for eval-only analysis:
            # global A + explicit global/routed B mixture.
            return {**A_global, **_mix_b(B_global, B_per_domain.get(d, {}))}
        if scheme == "base":
            # 无 LoRA 适配：B=0 → ΔW=0，等价冻结基座（冷启动下界）
            return {**A_global, **_zero_B_state(cid)}
        if scheme == "coldstart":
            # 新客户端：全局 A + 同域其它客户端池化的 B（不含自己）
            return {**A_global, **_domain_B_loo(d, cid)}
        if scheme == "coldstart_geom":
            routed_domain = geom_route_domain_by_client.get(int(cid), d)
            return {**A_global, **B_per_domain.get(routed_domain, {})}
        if scheme.startswith("coldstart_route_"):
            metric = scheme.replace("coldstart_route_", "", 1)
            audit = route_audits.get(metric, {})
            routed_domain = (audit.get("route_domain_by_client") or {}).get(int(cid))
            if routed_domain is None:
                return {**A_global, **B_global}
            return {**A_global, **B_per_domain.get(routed_domain, B_global)}
        if scheme == "v11c_coldstart":
            # 新客户端：全局 A + global/domain pools both exclude the target client.
            return {
                **A_global,
                **_mix_b(_global_B_loo(cid), _domain_B_loo(d, cid)),
            }
        if scheme.startswith("local_fewshot"):
            cap_txt = scheme.replace("local_fewshot", "")
            cap = int(cap_txt)
            if cid in fewshot_states.get(cap, {}):
                return fewshot_states[cap][cid]
            return {**A_global, **_zero_B_state(cid)}
        if scheme == "select":
            chosen = selected_choice_by_client.get(int(cid), "local")
            return client_state(chosen, cid)
        if scheme == "select_without_local":
            chosen = selected_choice_without_local_by_client.get(int(cid), "coldstart")
            return client_state(chosen, cid)
        raise ValueError(scheme)

    def _select_for_clients(target_clients, candidates):
        choices = {}
        for cid in target_clients:
            d = domain_of[cid]
            dl = val_eval_loaders.get(int(cid)) or eval_loaders.get(d)
            if dl is None:
                choices[int(cid)] = candidates[0]
                continue
            best = None
            for cand in candidates:
                _install(model, client_state(cand, cid))
                loss, acc = _eval(model, dl, device, args.eval_max_batches)
                score = loss if math.isfinite(loss) else float("inf")
                if best is None or score < best[0]:
                    best = (score, cand, acc)
            choices[int(cid)] = best[1] if best else candidates[0]
        return choices

    if "select" in schemes or "select_without_local" in schemes:
        raw_candidates = [
            s.strip()
            for s in str(getattr(args, "select_candidates", "") or "").split(",")
            if s.strip()
        ]
        known_selectable = {
            "local", "fedsa", "global", "v7", "v11c", "base",
            "coldstart", "coldstart_geom", "v11c_coldstart",
        } | {f"local_fewshot{cap}" for cap in few_shot_caps}
        candidates = [
            s
            for s in raw_candidates
            if s not in {"select", "select_without_local"} and (s in schemes or s in known_selectable)
        ]
        if not candidates:
            candidates = ["local", "v7", "global", "v11c"]
        if "select" in schemes:
            selected_choice_by_client = _select_for_clients(eval_client_ids, candidates)
            counts = defaultdict(int)
            for choice in selected_choice_by_client.values():
                counts[choice] += 1
            print(f"[select] candidates={candidates} selected_counts={dict(sorted(counts.items()))}", flush=True)
        if "select_without_local" in schemes:
            no_local = [
                c for c in candidates
                if c != "local" and not c.startswith("local_fewshot") and c != "fedsa"
            ]
            if not no_local:
                no_local = ["coldstart", "global", "base"]
            selected_choice_without_local_by_client = _select_for_clients(
                eval_client_ids, no_local
            )
            counts = defaultdict(int)
            for choice in selected_choice_without_local_by_client.values():
                counts[choice] += 1
            print(
                f"[select_without_local] candidates={no_local} selected_counts={dict(sorted(counts.items()))}",
                flush=True,
            )

    # --- per-domain personalized eval ---
    results = {}
    for scheme in schemes:
        per_dom = defaultdict(list)        # domain -> [acc per client]
        per_dom_loss = defaultdict(list)
        per_client_acc = {}
        for cid in eval_client_ids:
            d = domain_of[cid]
            # eval_on_local: 每客户端用自己的 test_local（缺失则回退域测试）
            dl = local_eval_loaders.get(int(cid)) if local_eval_loaders else None
            if dl is None:
                dl = eval_loaders.get(d)
            if dl is None:
                continue
            _install(model, client_state(scheme, cid))
            loss, acc = _eval(model, dl, device, args.eval_max_batches)
            if not math.isnan(acc):
                per_dom[d].append(acc); per_dom_loss[d].append(loss)
                per_client_acc[str(int(cid))] = float(acc)
        dom_acc = {d: float(np.mean(v)) for d, v in per_dom.items()}
        dom_loss = {d: float(np.mean(v)) for d, v in per_dom_loss.items()}
        macro = float(np.mean(list(dom_acc.values()))) if dom_acc else float("nan")
        worst = float(min(dom_acc.values())) if dom_acc else float("nan")
        results[scheme] = {"macro_acc": macro, "worst_acc": worst,
                           "per_domain_acc": dom_acc, "per_domain_loss": dom_loss,
                           "per_client_acc": per_client_acc}
        if scheme == "select":
            counts = defaultdict(int)
            for choice in selected_choice_by_client.values():
                counts[choice] += 1
            results[scheme]["selected_counts"] = dict(sorted(counts.items()))
            results[scheme]["selected_choice_by_client"] = {
                str(k): v for k, v in sorted(selected_choice_by_client.items())
            }
        if scheme == "select_without_local":
            counts = defaultdict(int)
            for choice in selected_choice_without_local_by_client.values():
                counts[choice] += 1
            results[scheme]["selected_counts"] = dict(sorted(counts.items()))
            results[scheme]["selected_choice_by_client"] = {
                str(k): v for k, v in sorted(selected_choice_without_local_by_client.items())
            }
        if scheme == "coldstart_geom":
            results[scheme]["geom_route_domain_by_client"] = {
                str(k): v for k, v in sorted(geom_route_domain_by_client.items())
            }
            results[scheme]["geom_route_score_by_client"] = {
                str(k): float(v) for k, v in sorted(geom_route_scores.items())
            }
            results[scheme]["geom_route_top2_by_client"] = {
                str(k): v for k, v in sorted(geom_route_top2_by_client.items())
            }
            results[scheme]["geom_route_margin_by_client"] = {
                str(k): (None if v is None else float(v))
                for k, v in sorted(geom_route_margin_by_client.items())
            }
            results[scheme]["geom_route_oracle_match_by_client"] = {
                str(k): bool(v)
                for k, v in sorted(geom_route_oracle_match_by_client.items())
            }
            if geom_route_oracle_match_by_client:
                results[scheme]["geom_route_oracle_match_rate"] = float(
                    np.mean([bool(v) for v in geom_route_oracle_match_by_client.values()])
                )
        if scheme.startswith("coldstart_route_"):
            metric = scheme.replace("coldstart_route_", "", 1)
            audit = route_audits.get(metric, {})
            results[scheme]["route_metric"] = metric
            results[scheme]["route_audit"] = {
                key: ({str(k): v for k, v in sorted(value.items())}
                      if isinstance(value, dict) and key != "summary" else value)
                for key, value in audit.items()
            }
        print(f"[scheme {scheme:7s}] per-domain macro_acc={macro:.4f} worst={worst:.4f}", flush=True)

    client_manifest = ((benchmark_fingerprint.get("clients") or {}).get("per_client_manifest") or {})
    held_out_client_manifest = {}
    for cid in held_out_clients:
        cid_i = int(cid)
        manifest = dict(client_manifest.get(str(cid_i), {}) or {})
        manifest.setdefault("domain", domain_of.get(cid_i, "?"))
        manifest["n_train_rows"] = int(len(train_rows_by_client.get(cid_i, []) or []))
        manifest["n_val_rows"] = int(len(val_rows_by_client.get(cid_i, []) or []))
        manifest["n_test_local_rows"] = int(len(test_local_rows_by_client.get(cid_i, []) or []))
        held_out_client_manifest[str(cid_i)] = manifest

    onboarding_accounting = {"enabled": False}
    if bool(getattr(args, "onboarding_accounting", False)):
        per_client_accounting = {}
        shared_a_bytes = _state_nbytes(A_global)
        for cid in held_out_clients:
            cid_i = int(cid)
            probe_state = route_probe_states.get(cid_i, {})
            selected_b_bytes = {}
            route_seconds = {}
            for metric, audit in route_audits.items():
                routed_domain = (audit.get("route_domain_by_client") or {}).get(cid_i)
                selected_b_bytes[metric] = _state_nbytes(
                    B_per_domain.get(routed_domain, {})
                )
                route_seconds[metric] = float(
                    (audit.get("route_compute_seconds_by_client") or {}).get(cid_i, 0.0)
                )
            legacy_domain = geom_route_domain_by_client.get(cid_i)
            if legacy_domain is not None:
                selected_b_bytes["legacy_flat_b_cosine"] = _state_nbytes(
                    B_per_domain.get(legacy_domain, {})
                )
            per_client_accounting[str(cid_i)] = {
                "domain": domain_of.get(cid_i, "?"),
                "probe_samples": int(route_probe_sample_count.get(cid_i, 0)),
                "probe_train_seconds": float(route_probe_train_seconds.get(cid_i, 0.0)),
                "probe_signature_b_bytes": _state_nbytes(
                    probe_state, is_lora_b_param_name
                ),
                "probe_a_bytes": _state_nbytes(probe_state, is_lora_a_param_name),
                "probe_upload_bytes_by_metric": {
                    metric: (
                        0 if metric in {"largest_domain", "random", "oracle"}
                        else _state_nbytes(probe_state, is_lora_b_param_name)
                        + (_state_nbytes(probe_state, is_lora_a_param_name)
                           if metric == "delta_w_cosine" else 0)
                    )
                    for metric in route_audits
                },
                "route_compute_seconds_by_metric": route_seconds,
                "shared_a_download_bytes": int(shared_a_bytes),
                "selected_expert_b_download_bytes_by_metric": selected_b_bytes,
                "full_adapter_download_bytes_by_metric": {
                    metric: int(shared_a_bytes + value)
                    for metric, value in selected_b_bytes.items()
                },
            }
        probe_times = [x["probe_train_seconds"] for x in per_client_accounting.values()]
        upload_bytes = [x["probe_signature_b_bytes"] for x in per_client_accounting.values()]
        onboarding_accounting = {
            "enabled": True,
            "measurement_scope": {
                "probe_train_seconds": "wall-clock local adapter training in this process",
                "route_compute_seconds": "server-side scoring only; excludes transport",
                "payload_bytes": "tensor bytes only; excludes protocol/serialization overhead",
                "base_model": "excluded because it is assumed pre-installed",
            },
            "per_client": per_client_accounting,
            "summary": {
                "num_clients": int(len(per_client_accounting)),
                "mean_probe_train_seconds": float(np.mean(probe_times)) if probe_times else None,
                "mean_probe_signature_b_bytes": float(np.mean(upload_bytes)) if upload_bytes else None,
                "shared_a_download_bytes": int(shared_a_bytes),
            },
        }

    report = {"protocol_tag": protocol_tag,
              "config": {k: getattr(args, k) for k in
                         ["model", "benchmark_dir", "target_modules", "max_steps",
                          "eval_max_batches", "v7_b_mode", "local_epochs",
                          "max_train_samples_per_client", "cold_start", "eval_on_local",
                          "v11c_mu", "select_candidates", "held_out_clients",
                          "held_out_policy", "held_out_offset",
                          "held_out_eval_all", "few_shot_caps",
                          "held_out_route_probe_samples", "held_out_route_metrics",
                          "onboarding_accounting"]},
              "benchmark_fingerprint": benchmark_fingerprint,
              "eval_objective": ("per_client_local_test" if bool(getattr(args, "eval_on_local", False))
                                 else "per_domain_shared_test"),
              "strict_held_out": {
                  "enabled": bool(held_out_set),
                  "selection_policy": str(getattr(args, "held_out_policy", "first")),
                  "selection_offset": int(getattr(args, "held_out_offset", 0) or 0),
                  "held_out_clients": [int(x) for x in held_out_clients],
                  "train_clients": [int(x) for x in train_client_ids],
                  "eval_clients": [int(x) for x in eval_client_ids],
                  "held_out_domains": {
                      str(int(cid)): domain_of.get(int(cid), "?")
                      for cid in held_out_clients
                  },
                  "held_out_client_manifest": held_out_client_manifest,
                  "geom_route_domain_by_client": {
                      str(k): v for k, v in sorted(geom_route_domain_by_client.items())
                  },
                  "geom_route_score_by_client": {
                      str(k): float(v) for k, v in sorted(geom_route_scores.items())
                  },
                  "geom_route_top2_by_client": {
                      str(k): v for k, v in sorted(geom_route_top2_by_client.items())
                  },
                  "geom_route_margin_by_client": {
                      str(k): (None if v is None else float(v))
                      for k, v in sorted(geom_route_margin_by_client.items())
                  },
                  "geom_route_oracle_match_by_client": {
                      str(k): bool(v)
                      for k, v in sorted(geom_route_oracle_match_by_client.items())
                  },
                  "geom_route_oracle_match_rate": (
                      float(np.mean([bool(v) for v in geom_route_oracle_match_by_client.values()]))
                      if geom_route_oracle_match_by_client else None
                  ),
                  "geom_route_summary": geom_route_summary,
                  "route_audits": {
                      metric: {
                          key: ({str(k): v for k, v in sorted(value.items())}
                                if isinstance(value, dict) and key != "summary" else value)
                          for key, value in audit.items()
                      }
                      for metric, audit in route_audits.items()
                  },
              },
              "onboarding_accounting": onboarding_accounting,
              "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- summary table ---
    domains = sorted({d for s in results.values() for d in s["per_domain_acc"]})
    print("\n========= per-domain 个性化评测 (token acc) =========")
    head = f"{'scheme':8s}{'macro':>8s}{'worst':>8s}  " + " ".join(f"{d[:4]:>6s}" for d in domains)
    print(head)
    for scheme in schemes:
        r = results[scheme]
        cells = " ".join(f"{r['per_domain_acc'].get(d, float('nan')):6.3f}" for d in domains)
        print(f"{scheme:8s}{r['macro_acc']:8.4f}{r['worst_acc']:8.4f}  {cells}")
    print("----------------------------------------------------")
    if "v7" in results and "global" in results:
        dv = results["v7"]["macro_acc"] - results["global"]["macro_acc"]
        dw = results["v7"]["worst_acc"] - results["global"]["worst_acc"]
        print(f"[判读] v7 − global(通才): macro {dv:+.4f}, worst {dw:+.4f}")
        if dv > 0.003 or dw > 0.003:
            print("  -> 个性化目标下 v7(域特异B) 反超通才，方法方向成立。")
        else:
            print("  -> v7 未明显超过通才；域内 IID 下池化收益有限，或需 per-domain 数据更异质/更长训练。")
    if "v7" in results and "local" in results:
        dvl = results["v7"]["macro_acc"] - results["local"]["macro_acc"]
        print(f"[判读] v7 vs local(个性化上界): macro {dvl:+.4f}")
        cap = int(getattr(args, "max_train_samples_per_client", 0) or 0)
        if dvl >= 0.003:
            print(f"  -> v7 反超纯本地（每客户端样本={cap or 'full'}）：联邦去噪生效，v7 在数据稀缺下有真实价值。")
        elif cap and cap > 0:
            print(f"  -> 数据稀缺(={cap})下 v7 仍未超本地；可再压小样本或看冷启动。")
        else:
            print("  -> 数据充足下 v7≈local：联邦去噪无空间，建议跑 --max_train_samples_per_client 50 看稀缺 regime。")
    # 冷启动判读：新客户端无数据时，coldstart(域池化B) 是否远超 base(无适配)
    if "coldstart" in results and "base" in results:
        dc = results["coldstart"]["macro_acc"] - results["base"]["macro_acc"]
        print(f"[判读] coldstart − base(无适配): macro {dc:+.4f}")
        if dc > 0.01:
            print("  -> 同域池化 B 让新客户端零样本获得显著域能力（local 做不到），这是联邦的真实价值点。")
        else:
            print("  -> 同域池化 B 对新客户端零样本帮助有限。")
        if "local" in results:
            print(f"  [参考] base={results['base']['macro_acc']:.4f} coldstart={results['coldstart']['macro_acc']:.4f} "
                  f"local(有数据上界)={results['local']['macro_acc']:.4f}")
    if "v11c_coldstart" in results and "base" in results:
        dv = results["v11c_coldstart"]["macro_acc"] - results["base"]["macro_acc"]
        print(f"[判读] v11c_coldstart − base(无适配): macro {dv:+.4f} (μ={args.v11c_mu:.2f})")
    if "select" in results and "local" in results:
        ds = results["select"]["macro_acc"] - results["local"]["macro_acc"]
        print(f"[判读] select − local(孤立部署): macro {ds:+.4f}; selected={results['select'].get('selected_counts', {})}")
    print(f"\n[eval] 已保存: {args.out}", flush=True)


if __name__ == "__main__":
    main()

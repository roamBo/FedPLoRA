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
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities.data_utils import (  # noqa: E402
    load_domain_sft_benchmark, create_domain_client_dataloaders,
    create_domain_eval_dataloader, group_rows_by_domain, group_rows_by_client,
)
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
    domain_of = {int(c["client_id"]): str(c["domain"]) for c in bench["clients"]}
    client_ids, loaders = create_domain_client_dataloaders(bench["train"], tok, args)
    if args.max_clients and len(client_ids) > args.max_clients:
        client_ids, loaders = client_ids[:args.max_clients], loaders[:args.max_clients]

    test_by_dom = group_rows_by_domain(bench["test_domain"])
    eval_loaders = {d: create_domain_eval_dataloader(rows, tok, args) for d, rows in test_by_dom.items()}

    # 可选：每客户端用自己的 test_local（不同子分布），bulletproof 冷启动/onboarding 验证。
    local_eval_loaders = {}
    if bool(getattr(args, "eval_on_local", False)):
        from utilities.data_utils import group_rows_by_client
        test_by_client = group_rows_by_client(bench.get("test_local", []) or [])
        for cid, rows in test_by_client.items():
            if rows:
                local_eval_loaders[int(cid)] = create_domain_eval_dataloader(rows, tok, args)
        n_missing = sum(1 for c in client_ids if int(c) not in local_eval_loaders)
        print(f"[eval] eval_on_local=ON: {len(local_eval_loaders)} 个客户端有本地测试集"
              f"（{n_missing} 个缺失将回退域测试）", flush=True)

    val_eval_loaders = {}
    val_by_client = group_rows_by_client(bench.get("val", []) or [])
    for cid, rows in val_by_client.items():
        if rows:
            val_eval_loaders[int(cid)] = create_domain_eval_dataloader(rows, tok, args)

    model = create_peft_causal_lm_model(args)
    init_fedplora_adapters(model)
    A0 = _snapshot(model, is_lora_a_param_name)
    a_keys = list(A0.keys())
    b_keys = [k.replace("lora_A", "lora_B") for k in a_keys]

    # --- capture per-client A_i, B_i ---
    A_by, B_by = {}, {}
    for n, cid in enumerate(client_ids):
        _reset_adapters(model, A0)
        _train_one_client(model, loaders[n], args, device)
        sd = model.state_dict()
        A_by[cid] = {k: sd[k].detach().cpu().clone() for k in a_keys}
        B_by[cid] = {bk: sd[bk].detach().cpu().clone() for bk in b_keys if bk in sd}
        print(f"[cap] {cid} ({domain_of.get(cid,'?')}) [{n+1}/{len(client_ids)}]", flush=True)

    # --- precompute aggregates ---
    A_global = aggregate_global_A(A_by, a_keys)
    B_global = {bk: torch.stack([B_by[c][bk].float() for c in client_ids if bk in B_by[c]], 0).mean(0)
                for bk in b_keys}
    B_per_domain = aggregate_per_domain_B(B_by, domain_of, b_keys, mode=args.v7_b_mode)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    if bool(getattr(args, "cold_start", False)):
        # 冷启动：模拟"新客户端无本地数据"。base=无适配(B=0) 下界；
        # coldstart=用同域其它客户端池化的 B（留一法），这是新机构能零样本拿到的。
        for s in ("base", "coldstart", "v11c_coldstart"):
            if s not in schemes:
                schemes.append(s)

    def _domain_B_loo(domain, exclude_cid):
        """同域 B 池化，排除 exclude_cid（留一法，模拟新客户端）。"""
        cids = [c for c in client_ids if domain_of.get(c) == domain and c != exclude_cid]
        out = {}
        for bk in b_keys:
            Bs = [B_by[c][bk] for c in cids if bk in B_by[c]]
            if Bs:
                out[bk] = _consolidate_B(Bs, args.v7_b_mode)
        return out

    def _global_B_loo(exclude_cid):
        """Global B pool excluding one client, used for true cold-start simulation."""
        cids = [c for c in client_ids if c != exclude_cid]
        out = {}
        for bk in b_keys:
            Bs = [B_by[c][bk] for c in cids if bk in B_by[c]]
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

    selected_choice_by_client = {}

    def client_state(scheme, cid):
        d = domain_of[cid]
        if scheme == "local":
            return {**A_by[cid], **B_by[cid]}
        if scheme == "fedsa":
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
            zb = {bk: torch.zeros_like(B_by[cid][bk]) for bk in b_keys if bk in B_by[cid]}
            return {**A_global, **zb}
        if scheme == "coldstart":
            # 新客户端：全局 A + 同域其它客户端池化的 B（不含自己）
            return {**A_global, **_domain_B_loo(d, cid)}
        if scheme == "v11c_coldstart":
            # 新客户端：全局 A + global/domain pools both exclude the target client.
            return {
                **A_global,
                **_mix_b(_global_B_loo(cid), _domain_B_loo(d, cid)),
            }
        if scheme == "select":
            chosen = selected_choice_by_client.get(int(cid), "local")
            return client_state(chosen, cid)
        raise ValueError(scheme)

    if "select" in schemes:
        raw_candidates = [
            s.strip()
            for s in str(getattr(args, "select_candidates", "") or "").split(",")
            if s.strip()
        ]
        candidates = [
            s
            for s in raw_candidates
            if s not in {"select"} and (s in schemes or s in {"local", "fedsa", "global", "v7", "v11c", "base", "coldstart", "v11c_coldstart"})
        ]
        if not candidates:
            candidates = ["local", "v7", "global", "v11c"]
        for cid in client_ids:
            d = domain_of[cid]
            dl = val_eval_loaders.get(int(cid)) or eval_loaders.get(d)
            if dl is None:
                selected_choice_by_client[int(cid)] = candidates[0]
                continue
            best = None
            for cand in candidates:
                _install(model, client_state(cand, cid))
                loss, acc = _eval(model, dl, device, args.eval_max_batches)
                score = loss if math.isfinite(loss) else float("inf")
                if best is None or score < best[0]:
                    best = (score, cand, acc)
            selected_choice_by_client[int(cid)] = best[1] if best else candidates[0]
        counts = defaultdict(int)
        for choice in selected_choice_by_client.values():
            counts[choice] += 1
        print(f"[select] candidates={candidates} selected_counts={dict(sorted(counts.items()))}", flush=True)

    # --- per-domain personalized eval ---
    results = {}
    for scheme in schemes:
        per_dom = defaultdict(list)        # domain -> [acc per client]
        per_dom_loss = defaultdict(list)
        for cid in client_ids:
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
        dom_acc = {d: float(np.mean(v)) for d, v in per_dom.items()}
        dom_loss = {d: float(np.mean(v)) for d, v in per_dom_loss.items()}
        macro = float(np.mean(list(dom_acc.values()))) if dom_acc else float("nan")
        worst = float(min(dom_acc.values())) if dom_acc else float("nan")
        results[scheme] = {"macro_acc": macro, "worst_acc": worst,
                           "per_domain_acc": dom_acc, "per_domain_loss": dom_loss}
        if scheme == "select":
            counts = defaultdict(int)
            for choice in selected_choice_by_client.values():
                counts[choice] += 1
            results[scheme]["selected_counts"] = dict(sorted(counts.items()))
            results[scheme]["selected_choice_by_client"] = {
                str(k): v for k, v in sorted(selected_choice_by_client.items())
            }
        print(f"[scheme {scheme:7s}] per-domain macro_acc={macro:.4f} worst={worst:.4f}", flush=True)

    report = {"config": {k: getattr(args, k) for k in
                         ["model", "benchmark_dir", "target_modules", "max_steps",
                          "eval_max_batches", "v7_b_mode", "local_epochs",
                          "max_train_samples_per_client", "cold_start", "eval_on_local",
                          "v11c_mu", "select_candidates"]},
              "eval_objective": ("per_client_local_test" if bool(getattr(args, "eval_on_local", False))
                                 else "per_domain_shared_test"),
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

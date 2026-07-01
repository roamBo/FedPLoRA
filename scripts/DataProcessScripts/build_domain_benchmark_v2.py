"""
build_domain_benchmark_v2.py — leak-free, K-consistent, balance-aware domain
benchmark builder for FedPLoRA, with an optional within-domain non-IID partition.

Fixes vs the original build_domain_benchmark_from_jsonl:
  P0  domain_test/val/pool split uses a per-domain RNG INDEPENDENT of
      num_clients_per_domain  -> identical test set for any client count.
  P1a group-by-prompt atomic splitting + cross-domain dedup -> zero prompt-level
      train/test leakage (asserted at the end).
  P1b --max_per_domain / --target_per_domain to cap or balance domain sizes.

Partition modes:
  iid        round-robin of prompt-groups (same-domain clients ~ IID).
  dirichlet  feature-skew non-IID: each client drawn a Dirichlet(alpha) mixture
             over sub-topics (response-length deciles by default).

Output is drop-in compatible with utilities/data_utils.load_domain_sft_benchmark:
  train.jsonl val.jsonl test_local.jsonl test_domain.jsonl test_global.jsonl
  clients.json domain_stats.json
"""

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Normalization / seeding
# --------------------------------------------------------------------------- #
def _norm(text):
    """NFKC (full->half width), collapse whitespace, lower — stable dedup/leak key."""
    s = unicodedata.normalize("NFKC", text or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _domain_seed(base_seed, domain, salt=0):
    h = hashlib.md5(f"{salt}:{domain}".encode("utf-8")).hexdigest()
    return (int(base_seed) + int(h[:8], 16)) % (2**32)


# --------------------------------------------------------------------------- #
# Sub-topic assignment for the non-IID (dirichlet) partition
# --------------------------------------------------------------------------- #
def _subtopic_ids(rows, mode, n_subtopics, seed):
    """Return an array of sub-topic ids in [0, n_subtopics) for each row."""
    if mode == "length":
        lengths = np.array([len((r.get("response") or "")) for r in rows], dtype=np.float64)
        if len(lengths) == 0:
            return np.zeros(0, dtype=np.int64)
        ranks = lengths.argsort().argsort()                  # 0..n-1 by length
        return (ranks * n_subtopics // max(len(rows), 1)).astype(np.int64)
    if mode == "hash":
        ids = np.empty(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            toks = _norm(r.get("prompt", "")) + " " + _norm(r.get("response", ""))
            grams = {toks[j:j + 3] for j in range(max(len(toks) - 2, 1))}
            mh = min((int(hashlib.md5(g.encode()).hexdigest()[:8], 16) for g in grams), default=0)
            ids[i] = mh % n_subtopics
        return ids
    if mode == "kmeans":
        # Optional semantic topics (TF-IDF + KMeans). Falls back to length on ImportError.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            texts = [(r.get("prompt", "") + " " + r.get("response", "")) for r in rows]
            X = TfidfVectorizer(max_features=4096).fit_transform(texts)
            km = KMeans(n_clusters=n_subtopics, random_state=seed, n_init=4)
            return km.fit_predict(X).astype(np.int64)
        except Exception:
            return _subtopic_ids(rows, "length", n_subtopics, seed)
    raise ValueError(f"unknown subtopic mode: {mode}")


# --------------------------------------------------------------------------- #
# Group-atomic split helper (no prompt straddles two splits)
# --------------------------------------------------------------------------- #
def _split_groups_by_ratio(group_keys, group_rows, ratios, rng):
    """Shuffle prompt-groups, then cut into chunks by row-count ratios.

    ratios: list summing to <= 1.0; returns list of row-lists, one per ratio,
    plus a trailing 'remainder' list.
    """
    keys = list(group_keys)
    rng.shuffle(keys)
    total = sum(len(group_rows[k]) for k in keys)
    cuts = [int(total * r) for r in ratios]
    out = [[] for _ in ratios]
    rem = []
    acc = 0
    bucket = 0
    targets = np.cumsum(cuts).tolist()
    for k in keys:
        grp = group_rows[k]
        if bucket < len(targets) and acc >= targets[bucket]:
            bucket += 1
        if bucket < len(out):
            out[bucket].extend(grp)
        else:
            rem.extend(grp)
        acc += len(grp)
    return out, rem


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def build_domain_benchmark_v2(
    input_path,
    output_dir,
    num_clients_per_domain=5,
    min_samples_per_client=50,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    max_per_domain=0,
    target_per_domain=0,
    dedup="prompt",
    partition="iid",
    dirichlet_alpha=0.5,
    subtopic="length",
    n_subtopics=10,
):
    rows = _load_jsonl(input_path)

    # group by domain (drop empties)
    by_domain = defaultdict(list)
    for r in rows:
        d = (r.get("domain") or "").strip()
        p = (r.get("prompt") or "").strip()
        a = (r.get("response") or "").strip()
        if d and p and a:
            by_domain[d].append(r)

    # cross-domain contamination: a normalized prompt may only belong to ONE domain
    prompt_domains = defaultdict(lambda: defaultdict(int))
    for d, drows in by_domain.items():
        for r in drows:
            prompt_domains[_norm(r["prompt"])][d] += 1
    keep_domain_of = {}
    for pk, dc in prompt_domains.items():
        keep_domain_of[pk] = max(dc.items(), key=lambda kv: kv[1])[0]

    split_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(split_dir, exist_ok=True)

    clients_manifest = []
    domain_stats = {}
    client_train, client_val, client_local_test = [], [], []
    domain_test_rows, global_test = [], []

    client_id = 0
    for domain in sorted(by_domain.keys()):
        rng_split = np.random.default_rng(_domain_seed(seed, domain, salt=0))    # INDEPENDENT of K
        rng_shard = np.random.default_rng(_domain_seed(seed, domain, salt=999))  # client allocation

        drows = by_domain[domain]

        # --- dedup + cross-domain purge ---
        if dedup == "pair":
            seen, ded = set(), []
            for r in drows:
                k = (_norm(r["prompt"]), _norm(r["response"]))
                if k in seen or keep_domain_of[_norm(r["prompt"])] != domain:
                    continue
                seen.add(k)
                ded.append(r)
            drows = ded
        else:
            drows = [r for r in drows if keep_domain_of[_norm(r["prompt"])] == domain]

        # group by prompt (atomic unit so no prompt straddles splits)
        groups = defaultdict(list)
        for r in drows:
            groups[_norm(r["prompt"])].append(r)
        if dedup == "prompt":
            groups = {k: v[:1] for k, v in groups.items()}     # one row per prompt
        group_keys = list(groups.keys())

        # --- cap / balance at group level (deterministic via rng_split) ---
        budget = 0
        if target_per_domain > 0:
            budget = int(target_per_domain)
        elif max_per_domain > 0:
            budget = int(max_per_domain)
        if budget > 0:
            rng_split.shuffle(group_keys)
            kept, acc = [], 0
            for k in group_keys:
                if acc >= budget:
                    break
                kept.append(k)
                acc += len(groups[k])
            group_keys = kept

        n_rows = sum(len(groups[k]) for k in group_keys)

        # --- domain-level test held out FIRST, atomic, INDEPENDENT of K ---
        (test_chunk,), pool_rem = _split_groups_by_ratio(
            group_keys, groups, [test_ratio], rng_split
        )
        for r in test_chunk:
            domain_test_rows.append(r)
            global_test.append(r)

        # remaining pool groups -> rebuild group map for the pool
        pool_groups = defaultdict(list)
        for r in pool_rem:
            pool_groups[_norm(r["prompt"])].append(r)
        pool_keys = list(pool_groups.keys())

        # --- allocate pool groups to clients ---
        if num_clients_per_domain <= 1:
            client_keys = [pool_keys]
        elif partition == "iid":
            rng_shard.shuffle(pool_keys)
            client_keys = [[] for _ in range(num_clients_per_domain)]
            for i, k in enumerate(pool_keys):
                client_keys[i % num_clients_per_domain].append(k)
        elif partition == "dirichlet":
            pool_rows_flat = [pool_groups[k][0] for k in pool_keys]   # 1 repr row / group
            sub = _subtopic_ids(pool_rows_flat, subtopic, n_subtopics, seed)
            client_keys = [[] for _ in range(num_clients_per_domain)]
            for t in range(n_subtopics):
                idx_t = [pool_keys[i] for i in range(len(pool_keys)) if sub[i] == t]
                if not idx_t:
                    continue
                rng_shard.shuffle(idx_t)
                props = rng_shard.dirichlet([dirichlet_alpha] * num_clients_per_domain)
                cuts = (np.cumsum(props) * len(idx_t)).astype(int)[:-1]
                chunks = np.split(np.array(idx_t, dtype=object), cuts)
                for c in range(num_clients_per_domain):
                    client_keys[c].extend(list(chunks[c]))
        else:
            raise ValueError(f"unknown partition: {partition}")

        # --- per-client local test/val/train (group-atomic) ---
        kept_clients = 0
        for ck in client_keys:
            ck_rows = sum(len(pool_groups[k]) for k in ck)
            if ck_rows < min_samples_per_client:
                continue
            rng_client = np.random.default_rng(_domain_seed(seed, f"{domain}:{client_id}", salt=7))
            (ltest, lval), ltrain = _split_groups_by_ratio(
                ck, pool_groups, [test_ratio, val_ratio], rng_client
            )
            if len(ltrain) < max(1, min_samples_per_client // 2):
                continue
            for r in ltrain:
                rr = dict(r); rr["client_id"] = client_id; client_train.append(rr)
            for r in lval:
                rr = dict(r); rr["client_id"] = client_id; client_val.append(rr)
            for r in ltest:
                rr = dict(r); rr["client_id"] = client_id; client_local_test.append(rr)
            clients_manifest.append({
                "client_id": client_id, "domain": domain,
                "n_train": len(ltrain), "n_val": len(lval), "n_local_test": len(ltest),
            })
            client_id += 1
            kept_clients += 1

        domain_stats[domain] = {
            "n_total": n_rows,
            "n_domain_test": len(test_chunk),
            "n_clients": kept_clients,
            "partition": partition,
        }

    # --- write ---
    _write_jsonl(os.path.join(split_dir, "train.jsonl"), client_train)
    _write_jsonl(os.path.join(split_dir, "val.jsonl"), client_val)
    _write_jsonl(os.path.join(split_dir, "test_local.jsonl"), client_local_test)
    _write_jsonl(os.path.join(split_dir, "test_domain.jsonl"), domain_test_rows)
    _write_jsonl(os.path.join(split_dir, "test_global.jsonl"), global_test)
    with open(os.path.join(split_dir, "clients.json"), "w", encoding="utf-8") as f:
        json.dump(clients_manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(split_dir, "domain_stats.json"), "w", encoding="utf-8") as f:
        json.dump(domain_stats, f, indent=2, ensure_ascii=False)

    # --- self-check: assert zero prompt-level train/test leakage ---
    def pset(rows):
        return set(_norm(r["prompt"]) for r in rows)
    tr = pset(client_train)
    leak_dt = len(tr & pset(domain_test_rows))
    leak_lt = len(tr & pset(client_local_test))
    leak_val = len(pset(client_val) & pset(domain_test_rows))
    print(f"[v2] split_dir={split_dir} clients={len(clients_manifest)} partition={partition}")
    for d, v in sorted(domain_stats.items()):
        print(f"[v2] {d:11s} n_total={v['n_total']:6d} domain_test={v['n_domain_test']:5d} clients={v['n_clients']}")
    print(f"[leakcheck] train∩domain_test(prompt)={leak_dt}  train∩local_test={leak_lt}  val∩domain_test={leak_val}")
    assert leak_dt == 0 and leak_lt == 0 and leak_val == 0, "prompt-level leakage detected!"
    print("[leakcheck] PASS — zero prompt-level leakage")
    return {"split_dir": split_dir, "num_clients": len(clients_manifest),
            "domains": sorted(domain_stats.keys())}


def main():
    ap = argparse.ArgumentParser(description="FedPLoRA domain benchmark builder v2 (leak-free + non-IID).")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", default="data/domain_benchmark_v2")
    ap.add_argument("--num_clients_per_domain", type=int, default=5)
    ap.add_argument("--min_samples_per_client", type=int, default=50)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_domain", type=int, default=0, help="cap each domain to this many rows (0=off)")
    ap.add_argument("--target_per_domain", type=int, default=0, help="balance: downsample each domain to this many rows (0=off)")
    ap.add_argument("--dedup", choices=["none", "pair", "prompt"], default="prompt")
    ap.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--dirichlet_alpha", type=float, default=0.5, help="lower = more non-IID")
    ap.add_argument("--subtopic", choices=["length", "hash", "kmeans"], default="length")
    ap.add_argument("--n_subtopics", type=int, default=10)
    args = ap.parse_args()

    build_domain_benchmark_v2(
        input_path=args.input_jsonl,
        output_dir=args.output_dir,
        num_clients_per_domain=args.num_clients_per_domain,
        min_samples_per_client=args.min_samples_per_client,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_per_domain=args.max_per_domain,
        target_per_domain=args.target_per_domain,
        dedup=args.dedup,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        subtopic=args.subtopic,
        n_subtopics=args.n_subtopics,
    )


if __name__ == "__main__":
    main()

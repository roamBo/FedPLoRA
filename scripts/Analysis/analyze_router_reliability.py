#!/usr/bin/env python3
"""Router reliability audit for FedPLoRA-OS/v13 result JSONs.

This is a zero-GPU analysis helper.  It reads training result JSONs containing
`lora_expert_client_clusters` and reports whether B-subspace routing creates
mixed clusters, tiny clusters, or domain over-segmentation.  If personalized
eval JSONs are also present under the same root, per-client accuracies are
joined for qualitative diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DOMAIN_BY_RANGE = [
    ("code", range(0, 5)),
    ("education", range(5, 10)),
    ("finance", range(10, 15)),
    ("general", range(15, 20)),
    ("legal", range(20, 25)),
    ("math", range(25, 30)),
    ("medical", range(30, 35)),
]


def _json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _final_round(data: Dict[str, Any]) -> Dict[str, Any]:
    rounds = data.get("rounds") or []
    for item in reversed(rounds):
        if not item.get("eval_skipped"):
            return item
    return rounds[-1] if rounds else {}


def _seed_from_data_or_path(data: Dict[str, Any], path: Path) -> Optional[int]:
    args = data.get("args") or {}
    if args.get("seed") is not None:
        try:
            return int(args.get("seed"))
        except Exception:
            pass
    text = str(path)
    for token in ("seed42", "seed43", "seed44", "seed_42", "seed_43", "seed_44"):
        if token in text:
            return int(token[-2:])
    return None


def _domain_from_manifest(data: Dict[str, Any]) -> Dict[int, str]:
    fp = data.get("benchmark_fingerprint") or {}
    clients = (fp.get("clients") or {}).get("per_client_manifest") or {}
    out = {}
    for cid, row in clients.items():
        try:
            out[int(cid)] = str(row.get("domain", "?"))
        except Exception:
            continue
    return out


def _fallback_domain(cid: int) -> str:
    for dom, rng in DOMAIN_BY_RANGE:
        if cid in rng:
            return dom
    return "?"


def _method_label(path: Path, data: Dict[str, Any]) -> str:
    parent = path.parent.name
    if parent and parent not in {"result_logs"}:
        return parent
    return str((data.get("args") or {}).get("agg_type") or path.stem)


def _collect_personalized(root: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    """seed -> scheme -> cid(str) -> acc"""
    out: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for path in sorted(root.rglob("*.json")):
        data = _json_load(path)
        if not data or "results" not in data or "rounds" in data:
            continue
        seed = None
        cfg = data.get("config") or {}
        try:
            seed = int(cfg.get("seed")) if cfg.get("seed") is not None else None
        except Exception:
            seed = _seed_from_data_or_path(data, path)
        if seed is None:
            seed = _seed_from_data_or_path(data, path)
        if seed is None:
            continue
        for scheme, row in (data.get("results") or {}).items():
            per_client = row.get("per_client_acc") or {}
            if per_client:
                out[seed][scheme] = {
                    str(k): float(v)
                    for k, v in per_client.items()
                    if _is_number(v)
                }
    return out


def _is_number(x: Any) -> bool:
    try:
        val = float(x)
    except Exception:
        return False
    return math.isfinite(val)


def _analyze_train_json(path: Path, data: Dict[str, Any], personalized: Dict[int, Dict[str, Dict[str, float]]]) -> Optional[Dict[str, Any]]:
    final = _final_round(data)
    clusters_raw = final.get("lora_expert_client_clusters") or {}
    if not clusters_raw:
        return None
    clusters = {int(cid): int(cluster) for cid, cluster in clusters_raw.items()}
    domains = _domain_from_manifest(data)
    for cid in clusters:
        domains.setdefault(cid, _fallback_domain(cid))

    by_cluster: Dict[int, List[int]] = defaultdict(list)
    by_domain: Dict[str, List[int]] = defaultdict(list)
    for cid, cluster in clusters.items():
        by_cluster[cluster].append(cid)
        by_domain[domains.get(cid, "?")].append(cid)

    cluster_hist = {}
    cluster_majority = {}
    for cluster, cids in sorted(by_cluster.items()):
        cnt = Counter(domains.get(cid, "?") for cid in cids)
        maj_dom, maj_n = cnt.most_common(1)[0]
        cluster_hist[str(cluster)] = dict(sorted(cnt.items()))
        cluster_majority[cluster] = {
            "majority_domain": maj_dom,
            "majority_count": int(maj_n),
            "size": int(len(cids)),
            "purity": float(maj_n / max(1, len(cids))),
        }

    domain_cluster_counts = {
        dom: len({clusters[cid] for cid in cids})
        for dom, cids in sorted(by_domain.items())
    }

    seed = _seed_from_data_or_path(data, path)
    joined_perf = personalized.get(seed or -1, {})
    client_rows = []
    minority_clients = []
    tiny_cluster_clients = []
    for cid, cluster in sorted(clusters.items()):
        maj = cluster_majority[cluster]
        dom = domains.get(cid, "?")
        row = {
            "client_id": int(cid),
            "domain": dom,
            "cluster": int(cluster),
            "cluster_size": int(maj["size"]),
            "cluster_majority_domain": maj["majority_domain"],
            "cluster_purity": float(maj["purity"]),
            "is_cluster_majority_domain": bool(dom == maj["majority_domain"]),
            "domain_num_clusters": int(domain_cluster_counts.get(dom, 0)),
        }
        for scheme, per_client in sorted(joined_perf.items()):
            if str(cid) in per_client:
                row[f"acc_{scheme}"] = float(per_client[str(cid)])
        if not row["is_cluster_majority_domain"]:
            minority_clients.append(int(cid))
        if int(row["cluster_size"]) <= 1:
            tiny_cluster_clients.append(int(cid))
        client_rows.append(row)

    expert = final.get("lora_expert_stats") or {}
    return {
        "path": str(path),
        "method": _method_label(path, data),
        "agg": (data.get("args") or {}).get("agg_type"),
        "seed": seed,
        "benchmark_dir": data.get("benchmark_dir") or (data.get("args") or {}).get("benchmark_dir"),
        "macro": final.get("domain_macro_token_accuracy"),
        "local": final.get("client_local_macro_token_accuracy"),
        "nmi": expert.get("domain_nmi"),
        "ari": expert.get("domain_ari"),
        "selected_k": expert.get("selected_k"),
        "silhouette": expert.get("silhouette"),
        "cluster_hist": cluster_hist,
        "domain_cluster_counts": domain_cluster_counts,
        "num_clients": len(client_rows),
        "num_clusters": len(by_cluster),
        "num_mixed_clusters": sum(1 for v in cluster_majority.values() if float(v["purity"]) < 1.0),
        "num_minority_clients": len(minority_clients),
        "minority_clients": minority_clients,
        "num_singleton_clients": len(tiny_cluster_clients),
        "singleton_clients": tiny_cluster_clients,
        "client_rows": client_rows,
    }


def _f(x: Any, digits: int = 4) -> str:
    if not _is_number(x):
        return ""
    return f"{float(x):.{digits}f}"


def _table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _build_md(payload: Dict[str, Any]) -> str:
    runs = payload.get("runs") or []
    lines = [
        "# FedPLoRA router reliability audit",
        "",
        f"- root: `{payload.get('root')}`",
        f"- runs: {len(runs)}",
        "",
        "## Run-level summary",
        "",
        _table(
            ["method", "seed", "Local", "NMI", "ARI", "K", "sil", "mixed_clusters", "minority_clients", "domain_cluster_counts"],
            [
                [
                    r.get("method", ""),
                    r.get("seed", ""),
                    _f(r.get("local")),
                    _f(r.get("nmi")),
                    _f(r.get("ari")),
                    r.get("selected_k", ""),
                    _f(r.get("silhouette")),
                    r.get("num_mixed_clusters", 0),
                    r.get("num_minority_clients", 0),
                    json.dumps(r.get("domain_cluster_counts", {}), ensure_ascii=False, sort_keys=True),
                ]
                for r in runs
            ],
        ),
        "",
        "## Suspicious clients",
        "",
    ]
    client_rows = []
    for r in runs:
        for row in r.get("client_rows", []):
            if (
                not row.get("is_cluster_majority_domain", True)
                or int(row.get("cluster_size", 0)) <= 1
                or int(row.get("domain_num_clusters", 0)) > 1
            ):
                client_rows.append([
                    r.get("method", ""),
                    r.get("seed", ""),
                    row.get("client_id", ""),
                    row.get("domain", ""),
                    row.get("cluster", ""),
                    row.get("cluster_size", ""),
                    row.get("cluster_majority_domain", ""),
                    _f(row.get("cluster_purity")),
                    row.get("domain_num_clusters", ""),
                ])
    if client_rows:
        lines.append(_table(
            ["method", "seed", "cid", "domain", "cluster", "cluster_size", "cluster_majority", "purity", "domain_K"],
            client_rows,
        ))
    else:
        lines.append("No minority/tiny/over-segmented client detected.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Result root to scan recursively.")
    ap.add_argument("--include", type=str, default="",
                    help="Optional regex; only JSON paths matching it are analyzed.")
    ap.add_argument("--exclude", type=str, default="",
                    help="Optional regex; JSON paths matching it are skipped.")
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--output_md", type=Path, required=True)
    args = ap.parse_args()

    personalized = _collect_personalized(args.root)
    runs = []
    inc = re.compile(args.include) if args.include else None
    exc = re.compile(args.exclude) if args.exclude else None
    for path in sorted(args.root.rglob("*.json")):
        path_text = str(path)
        if inc and not inc.search(path_text):
            continue
        if exc and exc.search(path_text):
            continue
        data = _json_load(path)
        if not data or "rounds" not in data:
            continue
        analyzed = _analyze_train_json(path, data, personalized)
        if analyzed is not None:
            runs.append(analyzed)

    payload = {"root": str(args.root), "runs": runs}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(_build_md(payload) + "\n", encoding="utf-8")
    print(f"[router-reliability] runs={len(runs)} json={args.output_json} md={args.output_md}")


if __name__ == "__main__":
    main()

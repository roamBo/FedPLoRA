#!/usr/bin/env python3
"""Run official lm-eval tasks on exported FedPLoRA PEFT adapters.

``global`` evaluates the reconstructed global adapter once. ``domain_clients``
evaluates every routed client adapter in the declared task domain and reports
the macro mean; it never cherry-picks the best client.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE_ROOT = REPO_ROOT / "data" / "external_lm_eval_hf_cache"
TASK_DATASETS = {
    "mmlu": ("cais/mmlu", "all"),
    "pubmedqa": ("pubmed_qa", "pqa_labeled"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}


def _resolve_hf_cache_root(explicit=None):
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"[external-eval][error] HF cache dir not found: {root}")
        return root
    if DEFAULT_HF_CACHE_ROOT.is_dir():
        return DEFAULT_HF_CACHE_ROOT
    raise SystemExit(
        "[external-eval][error] offline HF cache missing: "
        f"{DEFAULT_HF_CACHE_ROOT}\n"
        "rsync data/external_lm_eval_hf_cache/ to the repo, or pass --hf_cache_dir."
    )


def _hf_subprocess_env(cache_root, offline=True):
    datasets_cache = cache_root / "datasets"
    if not datasets_cache.is_dir():
        datasets_cache = cache_root
    env = os.environ.copy()
    env["HF_HOME"] = str(cache_root)
    env["HF_DATASETS_CACHE"] = str(datasets_cache)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env.pop("HF_ENDPOINT", None)
    return env


def _preflight_offline_datasets(task_names, env):
    try:
        from datasets import load_dataset
        import datasets as datasets_pkg
    except ImportError as exc:
        raise SystemExit("[external-eval][error] datasets package required for offline preflight") from exc
    print(
        "[external-eval][preflight] datasets="
        f"{datasets_pkg.__version__} cache={env['HF_DATASETS_CACHE']}",
        flush=True,
    )
    for task in task_names:
        spec = TASK_DATASETS.get(task)
        if spec is None:
            continue
        name, config = spec
        print(f"[external-eval][preflight] load_dataset {name!r} config={config!r}", flush=True)
        with _temporary_environ(env):
            ds = load_dataset(name, config, trust_remote_code=False)
        print(f"[external-eval][preflight][ok] {task} splits={list(ds.keys())}", flush=True)


class _temporary_environ:
    def __init__(self, values):
        self._values = values
        self._backup = {}

    def __enter__(self):
        for key, value in self._values.items():
            self._backup[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, old in self._backup.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        return False


def _parse_tasks(text):
    out = []
    for item in str(text).split(","):
        task, sep, domain = item.strip().partition(":")
        if not sep or not task or not domain:
            raise SystemExit(f"[external-eval][error] expected task:domain, got {item!r}")
        if not SAFE_NAME.match(task) or not SAFE_NAME.match(domain):
            raise SystemExit(f"[external-eval][error] unsafe task/domain token: {item!r}")
        out.append((task, domain))
    return out


def _find_result(root):
    candidates = sorted(Path(root).rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            return path, data
    raise RuntimeError(f"no lm-eval results JSON under {root}")


def _numeric_metrics(result, task):
    rows = result.get("results") or {}
    row = rows.get(task)
    if row is None:
        row = (result.get("groups") or {}).get(task)
    if isinstance(row, dict):
        return {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    child_rows = [
        value for key, value in rows.items()
        if str(key).startswith(f"{task}_") and isinstance(value, dict)
    ]
    if not child_rows:
        return {}
    keys = set.intersection(*(
        {key for key, value in row.items() if isinstance(value, (int, float))}
        for row in child_rows
    ))
    return {
        key: float(value)
        for key in sorted(keys)
        if math.isfinite(value := sum(float(row[key]) for row in child_rows) / len(child_rows))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_manifest", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task:domain mappings")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["global", "domain_clients", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", default="")
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help="Forward lm-eval's explicit opt-in required by code-execution tasks such as MBPP.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--hf_cache_dir",
        default="",
        help="Offline HuggingFace cache root (default: data/external_lm_eval_hf_cache under repo).",
    )
    parser.add_argument(
        "--allow_hf_network",
        action="store_true",
        help="Do not force HF_HUB_OFFLINE; only use --hf_cache_dir as cache location.",
    )
    parser.add_argument(
        "--skip_dataset_preflight",
        action="store_true",
        help="Skip offline load_dataset checks before launching lm_eval.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.adapter_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = str(manifest["model"])
    tasks = _parse_tasks(args.tasks)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    clients_by_domain = defaultdict(list)
    for cid, row in (manifest.get("clients") or {}).items():
        clients_by_domain[str(row.get("domain", "?"))].append((int(cid), row["adapter_dir"]))
    for domain in clients_by_domain:
        clients_by_domain[domain].sort()

    cache_root = _resolve_hf_cache_root(args.hf_cache_dir or None)
    hf_env = _hf_subprocess_env(cache_root, offline=not args.allow_hf_network)
    unique_tasks = sorted({task for task, _ in tasks})
    if not args.dry_run and not args.skip_dataset_preflight:
        _preflight_offline_datasets(unique_tasks, hf_env)

    jobs = []
    for task, domain in tasks:
        if args.mode in {"global", "both"}:
            jobs.append((task, domain, "global", manifest["global_adapter_dir"]))
        if args.mode in {"domain_clients", "both"}:
            adapters = clients_by_domain.get(domain, [])
            if not adapters:
                raise SystemExit(f"[external-eval][error] no exported clients for domain={domain}")
            jobs.extend((task, domain, f"client_{cid:03d}", path) for cid, path in adapters)

    completed = []
    for task, domain, deployment, adapter in jobs:
        run_dir = output / task / deployment
        run_dir.mkdir(parents=True, exist_ok=True)
        model_args = f"pretrained={model},peft={adapter},dtype={args.dtype},trust_remote_code=False"
        command = [
            sys.executable, "-m", "lm_eval", "--model", "hf",
            "--model_args", model_args, "--tasks", task,
            "--device", args.device, "--batch_size", str(args.batch_size),
            "--num_fewshot", str(args.num_fewshot), "--output_path", str(run_dir),
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        if args.confirm_run_unsafe_code:
            command.append("--confirm_run_unsafe_code")
        print("[external-eval][command]", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True, env=hf_env)
        result_path, result = _find_result(run_dir)
        completed.append({
            "task": task,
            "domain": domain,
            "deployment": deployment,
            "adapter": adapter,
            "result_path": str(result_path),
            "metrics": _numeric_metrics(result, task),
        })

    if args.dry_run:
        print(f"[external-eval][dry-run] jobs={len(jobs)}")
        return
    grouped = defaultdict(list)
    for row in completed:
        group = "global" if row["deployment"] == "global" else "domain_clients_macro"
        grouped[(row["task"], row["domain"], group)].append(row)
    summaries = []
    for (task, domain, group), rows in sorted(grouped.items()):
        keys = sorted(set.intersection(*(set(row["metrics"]) for row in rows))) if rows else []
        mean_metrics = {
            key: float(sum(row["metrics"][key] for row in rows) / len(rows)) for key in keys
        }
        summaries.append({
            "task": task, "domain": domain, "deployment": group,
            "num_adapter_runs": len(rows), "mean_metrics": mean_metrics,
        })
    payload = {
        "schema_version": 1,
        "adapter_manifest": str(manifest_path),
        "aggregation_rule": "unweighted macro mean over all routed client adapters in declared domain",
        "runs": completed,
        "summaries": summaries,
    }
    summary_path = output / "external_eval_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[external-eval][ok] summary={summary_path}")


if __name__ == "__main__":
    main()

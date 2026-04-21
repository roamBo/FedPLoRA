import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple


LOG_LINE_RE = re.compile(r"^\[log\] writing console output to (.+)$")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stream_subprocess(
    cmd: List[str], env: Dict[str, str], master_fp
) -> Tuple[int, Optional[str]]:
    """
    Stream child stdout to console and master log.
    Returns (exit_code, child_log_path).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    child_log_path = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        master_fp.write(line)
        match = LOG_LINE_RE.match(line.strip())
        if match and child_log_path is None:
            child_log_path = match.group(1).strip()
    proc.wait()
    return int(proc.returncode), child_log_path


def _write_manifest_row(fp, row: Dict) -> None:
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    fp.flush()


def build_glue_commands(args) -> List[Dict]:
    glue_tasks = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli"]
    methods = ["normal", "fedex", "ffa", "gp_lora"]

    jobs = []
    for task in glue_tasks:
        for agg in methods:
            base = [
                args.python,
                "fed_train_glue.py",
                "--model",
                args.glue_model,
                "--task",
                task,
                "--agg_type",
                agg,
                "--num_clients",
                str(args.num_clients),
                "--lora_r",
                str(args.lora_r),
                "--rounds",
                str(args.rounds),
                "--lr",
                str(args.lr),
                "--local_epochs",
                str(args.local_epochs),
                "--batch_size",
                str(args.batch_size),
            ]
            if args.print_partition_stats:
                base.append("--print_partition_stats")
            if args.pfl_eval_split:
                base.extend(["--pfl_eval_split", args.pfl_eval_split])

            jobs.append(
                {
                    "kind": "glue",
                    "task": task,
                    "agg_type": agg,
                    "partition": "iid",
                    "dirichlet_alpha": None,
                    "cmd": base + ["--partition", "iid"],
                }
            )

            for alpha in args.dirichlet_alphas:
                jobs.append(
                    {
                        "kind": "glue",
                        "task": task,
                        "agg_type": agg,
                        "partition": "dirichlet",
                        "dirichlet_alpha": float(alpha),
                        "cmd": base
                        + [
                            "--partition",
                            "dirichlet",
                            "--dirichlet_alpha",
                            str(alpha),
                        ],
                    }
                )
    return jobs


def build_e2e_commands(args) -> List[Dict]:
    methods = ["normal", "fedex", "ffa", "gp_lora"]
    jobs = []
    for agg in methods:
        cmd = [
            args.python,
            "fed_train_e2e.py",
            "--agg_type",
            agg,
            "--rounds",
            str(args.e2e_rounds),
            "--num_clients",
            str(args.num_clients),
            "--local_epochs",
            str(args.e2e_local_epochs),
            "--lr",
            str(args.e2e_lr),
            "--lora_r",
            str(args.lora_r),
            "--lora_alpha",
            str(args.e2e_lora_alpha),
            "--batch_size",
            str(args.e2e_batch_size),
        ]
        if args.e2e_log_text:
            cmd.append("--log")
        jobs.append(
            {
                "kind": "e2e",
                "task": "e2e",
                "agg_type": agg,
                "partition": "iid",
                "dirichlet_alpha": None,
                "cmd": cmd,
            }
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run experiments in bulk.")
    parser.add_argument(
        "--python", type=str, default=sys.executable, help="Python executable"
    )
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="", help="e.g. 0 or 0,1"
    )

    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=4)

    parser.add_argument("--glue_model", type=str, default="roberta-base")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--pfl_eval_split",
        type=str,
        default="global_val",
        choices=["global_val", "client_val"],
    )
    parser.add_argument("--print_partition_stats", action="store_true")
    parser.add_argument(
        "--dirichlet_alphas",
        type=str,
        default="0.1,0.5,1.0",
        help="Comma-separated list for non-IID runs.",
    )

    parser.add_argument("--e2e_rounds", type=int, default=6)
    parser.add_argument("--e2e_local_epochs", type=int, default=5)
    parser.add_argument("--e2e_lr", type=float, default=2e-3)
    parser.add_argument("--e2e_batch_size", type=int, default=8)
    parser.add_argument("--e2e_lora_alpha", type=int, default=32)
    parser.add_argument(
        "--e2e_log_text",
        action="store_true",
        help="Also save generations under text_store_new/",
    )

    parser.add_argument("--skip_glue", action="store_true")
    parser.add_argument("--skip_e2e", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.dirichlet_alphas = [
        x.strip() for x in args.dirichlet_alphas.split(",") if x.strip()
    ]

    _ensure_dir("log")
    master_path = os.path.join("log", f"run_script_{_now_tag()}.log")
    manifest_path = os.path.join("log", f"run_script_manifest_{_now_tag()}.jsonl")

    with open(master_path, "w", encoding="utf-8") as master_fp, open(
        manifest_path, "w", encoding="utf-8"
    ) as manifest_fp:
        print(f"[master] log: {master_path}")
        print(f"[master] manifest: {manifest_path}")
        master_fp.write(f"[master] log: {master_path}\n")
        master_fp.write(f"[master] manifest: {manifest_path}\n")

        env = os.environ.copy()
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        jobs: List[Dict] = []
        if not args.skip_glue:
            jobs.extend(build_glue_commands(args))
        if not args.skip_e2e:
            jobs.extend(build_e2e_commands(args))

        for idx, job in enumerate(jobs):
            cmd = job["cmd"]
            print(
                f"\n[master] ({idx + 1}/{len(jobs)}) RUN {job['kind']} "
                f"task={job['task']} agg={job['agg_type']} "
                f"part={job['partition']} alpha={job['dirichlet_alpha']}"
            )
            print(f"[master] cmd: {' '.join(cmd)}")
            master_fp.write(
                f"\n[master] ({idx + 1}/{len(jobs)}) RUN {job['kind']} "
                f"task={job['task']} agg={job['agg_type']} "
                f"part={job['partition']} alpha={job['dirichlet_alpha']}\n"
            )
            master_fp.write(f"[master] cmd: {' '.join(cmd)}\n")
            master_fp.flush()

            if args.dry_run:
                _write_manifest_row(
                    manifest_fp,
                    {
                        **{k: v for k, v in job.items() if k != "cmd"},
                        "cmd": cmd,
                        "status": "dry_run",
                        "exit_code": None,
                        "child_log_path": None,
                    },
                )
                continue

            exit_code, child_log_path = _stream_subprocess(
                cmd, env=env, master_fp=master_fp
            )
            status = "ok" if exit_code == 0 else "failed"
            _write_manifest_row(
                manifest_fp,
                {
                    **{k: v for k, v in job.items() if k != "cmd"},
                    "cmd": cmd,
                    "status": status,
                    "exit_code": exit_code,
                    "child_log_path": child_log_path,
                },
            )
            if exit_code != 0:
                print(f"[master] ERROR: exit_code={exit_code} (continuing)")


if __name__ == "__main__":
    main()

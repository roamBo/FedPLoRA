# FedPLoRA 20260715 大规模补表命令

######### FedPLoRA-OS 大规模 baseline / public data / scale 实验命令-20260715 #########

【命令介绍】

本文件覆盖三类实验：

1. G1：D1 split42 上 SmolLM2-1.7B 的 CORE-8 scale confirmation。
2. G2：FlowerTune-Mixed 公开数据三角，CORE-8 × 3 seeds。
3. G6：D1 IID / alpha=0.1 非 IID sweep，5 方法起步。

【命令目的】

本文件用于补齐投稿主表与外部效度：大模型规模、公开数据、非 IID 鲁棒性。

【命令设置】

```text
代码目录: /data2/minghao/code/FedPLoRA-main
默认环境: conda FedRepo2
默认主模型: /data2/minghao/model/SmolLM2-135M
规模确认模型: /data2/minghao/model/SmolLM2-1.7B
D1 benchmark: domain_benchmark_35c_dir05/seed_{42,43,44}
D2 benchmark: FlowerTune-Mixed, 4 domains × 5 clients = 20 clients
CORE-8: Normal, EcoLoRA, FedSA, FedALT, HydraLoRA, FedLEASE, FedPLoRA-OS(v13a), v13b
正式评测: EVAL_MAX_BATCHES=0
默认 one-shot: ROUNDS=1, LOCAL_EPOCHS=1
LoRA: r=8, alpha=16, dropout=0.05
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

【实验产物位置说明】

```text
run_logs:
/data2/minghao/result/FedPLoRA/<RUN_ID>_seed*/run_logs/

result_logs:
/data2/minghao/result/FedPLoRA/<RUN_ID>_seed*/result_logs/

result_files:
/data2/minghao/result/FedPLoRA/<RUN_ID>_seed*/result_files/

trained checkpoints:
/data2/minghao/model/trained_models_LW/<RUN_ID>_seed*/
```

【实验运行涉及场景】

```text
D1: 35c cross-domain domain SFT + Dirichlet alpha=0.5 non-IID + one-shot LoRA aggregation
D2: FlowerTune public mixed domains + 20 clients + one-shot LoRA aggregation
G6: IID / alpha=0.1 heterogeneity sweep
```

【实验前置命令】

## 0.1 本地同步代码到服务器

```bash
cd /Users/hawaiii/codex/FedPLoRA/FedPLoRA-main
REMOTE=minghao@172.26.191.30 REMOTE_DIR=/data2/minghao/code/FedPLoRA-main \
  bash scripts/RunScripts/sync_code_20260709_to_server.sh
```

## 0.2 服务器基础检查

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_baseline.sh
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/Analysis/summarize_fedplora_results.py \
  scripts/Analysis/write_benchmark_fingerprints.py \
  scripts/DataProcessScripts/build_flowertune_raw.py \
  scripts/DataProcessScripts/build_domain_benchmark_v2.py
```

【实验运行命令】

## 1. G1 smoke：SmolLM2-1.7B CORE-8

说明：先只跑 smoke，确认大模型路径、显存、LoRA target_modules、checkpoint 写入都正常。MMLU 生成式指标当前仓库没有独立脚本，本命令只产出 FedPLoRA 框架内 token-acc / communication / local metrics；MMLU 需后续用 lm-eval 或单独脚本补评。

### 1.1 baseline smoke

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
export MODEL_PATH=/data2/minghao/model/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
source scripts/RunScripts/preflight_20260709_baseline.sh
# source 之后再设；否则会落到 baseline_20260709_smoke_seed42
export SMOKE_RUN_ID=v13_20260715_scale_1p7b_smoke_seed42
refresh_smoke_paths
echo "[check] SMOKE_RUN_ID=$SMOKE_RUN_ID SMOKE_ROOT=$SMOKE_ROOT"
GPU=0 run_sft_smoke smoke_normal normal
GPU=0 run_sft_smoke smoke_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=0 run_sft_smoke smoke_fedsa_lora fedsa_lora
GPU=0 run_sft_smoke smoke_fedalt fedalt
GPU=1 run_sft_smoke smoke_hydralora hydralora
GPU=1 run_sft_smoke smoke_fedlease fedlease
```

### 1.2 ours smoke

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
export MODEL_PATH=/data2/minghao/model/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export SMOKE_RUN_ID_20260713=v13_20260715_scale_1p7b_smoke_seed42
mkdir -p /data2/minghao/result/FedPLoRA/v13_20260715_scale_1p7b_smoke_seed42/run_logs
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh --kind smoke --method smoke_v13a_os_1p7b --agg fedplora_v13a_os --gpu 2 -- --force_retrain > /data2/minghao/result/FedPLoRA/v13_20260715_scale_1p7b_smoke_seed42/run_logs/launch_v13a.log 2>&1 &
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh --kind smoke --method smoke_v13b_os_bonly_1p7b --agg fedplora_v13b_os_bonly --gpu 3 -- --force_retrain > /data2/minghao/result/FedPLoRA/v13_20260715_scale_1p7b_smoke_seed42/run_logs/launch_v13b.log 2>&1 &
```

## 2. G1 正式：SmolLM2-1.7B CORE-8 @ D1 split42

### 2.1 baseline CORE-6

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
export MODEL_PATH=/data2/minghao/model/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export ROUNDS=1
export EVAL_MAX_BATCHES=0
source scripts/RunScripts/preflight_20260709_baseline.sh
# source 之后再设；否则会落到 baseline_20260709_35c_dir05_r10_finaleval_seed42
export RUN_ID_PREFIX=scale_20260715_baseline_1p7b_d1_split42_core8_r1_finaleval
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT"
GPU=0 run_sft_full N7_baseline_normal_1p7b normal
GPU=0 run_sft_full N7_baseline_ecolora_1p7b ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_full N7_baseline_fedsa_lora_1p7b fedsa_lora
GPU=1 run_sft_full N7_baseline_fedalt_1p7b fedalt
GPU=2 run_sft_full N7_baseline_hydralora_1p7b hydralora
GPU=2 run_sft_full N7_baseline_fedlease_1p7b fedlease
```

### 2.2 FedPLoRA-OS/v13a 与 v13b

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
export MODEL_PATH=/data2/minghao/model/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export ROUNDS=1
export EVAL_MAX_BATCHES=0
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
# source 之后再设；否则会落到 v12_20260709_main_..._seed42
export RUN_ID_PREFIX=scale_20260715_ours_1p7b_d1_split42_core8_r1_finaleval
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT"
GPU=3 run_sft_full N7_ours_v13a_os_1p7b fedplora_v13a_os
GPU=3 run_sft_full N7_ours_v13b_os_bonly_1p7b fedplora_v13b_os_bonly
```

## 3. G2 FlowerTune-Mixed 数据构建

说明：FlowerTune-Mixed 是公开数据三角。构建 raw JSONL 需要 Hugging Face `datasets` 能下载四个 `flwrlabs/*` 数据集：

```text
general: flwrlabs/alpaca-gpt4
finance: flwrlabs/fingpt-sentiment-train
medical: flwrlabs/medical-meadow-medical-flashcards
code:    flwrlabs/code-alpaca-20k
```

推荐使用 3.1 的一键构建命令：raw 下载、三 split 切分、20-client 校验、fingerprint 输出串行执行，避免 raw 还没写完就开始切分。若服务器不能联网，用 3.2 在有网机器先生成 raw JSONL，再同步到服务器后运行 3.3 切分。

若需要强制重建已有 FlowerTune benchmark，建议先手动改名备份旧目录，例如 `mv data/domain_benchmark_flowertune_mixed_20c_dir05 data/domain_benchmark_flowertune_mixed_20c_dir05.bak_$(date +%Y%m%d_%H%M%S)`，再运行 3.1 或 3.3。

### 3.1 推荐：服务器一键串行构建 raw + 20c dir05 benchmark

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main

export CODE_DIR=/data2/minghao/code/FedPLoRA-main
export FLOWER_RAW=$CODE_DIR/data/raw_flowertune_mixed.jsonl
export FLOWER_OUT=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05
export FLOWER_BUILD_ROOT=/data2/minghao/result/FedPLoRA/flowertune_20260715_build
export HF_HOME=${HF_HOME:-/data2/minghao/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/minghao/cache/huggingface/datasets}

mkdir -p "$FLOWER_BUILD_ROOT/run_logs" "$FLOWER_BUILD_ROOT/fingerprints" "$HF_DATASETS_CACHE"

nohup bash -lc '
set -euo pipefail
cd "$CODE_DIR"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate FedRepo2

python - <<'"'"'PY'"'"'
missing = []
for name in ["datasets", "sklearn"]:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(
        "[flowertune][error] missing python deps: "
        + ",".join(missing)
        + "\\n请先在 FedRepo2 环境安装：python -m pip install -U datasets scikit-learn"
    )
print("[flowertune] python deps ok: datasets, sklearn")
PY

echo "[flowertune] step1 build raw -> $FLOWER_RAW"
python -u scripts/DataProcessScripts/build_flowertune_raw.py \
  --output_path "$FLOWER_RAW" \
  --cache_dir "$HF_DATASETS_CACHE" \
  --target_per_domain 4000 \
  --seed 42

python - "$FLOWER_RAW" <<'"'"'PY'"'"'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[flowertune][error] missing raw jsonl: {path}")
counts = collections.Counter()
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            counts[str(row.get("domain", ""))] += 1
expected = {"code", "finance", "general", "medical"}
if set(counts) != expected:
    raise SystemExit(f"[flowertune][error] expected domains={sorted(expected)}, got={dict(counts)}")
print(f"[flowertune] raw_ok rows={sum(counts.values())} domains={dict(sorted(counts.items()))}")
PY

echo "[flowertune] step2 build 20c dir05 splits -> $FLOWER_OUT"
for seed in 42 43 44; do
  python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
    --input_jsonl "$FLOWER_RAW" \
    --output_dir "$FLOWER_OUT" \
    --num_clients_per_domain 5 \
    --min_samples_per_client 50 \
    --seed "$seed" \
    --partition dirichlet \
    --dirichlet_alpha 0.5 \
    --subtopic kmeans \
    --n_subtopics 10
done

echo "[flowertune] step3 validate clients and fingerprints"
for seed in 42 43 44; do
  split="$FLOWER_OUT/seed_${seed}"
  python - "$split/clients.json" <<'"'"'PY'"'"'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[flowertune][error] missing clients.json: {path}")
clients = json.loads(path.read_text(encoding="utf-8"))
counts = collections.Counter(str(row.get("domain", "")) for row in clients)
if len(clients) != 20:
    raise SystemExit(f"[flowertune][error] expected 20 clients, found {len(clients)} in {path}")
if set(counts) != {"code", "finance", "general", "medical"} or set(counts.values()) != {5}:
    raise SystemExit(f"[flowertune][error] expected 4 domains x 5 clients, found {dict(counts)}")
print(f"[flowertune] split_ok path={path.parent} clients={len(clients)} domains={dict(sorted(counts.items()))}")
PY
  python utilities/benchmark_fingerprint.py "$split" \
    --output "$FLOWER_BUILD_ROOT/fingerprints/seed_${seed}.json"
done

echo "[flowertune][done] raw=$FLOWER_RAW"
echo "[flowertune][done] benchmark=$FLOWER_OUT/seed_{42,43,44}"
' > "$FLOWER_BUILD_ROOT/run_logs/build_flowertune_mixed_all.log" 2>&1 &

echo "[flowertune] pid=$!"
echo "[flowertune] log=$FLOWER_BUILD_ROOT/run_logs/build_flowertune_mixed_all.log"
```

查看构建日志：

```bash
tail -f /data2/minghao/result/FedPLoRA/flowertune_20260715_build/run_logs/build_flowertune_mixed_all.log
```

### 3.2 无网备选：在有网机器先构建 raw JSONL

在有网机器的项目目录运行：

```bash
cd /path/to/FedPLoRA-main
python -m pip install -U datasets scikit-learn
python -u scripts/DataProcessScripts/build_flowertune_raw.py \
  --output_path data/raw_flowertune_mixed.jsonl \
  --target_per_domain 4000 \
  --seed 42
```

同步 raw 到服务器：

```bash
rsync -avP /path/to/FedPLoRA-main/data/raw_flowertune_mixed.jsonl \
  minghao@172.26.191.30:/data2/minghao/code/FedPLoRA-main/data/raw_flowertune_mixed.jsonl
```

### 3.3 仅切分：服务器已有 raw JSONL 时运行

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main

export CODE_DIR=/data2/minghao/code/FedPLoRA-main
export FLOWER_RAW=$CODE_DIR/data/raw_flowertune_mixed.jsonl
export FLOWER_OUT=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05
export FLOWER_BUILD_ROOT=/data2/minghao/result/FedPLoRA/flowertune_20260715_build
mkdir -p "$FLOWER_BUILD_ROOT/run_logs" "$FLOWER_BUILD_ROOT/fingerprints"

nohup bash -lc '
set -euo pipefail
cd "$CODE_DIR"

for seed in 42 43 44; do
  python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
    --input_jsonl "$FLOWER_RAW" \
    --output_dir "$FLOWER_OUT" \
    --num_clients_per_domain 5 \
    --min_samples_per_client 50 \
    --seed "$seed" \
    --partition dirichlet \
    --dirichlet_alpha 0.5 \
    --subtopic kmeans \
    --n_subtopics 10
  split="$FLOWER_OUT/seed_${seed}"
  python utilities/benchmark_fingerprint.py "$split" \
    --output "$FLOWER_BUILD_ROOT/fingerprints/seed_${seed}.json"
done
echo "[flowertune][done] benchmark=$FLOWER_OUT/seed_{42,43,44}"
' > "$FLOWER_BUILD_ROOT/run_logs/build_flowertune_mixed_benchmark_only.log" 2>&1 &

echo "[flowertune] pid=$!"
echo "[flowertune] log=$FLOWER_BUILD_ROOT/run_logs/build_flowertune_mixed_benchmark_only.log"
```

### 3.4 构建完成后的必查命令

```bash
cd /data2/minghao/code/FedPLoRA-main
python - <<'PY'
import collections
import json
import pathlib

root = pathlib.Path("data/domain_benchmark_flowertune_mixed_20c_dir05")
for seed in (42, 43, 44):
    path = root / f"seed_{seed}" / "clients.json"
    if not path.is_file():
        raise SystemExit(f"[flowertune][error] missing {path}")
    clients = json.loads(path.read_text(encoding="utf-8"))
    counts = collections.Counter(str(row["domain"]) for row in clients)
    print(f"seed_{seed}: clients={len(clients)} domains={dict(sorted(counts.items()))}")
    assert len(clients) == 20
    assert set(counts.values()) == {5}
print("[flowertune][ok] FlowerTune-Mixed 20c dir05 benchmark is ready.")
PY
```

## 4. G2 FlowerTune-Mixed smoke

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
export MODEL_PATH=/data2/minghao/model/SmolLM2-135M
export RUN_TAG_MODEL=SmolLM2-135M
export RESULT_ROOT=/data2/minghao/result/FedPLoRA
export MODEL_ROOT=/data2/minghao/model/trained_models_LW
export FLOWER_ROOT=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05
export FLOWER_SMOKE_ROOT=$RESULT_ROOT/flowertune_20260715_smoke_seed42
mkdir -p "$FLOWER_SMOKE_ROOT/run_logs" "$FLOWER_SMOKE_ROOT/result_logs" "$FLOWER_SMOKE_ROOT/result_files/client_states" "$MODEL_ROOT/flowertune_20260715_smoke_seed42"

run_flower_smoke () {
  local gpu="$1"; local method="$2"; local agg="$3"; shift 3
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$FLOWER_ROOT/seed_42" \
    --num_clients 20 \
    --agg_type "$agg" \
    --rounds 1 --local_epochs 1 --lr 0.0002 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --batch_size 2 --max_seq_length 256 \
    --torch_dtype bfloat16 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --client_state_dir "$FLOWER_SMOKE_ROOT/result_files/client_states/$method" \
    --metrics_output_dir "$FLOWER_SMOKE_ROOT/result_logs/$method" \
    --save_run_checkpoint_dir "$MODEL_ROOT/flowertune_20260715_smoke_seed42/${method}_smoke" \
    --trained_models_root "$MODEL_ROOT/flowertune_20260715_smoke_seed42" \
    --eval_max_batches 1 --seed 42 \
    --train_max_steps_per_client 1 \
    --max_train_samples_per_client 10 \
    --save_client_state_to_disk \
    --gradient_checkpointing \
    --eval_personalization_metrics \
    --eval_final_only \
    --skip_post_agg_snapshots \
    "$@" \
    > "$FLOWER_SMOKE_ROOT/run_logs/test20260715_flower_smoke_${method}.log" 2>&1 &
}

run_flower_smoke 1 N9_flower_smoke_normal normal
run_flower_smoke 1 N9_flower_smoke_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
run_flower_smoke 1 N9_flower_smoke_fedsa_lora fedsa_lora
run_flower_smoke 1 N9_flower_smoke_fedalt fedalt
run_flower_smoke 1 N9_flower_smoke_hydralora hydralora
run_flower_smoke 1 N9_flower_smoke_fedlease fedlease
run_flower_smoke 1 N9_flower_smoke_v13a fedplora_v13a_os
run_flower_smoke 1 N9_flower_smoke_v13b fedplora_v13b_os_bonly
```

## 5. G2 FlowerTune-Mixed 正式 CORE-8 × 3 seeds

说明：本段每个 `run_flower_full` 调用都是一个独立后台实验。先跑 seed42，确认 metrics JSON 正常后再跑 seed43/44。

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_baseline.sh
export FLOWER_ROOT=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05
export RESULT_ROOT=/data2/minghao/result/FedPLoRA
export MODEL_ROOT=/data2/minghao/model/trained_models_LW
export MODEL_PATH=/data2/minghao/model/SmolLM2-135M
export RUN_TAG_MODEL=SmolLM2-135M

run_flower_full () {
  local seed="$1"; local gpu="$2"; local method="$3"; local agg="$4"; shift 4
  local run_id="flowertune_20260715_core8_seed${seed}"
  local run_root="$RESULT_ROOT/$run_id"
  local model_root="$MODEL_ROOT/$run_id"
  mkdir -p "$run_root/run_logs" "$run_root/result_logs/$method" "$run_root/result_files/client_states/$method" "$model_root"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$FLOWER_ROOT/seed_${seed}" \
    --num_clients 20 \
    --agg_type "$agg" \
    --rounds 1 --local_epochs 1 --lr 0.0002 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --batch_size 2 --max_seq_length 256 \
    --torch_dtype bfloat16 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --client_state_dir "$run_root/result_files/client_states/$method" \
    --metrics_output_dir "$run_root/result_logs/$method" \
    --save_run_checkpoint_dir "$model_root/${method}_SmolLM2-135M_flowertune_r1_seed${seed}" \
    --trained_models_root "$model_root" \
    --eval_max_batches 0 --seed "$seed" \
    --save_client_state_to_disk \
    --gradient_checkpointing \
    --eval_personalization_metrics \
    --eval_final_only \
    --skip_post_agg_snapshots \
    "$@" \
    > "$run_root/run_logs/test20260715_flower_${method}_seed${seed}.log" 2>&1 &
}

for seed in 42 43 44; do
  run_flower_full "$seed" 0 N9_flower_normal normal
  run_flower_full "$seed" 0 N9_flower_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
  run_flower_full "$seed" 1 N9_flower_fedsa_lora fedsa_lora
  run_flower_full "$seed" 1 N9_flower_fedalt fedalt
  run_flower_full "$seed" 2 N9_flower_hydralora hydralora
  run_flower_full "$seed" 2 N9_flower_fedlease fedlease
  run_flower_full "$seed" 3 N7_ours_flower_v13a fedplora_v13a_os
  run_flower_full "$seed" 3 N7_ours_flower_v13b fedplora_v13b_os_bonly
done
```

## 6. G6 非 IID sweep：构建 IID 与 alpha=0.1

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_baseline.sh
mkdir -p /data2/minghao/result/FedPLoRA/nonIID_20260715_build/run_logs

nohup bash -lc '
set -euo pipefail
python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_DOMAIN_JSONL" \
  --output_dir data/domain_benchmark_35c_iid \
  --num_clients_per_domain 5 \
  --seed 42 \
  --partition iid \
  --subtopic kmeans \
  --n_subtopics 10
python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_DOMAIN_JSONL" \
  --output_dir data/domain_benchmark_35c_dir01 \
  --num_clients_per_domain 5 \
  --seed 42 \
  --partition dirichlet \
  --dirichlet_alpha 0.1 \
  --subtopic kmeans \
  --n_subtopics 10
' > /data2/minghao/result/FedPLoRA/nonIID_20260715_build/run_logs/build_iid_a01.log 2>&1 &
```

## 7. G6 非 IID sweep 正式 5 方法 × 2 变体

说明：起步只跑 Normal/FedSA/FedALT/v13a/v13b，先验证趋势。如果资源充足再补 EcoLoRA/Hydra/FedLEASE。

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_baseline.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0

for variant in iid dir01; do
  if [ "$variant" = "iid" ]; then
    export BENCHMARK_DIR="$CODE_DIR/data/domain_benchmark_35c_iid/seed_42"
  else
    export BENCHMARK_DIR="$CODE_DIR/data/domain_benchmark_35c_dir01/seed_42"
  fi
  export RUN_ID_PREFIX=nonIID_20260715_baseline_${variant}_r1_finaleval
  set_run_paths 42
  GPU=0 run_sft_full N7_baseline_${variant}_normal normal
  GPU=0 run_sft_full N7_baseline_${variant}_fedsa_lora fedsa_lora
  GPU=1 run_sft_full N7_baseline_${variant}_fedalt fedalt
done
```

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0

for variant in iid dir01; do
  if [ "$variant" = "iid" ]; then
    export BENCHMARK_DIR="$CODE_DIR/data/domain_benchmark_35c_iid/seed_42"
  else
    export BENCHMARK_DIR="$CODE_DIR/data/domain_benchmark_35c_dir01/seed_42"
  fi
  export RUN_ID_PREFIX=nonIID_20260715_ours_${variant}_r1_finaleval
  set_run_paths 42
  GPU=2 run_sft_full N7_ours_${variant}_v13a_os fedplora_v13a_os
  GPU=3 run_sft_full N7_ours_${variant}_v13b_os_bonly fedplora_v13b_os_bonly
done
```

## 8. 大规模结果汇总

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_baseline.sh
mkdir -p /data2/minghao/result/FedPLoRA/summary_20260715

python scripts/Analysis/summarize_fedplora_results.py /data2/minghao/result/FedPLoRA \
  --exclude_smoke \
  --strict_fingerprint \
  --output /data2/minghao/result/FedPLoRA/summary_20260715/summary_all_non_smoke.md
```

【注意事项】

1. G1 大模型先跑 smoke；正式命令只有在 smoke 写出 metrics JSON 后再启动。
2. G2 FlowerTune 的公开数据下载可能失败或慢；失败时先同步 `data/raw_flowertune_mixed.jsonl`，不要改算法。
3. 当前仓库没有 MMLU 生成式评测脚本；G1 的 MMLU 是另一个评测环节，不能把 token-acc 当作 MMLU。
4. `summarize_fedplora_results.py --strict_fingerprint` 用于正式出表；如果失败，先补 fingerprint 或排除旧结果，不要手动混表。

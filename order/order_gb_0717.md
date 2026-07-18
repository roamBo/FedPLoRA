# FedPLoRA 20260717：投稿前缺失实验补齐（gb 服务器）

> 由 `order/order_20260717.md` 适配。超参数、算法与实验设计不变；仅修改路径、conda、**全部 GPU=1**（gb 单卡串行）。逐条粘贴后台 nohup，无 `wait`。

######### FedPLoRA 投稿前缺失实验命令（gb 单卡 GPU1 版）-20260717 #########

【命令介绍】

本轮重点不是继续改算法，而是补证据链：

1. P0-G1：FlowerTune-Mixed strict held-out × 3 seeds。
2. P0-G2：SmolLM2-1.7B seeds 43/44 × 4 方法补齐。
3. P0-G3：参照界补齐，centralized-per-domain 7c 与 local-only 复核。
4. P1-G4：nonIID/iid seeds 43/44 补误差棒。
5. P1-G5：mixed-richness 场景复活。
6. P1-G7：v13a sketch rank{1,4} formal 消融。
7. P1-G12/G10：Motivation 与 router reliability 的 0/轻 GPU 诊断。

【命令设置（gb）】

```text
代码目录: /data/yaominghao/gb/FedPLoRA
conda: fedplora
结果目录: /data/yaominghao/gb/result/FedPLoRA
模型 135M: /data/yaominghao/gb/models/SmolLM2-135M
模型 1.7B: /data/yaominghao/gb/models/SmolLM2-1.7B
主数据: domain_benchmark_35c_dir05/seed_{42,43,44}
FlowerTune: domain_benchmark_flowertune_mixed_20c_dir05/seed_{42,43,44}
7c: domain_benchmark_7c/seed_42
训练: rounds=1, local_epochs=1, lr=0.0002
LoRA: r=8, alpha=16, dropout=0.05
eval: EVAL_MAX_BATCHES=0
GPU: 全部物理 1 号卡；单卡同时只跑一个 GPU 实验
```

【路径对照（minghao → gb）】


| 项          | order_20260717.md                         | order_gb_0717.md                                |
| ---------- | ----------------------------------------- | ----------------------------------------------- |
| conda      | `FedRepo2`                                | `fedplora`                                      |
| 代码         | `/data2/minghao/code/FedPLoRA-main`       | `/data/yaominghao/gb/FedPLoRA`                  |
| 135M       | `/data2/minghao/model/SmolLM2-135M`       | `/data/yaominghao/gb/models/SmolLM2-135M`       |
| 1.7B       | `/data2/minghao/model/SmolLM2-1.7B`       | `/data/yaominghao/gb/models/SmolLM2-1.7B`       |
| 结果         | `/data2/minghao/result/FedPLoRA/`         | `/data/yaominghao/gb/result/FedPLoRA/`          |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU        | `0/1` 多卡                                  | 全部 `GPU=1` / `--gpu 1` 串行                       |


【期望 RUN_ID（禁止落到 baseline_20260709_* / v12_20260709_*）】

```text
G1 smoke:    flowertune_20260717_strict_heldout_smoke_seed42
G1 formal:   flowertune_20260717_strict_heldout_seed{42,43,44}
G2 baseline: scale_20260717_baseline_1p7b_d1_core4_r1_finaleval_seed{43,44}
G2 ours:     scale_20260717_ours_1p7b_d1_core4_r1_finaleval_seed{43,44}
G3 local:    ref_20260717_local_only_35c_dir05_10ep_seed{42,43,44}
G3 7c:       ref_20260717_centralized_per_domain_7c_seed42
G4:          nonIID_20260717_{baseline,ours}_{iid,dir01}_r1_finaleval_seed{43,44}
G5:          mixrich_20260717_{baseline,ours}_r1_finaleval_seed{42,43,44}
G7:          v13_20260717_rank_ablation_d1_split42_r1_finaleval_seed42
audit:       audit_20260717/
```

【preflight 防坑（必须遵守）】

```text
1. 禁止同壳连续 source baseline + main。
2. 正确顺序：source 对应 preflight → export RUN_ID_PREFIX → set_run_paths <seed>
   → echo 核对 RUN_ID/RUN_ROOT/BENCHMARK_DIR → 再跑实验。
3. run_sft_* 会 echo RUN_ID= / log=；必须含 20260717，禁止 20260709。
4. FlowerTune held-out 必须 EXPECTED_NUM_CLIENTS=20，并用 --run-id-prefix 显式指定。
5. nohup 子壳不要 source conda.sh + set -u（会 PS1 unbound）；构建用 conda run。
6. 单卡：export GPU=1；行首勿写 GPU=0；for 循环里的 run_sft_full 会后台启动，
   请拆开逐条粘贴，或每条后 wait $!。
```

【实验前置命令】

## 0.1 代码同步

```bash
cd /data/yaominghao/gb/FedPLoRA && git pull
```

确保含：`preflight_20260709_common.sh`（RUN_ID 保留 + `refresh_smoke_paths`）、`run_20260713_one_experiment.sh`、FlowerTune / 7c 数据。

## 0.2 路径与环境（每次新 shell；不要双 source）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA

export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=1

conda activate fedplora
```

---

# 第一部分：P0 必跑

## 1. G1 FlowerTune-Mixed strict held-out × 3 seeds

### 1.1 FlowerTune 20c fingerprint（0 GPU）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export FLOWER_ROOT=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05
export FLOWER_AUDIT_ROOT=/data/yaominghao/gb/result/FedPLoRA/flowertune_20260717_strict_heldout_audit
mkdir -p "$FLOWER_AUDIT_ROOT/run_logs" "$FLOWER_AUDIT_ROOT/fingerprints"

nohup bash -c '
set -eo pipefail
cd "'"$CODE_DIR"'"
export FLOWER_ROOT="'"$FLOWER_ROOT"'"
export FLOWER_AUDIT_ROOT="'"$FLOWER_AUDIT_ROOT"'"
conda run -n fedplora --no-capture-output python - "$FLOWER_ROOT" "$FLOWER_AUDIT_ROOT/fingerprints" <<'"'"'PY'"'"'
import collections, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for seed in (42, 43, 44):
    split = root / f"seed_{seed}"
    clients_path = split / "clients.json"
    if not clients_path.is_file():
        raise SystemExit(f"[flower-strict][error] missing {clients_path}")
    clients = json.loads(clients_path.read_text(encoding="utf-8"))
    counts = collections.Counter(str(c["domain"]) for c in clients)
    if len(clients) != 20 or len(counts) != 4 or set(counts.values()) != {5}:
        raise SystemExit(f"[flower-strict][error] expected 4x5, got {dict(counts)} in {split}")
    fp = out / f"seed_{seed}.json"
    subprocess.run(["python", "utilities/benchmark_fingerprint.py", str(split), "--output", str(fp)], check=True)
    print(f"[flower-strict][ok] seed={seed} clients=20 domains={dict(sorted(counts.items()))} fp={fp}")
PY
' > "$FLOWER_AUDIT_ROOT/run_logs/fingerprint_flowertune20c_3seeds.log" 2>&1 &

echo "pid=$! log=$FLOWER_AUDIT_ROOT/run_logs/fingerprint_flowertune20c_3seeds.log"
```

### 1.2 FlowerTune strict held-out smoke（seed42）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export CONDA_ENV_NAME=fedplora

mkdir -p "$RESULT_ROOT/flowertune_20260717_strict_heldout_smoke/pipeline_logs" \
         "$RESULT_ROOT/flowertune_20260717_strict_heldout_smoke_seed42/run_logs"

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_PATH" \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" \
RAW_DOMAIN_JSONL=$CODE_DIR/data/raw_flowertune_mixed.jsonl \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=1 \
PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_flower_strict_heldout_smoke_seed42 \
  --seed 42 \
  --split-seed 42 \
  --run-id-prefix flowertune_20260717_strict_heldout_smoke \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy first \
     --held_out_offset 0 \
     --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
     --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
     --few_shot_caps 5 \
     --held_out_route_probe_samples 5 \
     --eval_on_local \
     --cold_start \
     --max_steps 1 \
     --v11c_mu 0.4 \
  > "$RESULT_ROOT/flowertune_20260717_strict_heldout_smoke/pipeline_logs/X2_flower_strict_heldout_smoke_seed42.launch.log" 2>&1 &

echo "pid=$!"
echo "launch=$RESULT_ROOT/flowertune_20260717_strict_heldout_smoke/pipeline_logs/X2_flower_strict_heldout_smoke_seed42.launch.log"
echo "expect RUN_ROOT=$RESULT_ROOT/flowertune_20260717_strict_heldout_smoke_seed42"
```

验收：launch 日志里 `[one-exp] RUN_ID_PREFIX=flowertune_20260717_strict_heldout_smoke`，禁止 `v12_20260709`。

### 1.3 FlowerTune strict held-out 正式 3 seeds（串行，逐条粘贴）

说明：gb 单卡**不要** for 循环一次提交 3 个；等上一条 JSON 出齐再贴下一条。把 `SEED` 依次换成 42 / 43 / 44。

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export CONDA_ENV_NAME=fedplora
mkdir -p "$RESULT_ROOT/flowertune_20260717_strict_heldout_launcher/pipeline_logs"

# ===== 改 SEED=42 / 43 / 44，逐条跑 =====
SEED=44

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_PATH" \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" \
RAW_DOMAIN_JSONL=$CODE_DIR/data/raw_flowertune_mixed.jsonl \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=0 \
PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method "X2_flower_strict_heldout_seed${SEED}" \
  --seed "$SEED" \
  --split-seed "$SEED" \
  --run-id-prefix flowertune_20260717_strict_heldout \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy first \
     --held_out_offset 0 \
     --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
     --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
     --few_shot_caps 5,10 \
     --held_out_route_probe_samples 10 \
     --eval_on_local \
     --cold_start \
     --v11c_mu 0.4 \
  > "$RESULT_ROOT/flowertune_20260717_strict_heldout_launcher/pipeline_logs/X2_flower_strict_heldout_seed${SEED}.launch.log" 2>&1 &

echo "pid=$! SEED=$SEED"
echo "expect=$RESULT_ROOT/flowertune_20260717_strict_heldout_seed${SEED}"
```

验收：

```bash
1g/seed
```

```bash
find $RESULT_ROOT/flowertune_20260717_strict_heldout_seed*/result_logs -name '*.json' | sort
```

---

## 2. G2 SmolLM2-1.7B seeds 43/44 × 4 方法

说明：seed42 已有。1.7B **必须串行**；每条 `run_sft_full` 后台起进程，上一条结束后再贴下一条。

### 2.0 1.7B smoke（可选，20260715 已确认可跳过）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export SMOKE_RUN_ID_20260713=v13_20260717_scale_1p7b_smoke_seed42
mkdir -p "$RESULT_ROOT/v13_20260717_scale_1p7b_smoke_seed42/run_logs"

SMOKE_RUN_ID_20260713=v13_20260717_scale_1p7b_smoke_seed42 \
CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
MODEL_PATH="$MODEL_PATH" RUN_TAG_MODEL=SmolLM2-1.7B CONDA_ENV_NAME=fedplora \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind smoke --method smoke_v13a_os_1p7b_20260717 --agg fedplora_v13a_os --gpu 1 -- --force_retrain \
  > "$RESULT_ROOT/v13_20260717_scale_1p7b_smoke_seed42/run_logs/launch_v13a.log" 2>&1 &
```

### 2.1 seed43 baseline 三方法

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export RUN_TAG_DATASET=35c_dir05
export EXPECTED_NUM_CLIENTS=35
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_43
export BENCHMARK_REQUIRED_SPLIT_SEEDS="43"
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export GPU=1

source scripts/RunScripts/preflight_20260709_baseline.sh
# source 之后再设前缀，否则会落到 baseline_20260709_*
export RUN_ID_PREFIX=scale_20260717_baseline_1p7b_d1_core4_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 43
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT BENCHMARK_DIR=$BENCHMARK_DIR"
# 必须看到 scale_20260717_baseline_..._seed43 与 seed_43

run_sft_full N7_baseline_seed43_normal_1p7b normal
# 结束后再贴：
run_sft_full N7_baseline_seed43_fedalt_1p7b fedalt
run_sft_full N7_baseline_seed43_hydralora_1p7b hydralora

15g
```

### 2.2 seed43 v13a

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export RUN_TAG_DATASET=35c_dir05
export EXPECTED_NUM_CLIENTS=35
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_43
export BENCHMARK_REQUIRED_SPLIT_SEEDS="43"
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export GPU=1

source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export RUN_ID_PREFIX=scale_20260717_ours_1p7b_d1_core4_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 43
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT BENCHMARK_DIR=$BENCHMARK_DIR"
# 必须看到 scale_20260717_ours_..._seed43，禁止 v12_20260709

run_sft_full N7_ours_seed43_v13a_os_1p7b fedplora_v13a_os

5g
```

### 2.3 seed44 baseline 三方法

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export RUN_TAG_DATASET=35c_dir05
export EXPECTED_NUM_CLIENTS=35
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_44
export BENCHMARK_REQUIRED_SPLIT_SEEDS="44"
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export GPU=1

source scripts/RunScripts/preflight_20260709_baseline.sh
export RUN_ID_PREFIX=scale_20260717_baseline_1p7b_d1_core4_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 44
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full N7_baseline_seed44_normal_1p7b normal
run_sft_full N7_baseline_seed44_fedalt_1p7b fedalt
run_sft_full N7_baseline_seed44_hydralora_1p7b hydralora
```

### 2.4 seed44 v13a

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-1.7B
export RUN_TAG_MODEL=SmolLM2-1.7B
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export RUN_TAG_DATASET=35c_dir05
export EXPECTED_NUM_CLIENTS=35
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_44
export BENCHMARK_REQUIRED_SPLIT_SEEDS="44"
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export GPU=1

source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export RUN_ID_PREFIX=scale_20260717_ours_1p7b_d1_core4_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 44
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full N7_ours_seed44_v13a_os_1p7b fedplora_v13a_os
```

---

## 3. G3 参照界补齐

### 3.1 local-only clean 3 seeds（可选）

说明：若复用 NX4 的 local，可跳过。gb 单卡：把 `SEED` 换成 42/43/44 逐条跑。

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=1

source scripts/RunScripts/preflight_20260709_baseline.sh
export RUN_ID_PREFIX=ref_20260717_local_only_35c_dir05_10ep
export EVAL_MAX_BATCHES=0

# ===== SEED=42 / 43 / 44 逐条 =====
SEED=44
export BENCHMARK_DIR=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_${SEED}
check_benchmark "$BENCHMARK_DIR"
set_run_paths "$SEED"
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"
mkdir -p "$RUN_ROOT/run_logs" "$RUN_ROOT/result_logs"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --target_modules "$TARGET_MODULES" \
  --torch_dtype "$TORCH_DTYPE" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --lr "$LR" --local_epochs 10 \
  --eval_max_batches "$EVAL_MAX_BATCHES" \
  --seed "$SEED" \
  --schemes local \
  --eval_on_local \
  --out "$RUN_ROOT/result_logs/N6_local_only_10ep_seed${SEED}.json" \
  > "$RUN_ROOT/run_logs/test20260717_ref_local_only_10ep_seed${SEED}.log" 2>&1 &
echo "pid=$! RUN_ROOT=$RUN_ROOT"

2g
```

### 3.2 centralized-per-domain 7c（seed42）

说明：必须用真实 7c；缺数据则先同步 `data/domain_benchmark_7c/seed_42`。

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=1

source scripts/RunScripts/preflight_20260709_baseline.sh
export BENCHMARK_DIR_7C=$CODE_DIR/data/domain_benchmark_7c/seed_42
export RUN_ID_PREFIX=ref_20260717_centralized_per_domain_7c
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT"
mkdir -p "$RUN_ROOT/run_logs" "$RUN_ROOT/result_logs"

python - "$BENCHMARK_DIR_7C/clients.json" <<'PY'
import collections, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"[7c][error] missing clients.json: {p}")
clients = json.loads(p.read_text(encoding="utf-8"))
cnt = collections.Counter(str(c["domain"]) for c in clients)
if len(clients) != 7 or len(cnt) != 7 or set(cnt.values()) != {1}:
    raise SystemExit(f"[7c][error] expected 7 domains x 1 client, got {dict(cnt)}")
print(f"[7c][ok] {p.parent} domains={dict(sorted(cnt.items()))}")
PY

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR_7C" \
  --target_modules "$TARGET_MODULES" \
  --torch_dtype "$TORCH_DTYPE" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --lr "$LR" --local_epochs "$LOCAL_EPOCHS" \
  --eval_max_batches "$EVAL_MAX_BATCHES" \
  --seed "$SEED" \
  --schemes local \
  --eval_on_local \
  --out "$RUN_ROOT/result_logs/N6_centralized_per_domain_7c_seed${SEED}.json" \
  > "$RUN_ROOT/run_logs/test20260717_ref_centralized_per_domain_7c_seed${SEED}.log" 2>&1 &
echo "pid=$!"

2g
```

---

# 第二部分：P1 加分实验

## 4. G4 nonIID/iid seeds 43/44

### 4.1 构建 iid / dir01 seed43/44（0 GPU）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export RAW_DOMAIN_JSONL=${RAW_DOMAIN_JSONL:-$CODE_DIR/data/raw/domain_7_all.jsonl}
mkdir -p /data/yaominghao/gb/result/FedPLoRA/nonIID_20260717_build/run_logs

nohup bash -c '
set -eo pipefail
cd "'"$CODE_DIR"'"
export RAW_DOMAIN_JSONL="'"$RAW_DOMAIN_JSONL"'"
for SEED in 43 44; do
  conda run -n fedplora --no-capture-output python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
    --input_jsonl "$RAW_DOMAIN_JSONL" \
    --output_dir data/domain_benchmark_35c_iid \
    --num_clients_per_domain 5 \
    --seed "$SEED" \
    --partition iid \
    --subtopic kmeans \
    --n_subtopics 10
  conda run -n fedplora --no-capture-output python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
    --input_jsonl "$RAW_DOMAIN_JSONL" \
    --output_dir data/domain_benchmark_35c_dir01 \
    --num_clients_per_domain 5 \
    --seed "$SEED" \
    --partition dirichlet \
    --dirichlet_alpha 0.1 \
    --subtopic kmeans \
    --n_subtopics 10
done
' > /data/yaominghao/gb/result/FedPLoRA/nonIID_20260717_build/run_logs/build_iid_dir01_seed43_44.log 2>&1 &

echo "pid=$!"
```

### 4.2 baseline 三方法（串行拆开粘贴）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=0

source scripts/RunScripts/preflight_20260709_baseline.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0

# 示例：SEED=43 VARIANT=iid；再换 43/dir01、44/iid、44/dir01
SEED=43
VARIANT=dir01
export BENCHMARK_DIR=$CODE_DIR/data/domain_benchmark_35c_${VARIANT}/seed_${SEED}
# dir01 时: export BENCHMARK_DIR=$CODE_DIR/data/domain_benchmark_35c_dir01/seed_${SEED}
export RUN_TAG_DATASET=${VARIANT}
export RUN_ID_PREFIX=nonIID_20260717_baseline_${VARIANT}_r1_finaleval
check_benchmark "$BENCHMARK_DIR"
set_run_paths "$SEED"
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full N7_baseline_${VARIANT}_seed${SEED}_normal normal
run_sft_full N7_baseline_${VARIANT}_seed${SEED}_fedsa_lora fedsa_lora
run_sft_full N7_baseline_${VARIANT}_seed${SEED}_fedalt fedalt

4g
```

### 4.3 v13a/v13b

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=0

source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0

SEED=43
VARIANT=dir01
export BENCHMARK_DIR=$CODE_DIR/data/domain_benchmark_35c_${VARIANT}/seed_${SEED}
export RUN_TAG_DATASET=${VARIANT}
export RUN_ID_PREFIX=nonIID_20260717_ours_${VARIANT}_r1_finaleval
check_benchmark "$BENCHMARK_DIR"
set_run_paths "$SEED"
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full N7_ours_${VARIANT}_seed${SEED}_v13a_os fedplora_v13a_os
run_sft_full N7_ours_${VARIANT}_seed${SEED}_v13b_os_bonly fedplora_v13b_os_bonly
```

---

## 5. G5 mixed-richness

### 5.1 构建 mixrich 3 seeds（0 GPU）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
mkdir -p /data/yaominghao/gb/result/FedPLoRA/mixrich_20260717_build/run_logs

nohup bash -c '
set -eo pipefail
cd "'"$CODE_DIR"'"
for SEED in 42 43 44; do
  conda run -n fedplora --no-capture-output python scripts/DataProcessScripts/build_mixed_richness_benchmark.py \
    --input_benchmark_dir data/domain_benchmark_35c_dir05/seed_${SEED} \
    --output_dir data/domain_benchmark_35c_dir05_mixrich/seed_${SEED} \
    --seed "$SEED" \
    --rich_per_domain 1 \
    --rich_cap 0 \
    --poor_min 20 \
    --poor_max 50
done
' > /data/yaominghao/gb/result/FedPLoRA/mixrich_20260717_build/run_logs/build_mixrich_seed42_43_44.log 2>&1 &
```

### 5.2 baseline 三方法

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=1

source scripts/RunScripts/preflight_20260709_baseline.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=mixrich

# SEED=42 优先；资源够再 43/44
SEED=44
export BENCHMARK_DIR=$CODE_DIR/data/domain_benchmark_35c_dir05_mixrich/seed_${SEED}
export RUN_ID_PREFIX=mixrich_20260717_baseline_r1_finaleval
check_benchmark "$BENCHMARK_DIR"
set_run_paths "$SEED"
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full M3_mixrich_baseline_seed${SEED}_normal normal
run_sft_full M3_mixrich_baseline_seed${SEED}_fedsa_lora fedsa_lora
run_sft_full M3_mixrich_baseline_seed${SEED}_fedalt fedalt

5g
```

### 5.3 v13a/v13b

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=1

source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=mixrich

SEED=44
export BENCHMARK_DIR=$CODE_DIR/data/domain_benchmark_35c_dir05_mixrich/seed_${SEED}
export RUN_ID_PREFIX=mixrich_20260717_ours_r1_finaleval
check_benchmark "$BENCHMARK_DIR"
set_run_paths "$SEED"
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"

# method 名须匹配 main 白名单 M3_os_mixrich_*（已支持 seed 插在中间）
run_sft_full M3_os_mixrich_seed${SEED}_v13a_os fedplora_v13a_os
run_sft_full M3_os_mixrich_seed${SEED}_v13b_os_bonly fedplora_v13b_os_bonly

2.6g
```

---

## 6. G7 v13a sketch rank{1,4} formal

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export GPU=1

source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1
export EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=35c_dir05
export RUN_ID_PREFIX=v13_20260717_rank_ablation_d1_split42_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT"

run_sft_full N7_ours_rank1_v13a_os fedplora_v13a_os --v10_a_sketch_rank 1
run_sft_full N7_ours_rank4_v13a_os fedplora_v13a_os --v10_a_sketch_rank 4

3g
```

---

## 7. G10/G12 诊断

### 7.1 router reliability（0 GPU）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
mkdir -p "$RESULT_ROOT/audit_20260717/run_logs" "$RESULT_ROOT/audit_20260717/analysis"

# 注意：
# 1) 不要用 conda run 传带 | 的 --include（conda 临时脚本会把 | 当管道）
# 2) 脚本同时要求 --output_json 与 --output_md
nohup env PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:$PATH" bash -c '
set -eo pipefail
cd /data/yaominghao/gb/FedPLoRA
python scripts/Analysis/analyze_router_reliability.py /data/yaominghao/gb/result/FedPLoRA \
  --include "v13_2026071|NX|strict_heldout|flowertune_20260717_strict" \
  --output_json /data/yaominghao/gb/result/FedPLoRA/audit_20260717/analysis/router_reliability.json \
  --output_md /data/yaominghao/gb/result/FedPLoRA/audit_20260717/analysis/router_reliability.md
' > "$RESULT_ROOT/audit_20260717/run_logs/router_reliability.log" 2>&1 &

echo "pid=$! log=$RESULT_ROOT/audit_20260717/run_logs/router_reliability.log"
```

### 7.2 A/B subspace motivation（轻 GPU）

说明：若 gb 无 `A100_domain_benchmark_35c_dir05`，改用 `domain_benchmark_35c_dir05/seed_42`（并在笔记中注明非 9k canonical）。

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export DIAG_BENCH=${DIAG_BENCH:-/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05/seed_42}
mkdir -p "$RESULT_ROOT/audit_20260717/run_logs" "$RESULT_ROOT/audit_20260717/analysis"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/diag_subspace_AB.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$DIAG_BENCH" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 \
  --batch_size 2 \
  --max_seq_length 256 \
  --lr 0.0002 \
  --local_epochs 1 \
  --max_steps 0 \
  --seed 42 \
  --n_null 200 \
  --out "$RESULT_ROOT/audit_20260717/analysis/diag_subspace_AB_seed42.json" \
  --save_figs \
  > "$RESULT_ROOT/audit_20260717/run_logs/diag_subspace_AB_seed42.log" 2>&1 &

2g
```

### 7.3 B-swap 诊断（轻 GPU）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export DIAG_BENCH=${DIAG_BENCH:-/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05/seed_42}
mkdir -p "$RESULT_ROOT/audit_20260717/run_logs" "$RESULT_ROOT/audit_20260717/analysis"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/diag_b_swap.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$DIAG_BENCH" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --max_seq_length 256 \
  --lr 0.0002 \
  --local_epochs 1 \
  --max_steps 0 \
  --eval_max_batches 20 \
  --n_peers 4 \
  --n_cross 2 \
  --seed 42 \
  --out "$RESULT_ROOT/audit_20260717/analysis/diag_b_swap_seed42.json" \
  > "$RESULT_ROOT/audit_20260717/run_logs/diag_b_swap_seed42.log" 2>&1 &
```

---

# 第三部分：推荐执行顺序（gb 单卡）

```text
0) git pull + §0.2
1) §1.1 fingerprint → §1.2 smoke → §1.3 held-out 三 seed（串行）
2) §3.2 centralized 7c（P0 参照界）
3) §2 1.7B seed43/44（高成本，夜间串行）
4) §4 构建 → baseline/ours；§5 mixrich；§6 rank
5) §7 诊断（与写作并行；7.2/7.3 占 GPU1）
```

【注意事项】

1. 不再改算法；本文件只补证据链。
2. FlowerTune held-out：`EXPECTED_NUM_CLIENTS=20` + `--run-id-prefix flowertune_20260717_*`。
3. 7c 必须真实 7-client split，不能用 35c 顶替。
4. 每条 `run_sft_*` / launch 日志核对：`RUN_ID` 含 `20260717`，禁止 `20260709`。
5. `set_run_paths <seed>` 会：创建 `RUN_ROOT/.../run_logs`；若 `BENCHMARK_DIR` 以 `/seed_N` 结尾则同步切到对应 seed。
  换异构数据（iid/dir01/mixrich/7c）时仍需手动 `export BENCHMARK_DIR=...`。
6. 与其它 gb order（0715/0712）错开 GPU1。
7. 杀误跑：`pkill -u "$USER" -f "fed_train_sft.py"`；`pkill -u "$USER" -f "run_20260713_one_experiment.sh"`。


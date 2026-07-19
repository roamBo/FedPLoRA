# FedPLoRA 20260718：投稿前实验补强（gb 服务器）

> 由 `order/order_20260718_require.md` 适配。超参数、算法与实验设计不变；仅修改路径、conda、**全部 GPU=1**（gb 单卡串行）。逐条粘贴后台 nohup，无 `wait`。

######### FedPLoRA 投稿前实验补强命令（gb 单卡 GPU1 版）-20260718 #########

【命令目的】

1. FlowerTune 5-fold leave-one-client-out：补 offset1–4 × seeds42/43/44。
2. route-probe 1/2/5-shot 敏感性。
3. A100 9k（若无则回退 dir05）上池化 7c 并重跑 centralized 参照界。
4. A/B subspace 与 B-swap seeds43/44。
5. 同一 v13a 下 B-routing 消融（global / domain）。
6. cold-start paired-Δ/CI 审计。
7. exact-v13 gate 通过后，条件执行 Qwen2.5-3B。

【命令设置（gb）】

```text
代码目录: /data/yaominghao/gb/FedPLoRA
conda: fedplora
结果: /data/yaominghao/gb/result/FedPLoRA
135M: /data/yaominghao/gb/models/SmolLM2-135M
3B: /data/yaominghao/gb/models/Qwen2.5-3B
D1 优先: data/A100_domain_benchmark_35c_dir05/seed_{42,43,44}
D1 回退: data/domain_benchmark_35c_dir05/seed_{42,43,44}
FlowerTune: data/domain_benchmark_flowertune_mixed_20c_dir05/seed_{42,43,44}
rounds=1, local_epochs=1, lr=2e-4
LoRA r=8, alpha=16, dropout=0.05
eval_max_batches=0（formal）/ 1（smoke）
GPU: 全部物理 1 号卡；同时只跑一个 GPU 实验
```

【路径对照（minghao → gb）】


| 项          | order_20260718_require.md                 | order_gb_0718.md                                |
| ---------- | ----------------------------------------- | ----------------------------------------------- |
| conda      | `FedRepo2`                                | `fedplora`                                      |
| 代码         | `/data2/minghao/code/FedPLoRA-main`       | `/data/yaominghao/gb/FedPLoRA`                  |
| 135M       | `/data2/minghao/model/SmolLM2-135M`       | `/data/yaominghao/gb/models/SmolLM2-135M`       |
| 3B         | `/data2/minghao/model/Qwen2.5-3B`         | `/data/yaominghao/gb/models/Qwen2.5-3B`         |
| 结果         | `/data2/minghao/result/FedPLoRA/`         | `/data/yaominghao/gb/result/FedPLoRA/`          |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU        | `0/1/2/3` 多卡并行                            | 全部 `--gpu 1` / `GPU=1` 串行                       |


【期望 RUN_ID（禁止 20260709）】

```text
offset smoke:   flowertune_20260718_offset_smoke_seed42
LOCO:           flowertune_20260718_loco_offset{1..4}_seed{42,43,44}
probe:          flowertune_20260718_probe{1,2,5}_seed42
centralized:    ref_20260718_a1009k_centralized_7c_seed{42,43,44}
route ablation: v13_20260718_route_ablation_{d1,flower}_seed42
qwen3b:         qwen3b_20260718_*
audit:          audit_20260718/
```

【preflight 防坑】

```text
1. 禁止同壳双 source baseline+main。
2. source → export RUN_ID_PREFIX/SMOKE_RUN_ID → set_run_paths/refresh_smoke_paths → echo 核对。
3. FlowerTune：EXPECTED_NUM_CLIENTS=20 + --run-id-prefix 显式指定。
4. 不要用 conda run 传带 | 的参数；nohup 子壳勿 source conda.sh + set -u。
5. 单卡：for/多条 nohup 必须拆开逐条粘贴，上一条出 JSON 再贴下一条。
6. A100 目录不存在时用 D1 回退，并在笔记中注明非 9k canonical。
```

---

# 第一部分：前置与 smoke

## 0. 通用环境（每次新 shell）

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA

export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export GPU=1

# D1：优先 A100 9k，否则回退普通 dir05
if [ -f "$CODE_DIR/data/A100_domain_benchmark_35c_dir05/seed_42/clients.json" ]; then
  export D1_ROOT=$CODE_DIR/data/A100_domain_benchmark_35c_dir05
  export D1_TAG=a1009k_35c_dir05
else
  export D1_ROOT=$CODE_DIR/data/domain_benchmark_35c_dir05
  export D1_TAG=35c_dir05
  echo "[warn] A100 9k missing; fallback D1_ROOT=$D1_ROOT"
fi
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export FLOWER_ROOT=$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05

conda activate fedplora
python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/Analysis/eval_personalized.py \
  scripts/Analysis/diag_subspace_AB.py \
  scripts/Analysis/diag_b_swap.py \
  scripts/Analysis/analyze_router_reliability.py
```

## 0.1 R13 计算环境审计（0 GPU）

```bash
mkdir -p "$RESULT_ROOT/audit_20260718/run_logs"

nohup env PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:$PATH" bash -c '
set -eo pipefail
cd /data/yaominghao/gb/FedPLoRA
date -Is
uname -a
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python --version
python - <<'"'"'PY'"'"'
import importlib
for name in ("torch", "transformers", "peft", "numpy", "scipy", "sklearn"):
    try:
        m = importlib.import_module(name)
        print(f"{name}={getattr(m, \"__version__\", \"unknown\")}")
    except Exception as e:
        print(f"{name}=MISSING ({e})")
PY
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
' > "$RESULT_ROOT/audit_20260718/run_logs/compute_environment.log" 2>&1 &
```

## 0.2 FlowerTune offset smoke（seed42 / offset1）

```bash
mkdir -p "$RESULT_ROOT/flowertune_20260718_offset_smoke/pipeline_logs"

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_PATH" \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$FLOWER_ROOT/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=1 \
PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_flower_offset1_smoke_seed42 \
  --seed 42 --split-seed 42 \
  --run-id-prefix flowertune_20260718_offset_smoke \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy offset --held_out_offset 1 \
     --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
     --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
     --few_shot_caps 5 \
     --held_out_route_probe_samples 2 \
     --eval_on_local --cold_start --max_steps 1 --v11c_mu 0.4 \
  > "$RESULT_ROOT/flowertune_20260718_offset_smoke/pipeline_logs/X2_flower_offset1_smoke_seed42.launch.log" 2>&1 &

echo "pid=$! expect=$RESULT_ROOT/flowertune_20260718_offset_smoke_seed42"
```

Smoke 检查：

```bash
SMOKE_JSON=$RESULT_ROOT/flowertune_20260718_offset_smoke_seed42/result_logs/X2_flower_offset1_smoke_seed42_seed42.json
python - "$SMOKE_JSON" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
assert d["strict_held_out"]["selection_offset"] == 1
assert len(d["strict_held_out"]["held_out_clients"]) == 4
assert d["config"]["eval_max_batches"] == 1
assert d["config"]["max_steps"] == 1
print("[smoke][ok]", p, d.get("protocol_tag"))
PY
```

## 0.3 Qwen2.5-3B 下载/检查（条件）

```bash
mkdir -p /data/yaominghao/gb/models
if [ ! -f /data/yaominghao/gb/models/Qwen2.5-3B/config.json ]; then
  command -v modelscope >/dev/null 2>&1 || pip install -U modelscope
  modelscope download --model Qwen/Qwen2.5-3B --local_dir /data/yaominghao/gb/models/Qwen2.5-3B
fi
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
p = "/data/yaominghao/gb/models/Qwen2.5-3B"
c = AutoConfig.from_pretrained(p)
t = AutoTokenizer.from_pretrained(p, use_fast=False)
print("[3b][ok]", c.model_type, getattr(c, "num_hidden_layers", None), len(t))
PY
```

## 0.4 3B cold-start smoke（条件）

```bash
mkdir -p "$RESULT_ROOT/qwen3b_20260718_coldstart_smoke/pipeline_logs"

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
MODEL_PATH=/data/yaominghao/gb/models/Qwen2.5-3B RUN_TAG_MODEL=Qwen2.5-3B \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$FLOWER_ROOT/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=1 PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_qwen3b_flower_coldstart_smoke_seed42 \
  --seed 42 --split-seed 42 \
  --run-id-prefix qwen3b_20260718_coldstart_smoke \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy first --held_out_offset 0 \
     --schemes base,global,coldstart,coldstart_geom \
     --few_shot_caps 5 \
     --held_out_route_probe_samples 2 \
     --eval_on_local --cold_start --max_steps 1 \
  > "$RESULT_ROOT/qwen3b_20260718_coldstart_smoke/pipeline_logs/X2_qwen3b_flower_coldstart_smoke_seed42.launch.log" 2>&1 &
```

## 0.5 3B Normal/FedALT/v13a smoke（条件，串行）

```bash
export MODEL_PATH=/data/yaominghao/gb/models/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
export GPU=1
source scripts/RunScripts/preflight_20260709_baseline.sh
export SMOKE_RUN_ID=qwen3b_20260718_sft_smoke_seed42
refresh_smoke_paths
echo "[check] SMOKE_RUN_ID=$SMOKE_RUN_ID SMOKE_ROOT=$SMOKE_ROOT"

run_sft_smoke smoke_qwen3b_normal normal --force_retrain
# 结束后：
run_sft_smoke smoke_qwen3b_fedalt fedalt --force_retrain
```

```bash
export MODEL_PATH=/data/yaominghao/gb/models/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export EXPECTED_NUM_CLIENTS=35
export GPU=1
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export SMOKE_RUN_ID=qwen3b_20260718_sft_smoke_seed42
refresh_smoke_paths
echo "[check] SMOKE_RUN_ID=$SMOKE_RUN_ID"

run_sft_smoke smoke_v13a_qwen3b fedplora_v13a_os --force_retrain
```

## 0.6 R16 B-routing 消融 smoke

```bash
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export EXPECTED_NUM_CLIENTS=35
export GPU=1
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export SMOKE_RUN_ID=v13_20260718_route_ablation_smoke_seed42
refresh_smoke_paths
echo "[check] SMOKE_RUN_ID=$SMOKE_RUN_ID"

run_sft_smoke smoke_v13a_route_global fedplora_v13a_os --expert_cluster_mode global --force_retrain
run_sft_smoke smoke_v13a_route_oracle_domain fedplora_v13a_os --expert_cluster_mode domain --force_retrain
```

---

# 第二部分：正式补强

## 1. R1 FlowerTune leave-one-client-out（offset1–4 × 3 seeds）

说明：共 12 条。gb **必须串行**：改 `OFFSET` 与 `SEED` 后整段粘贴，等 JSON 再换下一组。

```bash
mkdir -p "$RESULT_ROOT/flowertune_20260718_loco_launcher/pipeline_logs"

# ===== 改 OFFSET=1..4 与 SEED=42/43/44，逐条跑 =====
OFFSET=1
SEED=42

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_PATH" \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$FLOWER_ROOT/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method "X2_flower_loco_offset${OFFSET}_seed${SEED}" \
  --seed "$SEED" --split-seed "$SEED" \
  --run-id-prefix "flowertune_20260718_loco_offset${OFFSET}" \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy offset --held_out_offset "$OFFSET" \
     --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
     --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
     --few_shot_caps 5,10 \
     --held_out_route_probe_samples 10 \
     --eval_on_local --cold_start --v11c_mu 0.4 \
  > "$RESULT_ROOT/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset${OFFSET}_seed${SEED}.launch.log" 2>&1 &

echo "pid=$! OFFSET=$OFFSET SEED=$SEED"
echo "expect=$RESULT_ROOT/flowertune_20260718_loco_offset${OFFSET}_seed${SEED}"
```

资源不足最小队列：先跑 offset1/2 ×3；完整 5-fold 需补完 offset3/4。

验收：

```bash
find $RESULT_ROOT/flowertune_20260718_loco_offset*_seed*/result_logs -name '*.json' | sort | wc -l
# 完整应为 12
```

## 2. R0b：route-probe 1/2/5-shot（seed42，串行）

```bash
mkdir -p "$RESULT_ROOT/flowertune_20260718_probe_launcher/pipeline_logs"

# ===== PROBE=1 / 2 / 5 逐条 =====
PROBE=1

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_PATH" \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$FLOWER_ROOT/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method "X2_flower_probe${PROBE}_seed42" \
  --seed 42 --split-seed 42 \
  --run-id-prefix "flowertune_20260718_probe${PROBE}" \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy first --held_out_offset 0 \
     --schemes base,global,coldstart,coldstart_geom \
     --few_shot_caps "$PROBE" \
     --held_out_route_probe_samples "$PROBE" \
     --eval_on_local --cold_start \
  > "$RESULT_ROOT/flowertune_20260718_probe_launcher/pipeline_logs/X2_flower_probe${PROBE}_seed42.launch.log" 2>&1 &

echo "pid=$! PROBE=$PROBE"
```

正文须写 **supervised probe examples**，不是 zero-data。

## 3. R5：canonical 7c centralized ×3

### 3.1 从 35c 池化为 7c（0 GPU）

```bash
mkdir -p "$RESULT_ROOT/audit_20260718/run_logs" "$RESULT_ROOT/audit_20260718/analysis"

nohup env PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:$PATH" \
  D1_ROOT="$D1_ROOT" CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" \
  bash -c '
set -eo pipefail
cd "$CODE_DIR"
python - <<'"'"'PY'"'"'
import json, os, pathlib, shutil, tempfile
src_root = pathlib.Path(os.environ["D1_ROOT"])
dst_root = pathlib.Path(os.environ["CODE_DIR"]) / "data" / (
    "A100_domain_benchmark_7c_dir05" if "A100" in str(src_root) else "domain_benchmark_7c_dir05_pooled"
)
dst_root.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

for seed in (42, 43, 44):
    src = src_root / f"seed_{seed}"
    dst = dst_root / f"seed_{seed}"
    if not (src / "clients.json").is_file():
        raise SystemExit(f"[pool7c][error] missing {src / \"clients.json\"}")
    if (dst / "clients.json").is_file():
        clients = json.loads((dst / "clients.json").read_text())
        if len(clients) != 7:
            raise SystemExit(f"[pool7c][error] invalid existing {dst}")
        print(f"[pool7c][skip] {dst}")
        continue
    src_clients = json.loads((src / "clients.json").read_text())
    domains = []
    for row in src_clients:
        dom = str(row["domain"])
        if dom not in domains:
            domains.append(dom)
    if len(src_clients) != 35 or len(domains) != 7:
        raise SystemExit(f"[pool7c][error] expected 35/7 at {src}")
    domain_to_cid = {dom: i for i, dom in enumerate(domains)}
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"seed_{seed}.tmp.", dir=dst_root))
    counts = {dom: {"n_train": 0, "n_val": 0, "n_local_test": 0} for dom in domains}
    for filename, count_key in (("train.jsonl", "n_train"), ("val.jsonl", "n_val"), ("test_local.jsonl", "n_local_test")):
        with (tmp / filename).open("w", encoding="utf-8") as out:
            for row in read_jsonl(src / filename):
                dom = str(row["domain"])
                row["client_id"] = domain_to_cid[dom]
                counts[dom][count_key] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for filename in ("test_domain.jsonl", "test_global.jsonl"):
        shutil.copy2(src / filename, tmp / filename)
    clients = [{"client_id": domain_to_cid[dom], "domain": dom, **counts[dom]} for dom in domains]
    (tmp / "clients.json").write_text(json.dumps(clients, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = json.loads((src / "domain_stats.json").read_text())
    for dom in stats:
        stats[dom]["n_clients"] = 1
    (tmp / "domain_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, dst)
    print(f"[pool7c][ok] {src} -> {dst}")
print("[pool7c][done] dst_root=", dst_root)
PY
' > "$RESULT_ROOT/audit_20260718/run_logs/build_a1009k_7c_3seeds.log" 2>&1 &

echo "pid=$!"
```

构建后设置：

```bash
if [ -d "$CODE_DIR/data/A100_domain_benchmark_7c_dir05/seed_42" ]; then
  export CENTRAL_7C_ROOT=$CODE_DIR/data/A100_domain_benchmark_7c_dir05
else
  export CENTRAL_7C_ROOT=$CODE_DIR/data/domain_benchmark_7c_dir05_pooled
fi
echo "CENTRAL_7C_ROOT=$CENTRAL_7C_ROOT"
```

### 3.2 centralized smoke（seed42）

```bash
export CENTRAL_SMOKE_ROOT=$RESULT_ROOT/ref_20260718_a1009k_centralized_smoke_seed42
mkdir -p "$CENTRAL_SMOKE_ROOT/run_logs" "$CENTRAL_SMOKE_ROOT/result_logs"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$CENTRAL_7C_ROOT/seed_42" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 \
  --lr 0.0002 --local_epochs 1 --max_steps 1 --eval_max_batches 1 \
  --seed 42 --schemes local --eval_on_local \
  --out "$CENTRAL_SMOKE_ROOT/result_logs/N6_a1009k_centralized_smoke_seed42.json" \
  > "$CENTRAL_SMOKE_ROOT/run_logs/N6_a1009k_centralized_smoke_seed42.log" 2>&1 &
```

### 3.3 centralized formal（SEED=42/43/44 逐条）

```bash
SEED=42
export CENTRAL_ROOT=$RESULT_ROOT/ref_20260718_a1009k_centralized_7c_seed${SEED}
mkdir -p "$CENTRAL_ROOT/run_logs" "$CENTRAL_ROOT/result_logs"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$CENTRAL_7C_ROOT/seed_${SEED}" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 \
  --lr 0.0002 --local_epochs 1 --eval_max_batches 0 \
  --seed "$SEED" --schemes local --eval_on_local \
  --out "$CENTRAL_ROOT/result_logs/N6_a1009k_centralized_7c_seed${SEED}.json" \
  > "$CENTRAL_ROOT/run_logs/N6_a1009k_centralized_7c_seed${SEED}.log" 2>&1 &

echo "pid=$! CENTRAL_ROOT=$CENTRAL_ROOT"
```

## 4. R3 Motivation 诊断 seed43/44（串行）

```bash
mkdir -p "$RESULT_ROOT/audit_20260718/analysis" "$RESULT_ROOT/audit_20260718/run_logs"

# ===== SEED=43 然后 44；subspace 完成后再跑同 seed 的 b_swap =====
SEED=43

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/diag_subspace_AB.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$D1_ROOT/seed_${SEED}" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 \
  --lr 0.0002 --local_epochs 1 --max_steps 0 \
  --seed "$SEED" --n_null 200 \
  --out "$RESULT_ROOT/audit_20260718/analysis/diag_subspace_AB_seed${SEED}.json" \
  --save_figs \
  > "$RESULT_ROOT/audit_20260718/run_logs/diag_subspace_AB_seed${SEED}.log" 2>&1 &
```

```bash
SEED=43
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/diag_b_swap.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$D1_ROOT/seed_${SEED}" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 --batch_size 2 --eval_batch_size 2 --max_seq_length 256 \
  --lr 0.0002 --local_epochs 1 --max_steps 0 --eval_max_batches 20 \
  --n_peers 4 --n_cross 2 --seed "$SEED" \
  --out "$RESULT_ROOT/audit_20260718/analysis/diag_b_swap_seed${SEED}.json" \
  > "$RESULT_ROOT/audit_20260718/run_logs/diag_b_swap_seed${SEED}.log" 2>&1 &
```

注意：`--save_figs` 是 histogram，不是 35×35 heatmap（R0c 代码未完成前不可冒充）。

## 5. R16 B-routing 消融

### 5.1 D1 seed42

```bash
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export EXPECTED_NUM_CLIENTS=35
export GPU=1
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=$D1_TAG
export RUN_ID_PREFIX=v13_20260718_route_ablation_d1
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full N7_ours_v13a_route_global_d1 fedplora_v13a_os --expert_cluster_mode global --force_retrain
run_sft_full N7_ours_v13a_route_oracle_domain_d1 fedplora_v13a_os --expert_cluster_mode domain --force_retrain
```

### 5.2 FlowerTune seed42

```bash
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export BENCHMARK_DIR_MAIN=$FLOWER_ROOT/seed_42
export EXPECTED_NUM_CLIENTS=20
export GPU=1
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=flowertune20c_dir05
export RUN_ID_PREFIX=v13_20260718_route_ablation_flower
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID BENCHMARK_DIR=$BENCHMARK_DIR"

run_sft_full N7_ours_v13a_route_global_flower fedplora_v13a_os --expert_cluster_mode global --force_retrain
run_sft_full N7_ours_v13a_route_oracle_domain_flower fedplora_v13a_os --expert_cluster_mode domain --force_retrain
```

## 6. R9/R10 Qwen2.5-3B（条件，串行）

门槛：§0.3–0.5 smoke 通过；且不抢占 §1 LOCO。

### 6.1 FlowerTune cold-start @3B

```bash
mkdir -p "$RESULT_ROOT/qwen3b_20260718_flower_coldstart_launcher/pipeline_logs"

CODE_DIR="$CODE_DIR" RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
MODEL_PATH=/data/yaominghao/gb/models/Qwen2.5-3B RUN_TAG_MODEL=Qwen2.5-3B \
CONDA_ENV_NAME=fedplora \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=$FLOWER_ROOT/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 \
nohup bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_qwen3b_flower_coldstart_seed42 \
  --seed 42 --split-seed 42 \
  --run-id-prefix qwen3b_20260718_flower_coldstart \
  --gpu 1 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy first --held_out_offset 0 \
     --schemes base,global,coldstart,coldstart_geom,v11c_coldstart \
     --few_shot_caps 5,10 \
     --held_out_route_probe_samples 10 \
     --eval_on_local --cold_start --v11c_mu 0.4 \
  > "$RESULT_ROOT/qwen3b_20260718_flower_coldstart_launcher/pipeline_logs/X2_qwen3b_flower_coldstart_seed42.launch.log" 2>&1 &
```

### 6.2 D1 @3B Normal/FedALT

```bash
export MODEL_PATH=/data/yaominghao/gb/models/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export EXPECTED_NUM_CLIENTS=35
export GPU=1
source scripts/RunScripts/preflight_20260709_baseline.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=$D1_TAG
export RUN_ID_PREFIX=qwen3b_20260718_d1_baseline_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID"

run_sft_full N7_baseline_qwen3b_normal normal --force_retrain
run_sft_full N7_baseline_qwen3b_fedalt fedalt --force_retrain
```

### 6.3 D1 @3B v13a

```bash
export MODEL_PATH=/data/yaominghao/gb/models/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=$D1_ROOT/seed_42
export EXPECTED_NUM_CLIENTS=35
export GPU=1
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0
export RUN_TAG_DATASET=$D1_TAG
export RUN_ID_PREFIX=qwen3b_20260718_d1_ours_r1_finaleval
export BENCHMARK_DIR=$BENCHMARK_DIR_MAIN
set_run_paths 42
echo "[check] RUN_ID=$RUN_ID"

run_sft_full N7_ours_qwen3b_v13a_os fedplora_v13a_os --force_retrain
```

---

# 第三部分：0-GPU 汇总

## 7. paired Δ / bootstrap CI（offset1–4 齐后再跑）

```bash
mkdir -p "$RESULT_ROOT/audit_20260718/analysis" "$RESULT_ROOT/audit_20260718/run_logs"

nohup env PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:$PATH" RESULT_ROOT="$RESULT_ROOT" bash -c '
set -eo pipefail
python - <<'"'"'PY'"'"'
import csv, json, pathlib, random, statistics, os
root = pathlib.Path(os.environ["RESULT_ROOT"])
out_root = root / "audit_20260718" / "analysis"
out_root.mkdir(parents=True, exist_ok=True)
paths = []
paths += sorted(root.glob("flowertune_20260717_strict_heldout_seed*/result_logs/*.json"))
for off in (1, 2, 3, 4):
    paths += sorted(root.glob(f"flowertune_20260718_loco_offset{off}_seed*/result_logs/*.json"))
if len(paths) not in (9, 15):  # partial offset0+1/2=9 or full=15
    print(f"[stats][warn] found {len(paths)} JSONs; full LOCO expects 15")
if len(paths) < 3:
    raise SystemExit(f"[stats][error] too few JSONs: {len(paths)}")
rows = []
for p in paths:
    d = json.loads(p.read_text())
    cfg, strict = d["config"], d["strict_held_out"]
    seed, offset = int(cfg["seed"]), int(strict["selection_offset"])
    global_acc = d["results"]["global"]["per_client_acc"]
    cold_acc = d["results"]["coldstart"]["per_client_acc"]
    geom = d["results"].get("coldstart_geom", {})
    geom_acc = geom.get("per_client_acc", {})
    margins = geom.get("geom_route_margin_by_client", {})
    matches = geom.get("geom_route_oracle_match_by_client", {})
    domains = strict["held_out_domains"]
    for cid, g in global_acc.items():
        rows.append({
            "seed": seed, "offset": offset, "client_id": int(cid),
            "domain": domains[str(cid)],
            "global_acc": float(g), "coldstart_acc": float(cold_acc[cid]),
            "coldstart_geom_acc": float(geom_acc[cid]),
            "delta_oracle_vs_global": float(cold_acc[cid]) - float(g),
            "delta_geom_vs_global": float(geom_acc[cid]) - float(g),
            "route_margin": None if margins.get(cid) is None else float(margins[cid]),
            "route_match": None if cid not in matches else bool(matches[cid]),
        })
csv_path = out_root / "flower_loco_client_paired_delta.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
rng = random.Random(20260718)
by_seed = {}
for row in rows:
    by_seed.setdefault(row["seed"], []).append(row)
seeds = sorted(by_seed)
boot_oracle, boot_geom = [], []
for _ in range(20000):
    sampled = []
    for seed in [rng.choice(seeds) for _ in seeds]:
        seed_rows = by_seed[seed]
        offsets = sorted({r["offset"] for r in seed_rows})
        for off in [rng.choice(offsets) for _ in offsets]:
            cell = [r for r in seed_rows if r["offset"] == off]
            sampled.extend(rng.choice(cell) for _ in cell)
    boot_oracle.append(statistics.fmean(r["delta_oracle_vs_global"] for r in sampled))
    boot_geom.append(statistics.fmean(r["delta_geom_vs_global"] for r in sampled))
def ci(xs):
    xs = sorted(xs); return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]]
report = {
    "protocol_warning": "coldstart=oracle/domain-metadata full-A; coldstart_geom=supervised probe full-A; neither is exact v13a A-sketch",
    "n_json": len(paths), "n_client_evaluations": len(rows),
    "oracle_delta_mean": statistics.fmean(r["delta_oracle_vs_global"] for r in rows),
    "oracle_delta_hier_bootstrap_95ci": ci(boot_oracle),
    "geom_delta_mean": statistics.fmean(r["delta_geom_vs_global"] for r in rows),
    "geom_delta_hier_bootstrap_95ci": ci(boot_geom),
    "route_match_rate": statistics.fmean(float(r["route_match"]) for r in rows if r["route_match"] is not None),
    "route_margin_mean": statistics.fmean(r["route_margin"] for r in rows if r["route_margin"] is not None),
    "source_jsons": [str(p) for p in paths],
}
(out_root / "flower_loco_paired_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("[stats][ok]", csv_path)
PY
' > "$RESULT_ROOT/audit_20260718/run_logs/flower_loco_paired_stats.log" 2>&1 &
```

## 8. JSON / 失败日志总检查

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA")
groups = {
    "flower_loco_new": list(root.glob("flowertune_20260718_loco_offset*_seed*/result_logs/*.json")),
    "centralized_9k": list(root.glob("ref_20260718_a1009k_centralized_7c_seed*/result_logs/*.json")),
    "diag_new": list((root / "audit_20260718" / "analysis").glob("diag_*seed4[34].json")),
    "route_ablation": list(root.glob("v13_20260718_route_ablation_*_seed42/result_logs/**/*.json")),
    "qwen3b": list(root.glob("qwen3b_20260718_*/**/*.json")),
}
for name, paths in groups.items():
    bad = []
    for p in paths:
        try: json.loads(p.read_text())
        except Exception as e: bad.append((str(p), repr(e)))
    print(f"[check] {name}: json={len(paths)} bad={len(bad)}")
print("[hint] flower_loco full=12; partial offset1/2=6; centralized=3; diag=4; route=4")
PY

grep -RIn -E 'Traceback|CUDA out of memory|ModuleNotFoundError|FileNotFoundError|\[.*error\]' \
  "$RESULT_ROOT"/flowertune_20260718_* \
  "$RESULT_ROOT"/ref_20260718_* \
  "$RESULT_ROOT"/audit_20260718/run_logs \
  "$RESULT_ROOT"/qwen3b_20260718_* 2>/dev/null || true
```

---

# 第四部分：BLOCKED（需先改代码）

与原 order 相同，gb 不能靠命令补齐：

- **R0** exact-v13a strict-heldout evaluator  
- **R4** exact per-client Local  
- **R0c** 35×35 pairwise matrix  
- **R14** 真实下游任务指标

---

# 第五部分：gb 推荐顺序

```text
0) §0 环境 + §0.1 审计 + §0.2 offset smoke
1) §1 LOCO（最高优先，串行；可先 offset1/2）
2) §3 池化 7c + centralized；§2 probe；§5 route ablation（错开时段）
3) §4 diag seed43/44（占 GPU，夜间）
4) §7 paired stats（等 LOCO JSON）
5) §6 3B（smoke 通过且不抢 LOCO 后再开）
```

【注意事项】

1. 不重复跑已完成的 offset0；不把 19k `domain_benchmark_7c` 混入 A100 9k 主表。
2. 不把 `coldstart` 写成 label-free；不把 `coldstart_geom` 写成 zero-data。
3. 不把 full-A cold-start 标成 exact v13a A-sketch。
4. 不把 diag histogram 当成 35×35 heatmap。
5. Qwen2.5-3B license 以模型卡为准（Qwen Research License）。
6. 每条启动后核对 `RUN_ID` / launch 日志含 **20260718**，禁止 **20260709**。
7. 与 0715/0717 order 错开 GPU1。


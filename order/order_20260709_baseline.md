# FedPLoRA baseline 专用命令-20260709

######### baseline 运行：Normal/EcoLoRA/FedALT/其它对照 #########

【防误跑说明】

这份文件只给“baseline”使用。
```

本文件允许的聚合器：

```text
normal
ecolora
fedalt
flexlora
flora
ffa
fedsa_lora
hydralora
hilora
fedlease
yoco
feddat
fedplora_oneshot
fedplora_v9_mix_ab
```

核心目标：

```text
1. 跑 OS1 one-shot baseline，尤其 FedALT。【最优先】
2. 跑 r=10 多 seed baseline：Normal/EcoLoRA/v9_mix_ab。
3. 补 yoco/feddat。
4. 跑 N6 local-only@10ep 与 centralized-per-domain 上界。
5. 可选跑 1–3B baseline 和 FlowerTune baseline。
```

【实验前置命令：推荐脚本版】

不要再手动复制大段前置命令；直接在服务器上进入代码目录后 source 脚本。必须用 `source`，否则后续 `run_sft_full` / `run_sft_smoke` 函数不会保留在当前 shell。

```bash
exec bash
cd /data2/minghao/code/FedPLoRA-main
source scripts/RunScripts/preflight_20260709_baseline.sh
```

如果需要覆盖默认路径，可在 `source` 前先设置：

```bash
export CODE_DIR=/data2/minghao/code/FedPLoRA-main
export MODEL_PATH=/data2/minghao/model/SmolLM2-135M
export BENCHMARK_DIR_MAIN=/abs/path/to/domain_benchmark_35c_dir05/seed_42
source "$CODE_DIR/scripts/RunScripts/preflight_20260709_baseline.sh"
```

【1. baseline smoke】

```bash
GPU=0 run_sft_smoke smoke_normal normal
GPU=0 run_sft_smoke smoke_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=0 run_sft_smoke smoke_fedalt fedalt
GPU=0 run_sft_smoke smoke_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm
GPU=0 run_sft_smoke smoke_feddat feddat --feddat_teacher_lambda 0.01
GPU=0 run_sft_smoke smoke_flexlora flexlora
GPU=0 run_sft_smoke smoke_flora flora
GPU=0 run_sft_smoke smoke_ffa ffa
GPU=0 run_sft_smoke smoke_fedsa_lora fedsa_lora
GPU=0 run_sft_smoke smoke_hydralora hydralora
GPU=0 run_sft_smoke smoke_hilora hilora
GPU=0 run_sft_smoke smoke_fedlease fedlease
GPU=0 run_sft_smoke smoke_fedplora_oneshot fedplora_oneshot
GPU=0 run_sft_smoke smoke_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
```

【2. P0-OS1：one-shot baseline 主表】

说明：我这边跑 OS1_v8 / OS1_v11c / OS1_v11a；你这边只跑 baseline就行。

```bash
export ROUNDS=1
export RUN_ID_PREFIX=os_20260709_baseline_35c_dir05_r1_finaleval

for s in 42 43 44; do
  set_run_paths "$s"
  GPU=0 run_sft_full OS1_normal normal
  GPU=0 run_sft_full OS1_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
  GPU=0 run_sft_full OS1_fedalt fedalt
  GPU=1 run_sft_full OS1_flexlora flexlora
  GPU=1 run_sft_full OS1_flora flora
  GPU=1 run_sft_full OS1_ffa ffa
  GPU=1 run_sft_full OS1_fedsa_lora fedsa_lora
  GPU=2 run_sft_full OS1_hydralora hydralora
  GPU=2 run_sft_full OS1_hilora hilora
  GPU=2 run_sft_full OS1_fedlease fedlease
  GPU=2 run_sft_full OS1_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm
  GPU=2 run_sft_full OS1_feddat feddat --feddat_teacher_lambda 0.01
  GPU=3 run_sft_full OS1_fedplora_oneshot fedplora_oneshot
done

export ROUNDS=10
export RUN_ID_PREFIX=baseline_20260709_35c_dir05_r10_finaleval
set_run_paths 42
```

【3. P1-X3：r=10 baseline 最小多 seed】

```bash
for s in 42 43 44; do
  set_run_paths "$s"
  GPU=0 run_sft_full X3_normal normal
  GPU=0 run_sft_full X3_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
  GPU=1 run_sft_full X3_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
done
```

【4. P2：缺失 baseline 补表】

```bash
set_run_paths 42
GPU=2 run_sft_full P2_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm
GPU=2 run_sft_full P2_feddat feddat --feddat_teacher_lambda 0.01
```

【5. P1-M3：mixed-richness baseline 子集】

如果已经构建 `BENCHMARK_DIR_MIXRICH`，直接跑；否则这里也给 builder，注意两边不要同时构建同一路径。

```bash
export MIXRICH_OUTPUT_DIR=${MIXRICH_OUTPUT_DIR:-$CODE_DIR/data/domain_benchmark_35c_dir05_mixrich}

python scripts/DataProcessScripts/build_mixed_richness_benchmark.py \
  --input_benchmark_dir "$BENCHMARK_DIR_MAIN" \
  --output_dir "$MIXRICH_OUTPUT_DIR" \
  --seed 42 \
  --rich_per_domain 2 \
  --poor_min 50 \
  --poor_max 100

export BENCHMARK_DIR_MIXRICH=${BENCHMARK_DIR_MIXRICH:-$MIXRICH_OUTPUT_DIR/seed_42}
```

r=10 baseline：

```bash
export BENCHMARK_DIR="$BENCHMARK_DIR_MIXRICH"
export RUN_ID_PREFIX=baseline_20260709_35c_dir05_mixrich_r10_finaleval
set_run_paths 42
GPU=0 run_sft_full M3_mixrich_normal normal
GPU=0 run_sft_full M3_mixrich_fedalt fedalt

export BENCHMARK_DIR="$BENCHMARK_DIR_MAIN"
export RUN_ID_PREFIX=baseline_20260709_35c_dir05_r10_finaleval
set_run_paths 42
```

one-shot baseline：

```bash
export BENCHMARK_DIR="$BENCHMARK_DIR_MIXRICH"
export ROUNDS=1
export RUN_ID_PREFIX=os_20260709_baseline_35c_dir05_mixrich_r1_finaleval
set_run_paths 42
GPU=0 run_sft_full M3_os_mixrich_normal normal
GPU=0 run_sft_full M3_os_mixrich_fedalt fedalt
GPU=1 run_sft_full M3_os_mixrich_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin

export BENCHMARK_DIR="$BENCHMARK_DIR_MAIN"
export ROUNDS=10
export RUN_ID_PREFIX=baseline_20260709_35c_dir05_r10_finaleval
set_run_paths 42
```

【6. P1-N6：local 公平对比 + centralized-per-domain 上界】

local-only @ 10 epochs：

```bash
set_run_paths 42
GPU=3 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR_MAIN" \
  --target_modules "$TARGET_MODULES" \
  --torch_dtype "$TORCH_DTYPE" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --lr "$LR" --local_epochs 10 \
  --eval_max_batches "$EVAL_MAX_BATCHES" \
  --seed "$SEED" \
  --schemes local \
  --eval_on_local \
  --out "$RUN_ROOT/result_logs/N6_local_only_10ep_seed${SEED}.json" \
  > "$RUN_ROOT/run_logs/test20260709_baseline_N6_local_only_10ep_seed${SEED}.log" 2>&1 &
```

centralized-per-domain 7c 上界：

```bash
export BENCHMARK_DIR_7C=${BENCHMARK_DIR_7C:-/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_7c/seed_42}
export RUN_ID_PREFIX=baseline_20260709_7c_centralized_per_domain
set_run_paths 42
GPU=3 nohup python -u scripts/Analysis/eval_personalized.py \
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
  > "$RUN_ROOT/run_logs/test20260709_baseline_N6_centralized_per_domain_7c_seed${SEED}.log" 2>&1 &

export RUN_ID_PREFIX=baseline_20260709_35c_dir05_r10_finaleval
set_run_paths 42
```

【7. 可选：N7 1–3B baseline 点】

```bash
export MODEL_PATH_ORIG="$MODEL_PATH"
export MODEL_PATH=/data2/minghao/model/CHANGE_TO_1B_OR_3B
export ROUNDS=1
export RUN_ID_PREFIX=os_20260709_baseline_35c_dir05_r1_scale_check
set_run_paths 42
GPU=0 run_sft_full N7_baseline_normal normal
GPU=0 run_sft_full N7_baseline_fedalt fedalt
GPU=2 run_sft_full N7_baseline_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm

export MODEL_PATH="$MODEL_PATH_ORIG"
export ROUNDS=10
export RUN_ID_PREFIX=baseline_20260709_35c_dir05_r10_finaleval
set_run_paths 42
```

【8. 可选：N9 FlowerTune-Mixed baseline】

先下载 4 个 FlowerTune 公开训练数据集并构建 4 域 × 5 client = 20c benchmark。第一次运行会从 Hugging Face 下载数据；如果服务器不能联网，先在能联网机器上跑完后把 `data/raw_flowertune_mixed.jsonl` 和 `data/flowertune_mixed_20c/` 同步到服务器。

```bash
python scripts/DataProcessScripts/build_flowertune_raw.py \
  --output_path data/raw_flowertune_mixed.jsonl \
  --target_per_domain 4000 \
  --seed 42

for s in 42 43 44; do
  python scripts/DataProcessScripts/build_domain_benchmark_v2.py \
    --input_jsonl data/raw_flowertune_mixed.jsonl \
    --output_dir data/flowertune_mixed_20c \
    --num_clients_per_domain 5 \
    --seed "$s" \
    --target_per_domain 4000 \
    --partition iid
done

python - <<'PY'
import collections, json, pathlib
for s in (42, 43, 44):
    p = pathlib.Path(f"data/flowertune_mixed_20c/seed_{s}/clients.json")
    clients = json.loads(p.read_text(encoding="utf-8"))
    counts = collections.Counter(row["domain"] for row in clients)
    print(f"[check] {p.parent} clients={len(clients)} domains={dict(sorted(counts.items()))}")
    assert len(clients) == 20 and set(counts.values()) == {5}, counts
PY
```

然后跑 N9 smoke + one-shot baseline：

```bash
export BENCHMARK_DIR_FLOWERTUNE=${BENCHMARK_DIR_FLOWERTUNE:-$CODE_DIR/data/flowertune_mixed_20c/seed_42}

if [ ! -f "$BENCHMARK_DIR_FLOWERTUNE/clients.json" ]; then
  echo "[skip][N9] FlowerTune clients.json not found: $BENCHMARK_DIR_FLOWERTUNE/clients.json"
  echo "[skip][N9] 请先设置正确路径，例如：export BENCHMARK_DIR_FLOWERTUNE=/abs/path/flowertune_mixed/seed_42"
else
  export FLOWERTUNE_NUM_CLIENTS=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$BENCHMARK_DIR_FLOWERTUNE/clients.json")
  if [ "$FLOWERTUNE_NUM_CLIENTS" -le 0 ]; then
    echo "[skip][N9] FLOWERTUNE_NUM_CLIENTS=$FLOWERTUNE_NUM_CLIENTS is invalid"
  else
    export BENCHMARK_DIR_OLD="$BENCHMARK_DIR"
    export EXPECTED_NUM_CLIENTS_OLD="$EXPECTED_NUM_CLIENTS"
    export RUN_TAG_DATASET_OLD="$RUN_TAG_DATASET"
    export ROUNDS_OLD="$ROUNDS"
    export RUN_ID_PREFIX_OLD="$RUN_ID_PREFIX"
    export EVAL_MAX_BATCHES_OLD="$EVAL_MAX_BATCHES"

    export BENCHMARK_DIR="$BENCHMARK_DIR_FLOWERTUNE"
    export EXPECTED_NUM_CLIENTS="$FLOWERTUNE_NUM_CLIENTS"
    export RUN_TAG_DATASET=flowertune_mixed

    # N9 smoke：先确认 FlowerTune-Mixed 数据路径、客户端数、三种 baseline 入口都能跑通。
    export ROUNDS=1
    export RUN_ID_PREFIX=smoke_20260709_baseline_flowertune_mixed
    export EVAL_MAX_BATCHES=1
    if set_run_paths 42; then
      GPU=0 run_sft_full N9_flower_smoke_normal normal --train_max_steps_per_client 1 --max_train_samples_per_client 10
      GPU=0 run_sft_full N9_flower_smoke_fedalt fedalt --train_max_steps_per_client 1 --max_train_samples_per_client 10
      GPU=2 run_sft_full N9_flower_smoke_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm --train_max_steps_per_client 1 --max_train_samples_per_client 10
      echo "[N9-smoke] RUN_ROOT=$RUN_ROOT"
      echo "[N9-smoke] logs: $RUN_ROOT/run_logs/test20260709_baseline_N9_flower_smoke_*_SmolLM2-135M_flowertune_mixed_r1_e1_lr${LR}_seed42.log"
    else
      echo "[skip][N9] set_run_paths failed for FlowerTune smoke; no smoke jobs launched."
    fi

    export EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES_OLD"

    # N9 正式 one-shot baseline。
    export ROUNDS=1
    export RUN_ID_PREFIX=os_20260709_baseline_flowertune_mixed_r1
    if set_run_paths 42; then
      GPU=0 run_sft_full N9_flower_normal normal
      GPU=0 run_sft_full N9_flower_fedalt fedalt
      GPU=2 run_sft_full N9_flower_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm
    else
      echo "[skip][N9] set_run_paths failed for FlowerTune formal runs; no formal jobs launched."
    fi

    export BENCHMARK_DIR="$BENCHMARK_DIR_OLD"
    export EXPECTED_NUM_CLIENTS="$EXPECTED_NUM_CLIENTS_OLD"
    export RUN_TAG_DATASET="$RUN_TAG_DATASET_OLD"
    export ROUNDS="$ROUNDS_OLD"
    export RUN_ID_PREFIX="$RUN_ID_PREFIX_OLD"
    export EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES_OLD"
    set_run_paths 42
  fi
fi
```

【9. 汇总检查】

```bash
python scripts/Analysis/summarize_fedplora_results.py "$RUN_ROOT" --output "$RUN_ROOT/result_logs/summary_baseline_seed${SEED}.md"
grep -R "\[guard\]\\|\[resume\] Run fully complete\\|Traceback\\|CUDA out of memory\\|nan" "$RUN_ROOT/run_logs" || true
```

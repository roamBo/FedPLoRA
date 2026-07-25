# FedPLoRA baseline 缺口实验命令（2026-07-25）

######### FedPLoRA external baseline：主实验、正文实验与附录实验补齐-20260725 #########

> 本文件只安排 baseline 训练、baseline eval-only 和共享主算法输出上的检索对照。FedPLoRA-OS 的训练及 held-out 作业见 `order_Main_20260725.md`。两份文件共享正式协议，但结果根和 checkpoint 根完全分离，避免覆盖。

## 【命令介绍】

1. **主实验：主表可比性与外部有效性**——补 D1/FlowerTune baseline 的 Worst In-Domain（原编号 1、10），补 FlowerTune 表中 `--` 的方法，优先 YOCO（原编号 2），并运行 Normal/FedALT/HydraLoRA 官方任务评估（原编号 9）。
2. **正文实验：替代解释与参与者规模**——从主方法同一 held-out 作业提取 nearest-training-client retrieval（原编号 4），补 70-client Normal/FedALT（原编号 6）。
3. **附录实验：异构与容量对照**——补 common-test `alpha=0.5` 的 Normal/FedSA-LoRA/FedALT（原编号 5）、`r=16` 的 Normal/FedALT（原编号 7），并审计 D1 strict held-out 最近客户端对照（原编号 8）。

## 【作业边界】

| 缺口 | 本文件负责 | 不在本文件重复 |
|---|---|---|
| Worst In-Domain | D1 12 baselines；Flower 既有 6 baselines；新补 Flower baselines | FedPLoRA-OS 6 jobs |
| FlowerTune `--` | FFA-LoRA、FLoRA、FlexLoRA、FedDAT、YOCO、HiLoRA × 3 seeds | 已有 Normal/Eco/FedSA/FedALT/Hydra/FedLEASE |
| 外部任务 | Normal、FedALT、HydraLoRA × 3 seeds | FedPLoRA-OS × 3 seeds |
| routing retrieval | 读取主方法 15+15 个 held-out JSON，零 GPU 汇总 | 不再启动相同 held-out 训练 |
| 70 clients | Normal、FedALT × 3 seeds | FedPLoRA-OS × 3 seeds |
| Non-IID alpha=0.5 | Normal、FedSA-LoRA、FedALT × 3 seeds | FedPLoRA-OS × 3 seeds |
| r=16 | Normal、FedALT × 3 seeds | FedPLoRA-OS × 3 seeds |

FedP-OneShot 按既定决定排除，不加入任何列表。

## 【统一设置与产物】

```text
代码: /data2/minghao/code/FedPLoRA-main
环境: FedRepo2
模型: SmolLM2-135M
rounds=1, local_epochs=1, lr=2e-4
LoRA r=8/alpha=16；r16 扩展为 r=16/alpha=32
batch=2, max_seq_length=256, bfloat16, full eval
seeds={42,43,44}

baseline result root:
/data2/minghao/result/FedPLoRA/order_baseline_20260725

baseline checkpoint root:
/data2/minghao/model/trained_models_LW/order_baseline_20260725

original-node matched-domain root:
/data/yaominghao/gb/result/FedPLoRA/eval_only_baseline_20260725
```

---

# 0. 共同前置与 baseline launcher

## 0.1 环境、代码与数据门禁

先执行 `order_Main_20260725.md` 的 0.1 脚本同步。然后在 A100 节点执行：

```bash
ssh minghao@172.26.191.30
exec bash
source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate FedRepo2

export CODE_DIR=/data2/minghao/code/FedPLoRA-main
export RESULT_ROOT=/data2/minghao/result/FedPLoRA/order_baseline_20260725
export MAIN_RESULT_ROOT=/data2/minghao/result/FedPLoRA/order_main_20260725
export MODEL_ROOT=/data2/minghao/model/trained_models_LW/order_baseline_20260725
export MODEL_135M=/data2/minghao/model/SmolLM2-135M
export D1_ROOT="$CODE_DIR/data/A100_domain_benchmark_35c_dir05"
export FLOWER_ROOT="$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05"
export D1_70C_ROOT="$CODE_DIR/data/A100_domain_benchmark_70c_dir05_frozen_test"
export IID_ROOT="$CODE_DIR/data/domain_benchmark_35c_iid"
export DIR01_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir01"
export DIR05_COMMON_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir05_common_test_v2"
export GPU_ID=${GPU_ID:-0}

cd "$CODE_DIR"
mkdir -p "$RESULT_ROOT/launcher_logs" "$RESULT_ROOT/pids" "$RESULT_ROOT/analysis" "$MODEL_ROOT"
python -m py_compile tasks/fed_train_sft.py \
  scripts/Analysis/checkpoint_manifest.py \
  scripts/Analysis/run_external_lm_eval.py \
  scripts/Analysis/summarize_matched_domain_eval.py
bash -n scripts/RunScripts/run_eval_only_matched_domain.sh
grep -q -- '--max_seq_length "${EVAL_MAX_SEQ_LENGTH}"' scripts/RunScripts/run_eval_only_matched_domain.sh

for SEED in 42 43 44; do
  test -f "$D1_ROOT/seed_${SEED}/clients.json"
  test -f "$FLOWER_ROOT/seed_${SEED}/clients.json"
done
```

## 0.2 baseline 统一 launcher

```bash
launch_baseline_sft () {
  local tag="$1" benchmark="$2" n_clients="$3" agg="$4" seed="$5" rank="$6" alpha="$7" eval_cap="$8"
  shift 8
  local method="N9_${tag}_${agg}"
  mkdir -p "$RESULT_ROOT/$tag/run_logs" \
           "$RESULT_ROOT/$tag/result_logs/$method" \
           "$RESULT_ROOT/$tag/result_files/client_states/$method" \
           "$MODEL_ROOT/$tag/$method"
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
    --model "$MODEL_135M" --benchmark_dir "$benchmark" --num_clients "$n_clients" \
    --agg_type "$agg" --rounds 1 --local_epochs 1 --lr 0.0002 \
    --lora_r "$rank" --lora_alpha "$alpha" --lora_dropout 0.05 \
    --batch_size 2 --max_seq_length 256 --torch_dtype bfloat16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --save_client_state_to_disk --gradient_checkpointing \
    --eval_personalization_metrics --eval_final_only --skip_post_agg_snapshots \
    --client_state_dir "$RESULT_ROOT/$tag/result_files/client_states/$method" \
    --metrics_output_dir "$RESULT_ROOT/$tag/result_logs/$method" \
    --save_run_checkpoint_dir "$MODEL_ROOT/$tag/$method" --trained_models_root "$MODEL_ROOT/$tag" \
    --eval_max_batches "$eval_cap" --seed "$seed" --force_retrain "$@" \
    > "$RESULT_ROOT/$tag/run_logs/test20260725_baseline_${tag}_${agg}.log" 2>&1 &
  echo $! > "$RESULT_ROOT/pids/${tag}_${agg}.pid"
  echo "[launch] tag=$tag agg=$agg pid=$(cat "$RESULT_ROOT/pids/${tag}_${agg}.pid") gpu=${GPU_ID:-0}"
}

baseline_extra () {
  case "$1" in
    yoco) printf '%s\n' '--yoco_sparse_lambda 0.0001 --yoco_pcwa_components 3 --yoco_aggregate_mode conflict --yoco_conflict_method avgm --yoco_sign_lambda 0.01' ;;
    feddat) printf '%s\n' '--feddat_teacher_lambda 0.01' ;;
    hilora) printf '%s\n' '--hilora_leaf_blend 0.25' ;;
    *) printf '\n' ;;
  esac
}
```

## 0.3 smoke

YOCO 覆盖最复杂的 baseline 聚合路径，先用它 smoke：

```bash
read -r -a YOCO_EXTRA <<< "$(baseline_extra yoco)"
GPU_ID=0 launch_baseline_sft smoke_flower_seed42 "$FLOWER_ROOT/seed_42" 20 yoco 42 8 16 1 \
  --train_max_steps_per_client 1 --max_train_samples_per_client 10 "${YOCO_EXTRA[@]}"
```

只有日志无 traceback、checkpoint meta 为 final 且写出 metrics JSON 后，才运行正式 baseline。

---

# 第一部分：主实验——主表可比性与任务级外部有效性

## 1.1 【原编号 2】FlowerTune-Mixed 表中缺失 baseline

**作用：** 在 fingerprint `86603887` 的正式 FlowerTune-Mixed split 上补齐当前 `--`。YOCO 为 P0；FFA/FLoRA/FlexLoRA/FedDAT/HiLoRA 为 P1，若算力不足，主表仍保留 `--`，绝不能复制历史协议结果。

### 1.1.1 YOCO，3 seeds（P0）

```bash
read -r -a YOCO_EXTRA <<< "$(baseline_extra yoco)"
GPU_ID=0 launch_baseline_sft flower_yoco_seed42 "$FLOWER_ROOT/seed_42" 20 yoco 42 8 16 0 "${YOCO_EXTRA[@]}"
GPU_ID=1 launch_baseline_sft flower_yoco_seed43 "$FLOWER_ROOT/seed_43" 20 yoco 43 8 16 0 "${YOCO_EXTRA[@]}"
GPU_ID=2 launch_baseline_sft flower_yoco_seed44 "$FLOWER_ROOT/seed_44" 20 yoco 44 8 16 0 "${YOCO_EXTRA[@]}"
```

### 1.1.2 其余五个缺失 baseline，3 seeds（P1）

```bash
GPU_ID=0 launch_baseline_sft flower_ffa_seed42 "$FLOWER_ROOT/seed_42" 20 ffa 42 8 16 0
GPU_ID=1 launch_baseline_sft flower_ffa_seed43 "$FLOWER_ROOT/seed_43" 20 ffa 43 8 16 0
GPU_ID=2 launch_baseline_sft flower_ffa_seed44 "$FLOWER_ROOT/seed_44" 20 ffa 44 8 16 0

GPU_ID=3 launch_baseline_sft flower_flora_seed42 "$FLOWER_ROOT/seed_42" 20 flora 42 8 16 0
GPU_ID=4 launch_baseline_sft flower_flora_seed43 "$FLOWER_ROOT/seed_43" 20 flora 43 8 16 0
GPU_ID=5 launch_baseline_sft flower_flora_seed44 "$FLOWER_ROOT/seed_44" 20 flora 44 8 16 0

GPU_ID=6 launch_baseline_sft flower_flexlora_seed42 "$FLOWER_ROOT/seed_42" 20 flexlora 42 8 16 0
GPU_ID=7 launch_baseline_sft flower_flexlora_seed43 "$FLOWER_ROOT/seed_43" 20 flexlora 43 8 16 0
wait

GPU_ID=0 launch_baseline_sft flower_flexlora_seed44 "$FLOWER_ROOT/seed_44" 20 flexlora 44 8 16 0

read -r -a FEDDAT_EXTRA <<< "$(baseline_extra feddat)"
GPU_ID=1 launch_baseline_sft flower_feddat_seed42 "$FLOWER_ROOT/seed_42" 20 feddat 42 8 16 0 "${FEDDAT_EXTRA[@]}"
GPU_ID=2 launch_baseline_sft flower_feddat_seed43 "$FLOWER_ROOT/seed_43" 20 feddat 43 8 16 0 "${FEDDAT_EXTRA[@]}"
GPU_ID=3 launch_baseline_sft flower_feddat_seed44 "$FLOWER_ROOT/seed_44" 20 feddat 44 8 16 0 "${FEDDAT_EXTRA[@]}"

read -r -a HILORA_EXTRA <<< "$(baseline_extra hilora)"
GPU_ID=4 launch_baseline_sft flower_hilora_seed42 "$FLOWER_ROOT/seed_42" 20 hilora 42 8 16 0 "${HILORA_EXTRA[@]}"
GPU_ID=5 launch_baseline_sft flower_hilora_seed43 "$FLOWER_ROOT/seed_43" 20 hilora 43 8 16 0 "${HILORA_EXTRA[@]}"
GPU_ID=6 launch_baseline_sft flower_hilora_seed44 "$FLOWER_ROOT/seed_44" 20 hilora 44 8 16 0 "${HILORA_EXTRA[@]}"
```

> 注意：同一 GPU 上必须串行。上面只是给出卡位示例；若前一作业未结束，不得直接启动下一行。

## 1.2 【原编号 1、10】baseline Worst In-Domain

### 1.2.1 原训练节点：D1 12 baselines + Flower 已有 6 baselines

在 `/data/yaominghao/gb/FedPLoRA` 节点执行：

```bash
set -euo pipefail
cd /data/yaominghao/gb/FedPLoRA
export GB_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MD_ROOT="$GB_RESULT_ROOT/eval_only_baseline_20260725"
export MD_RUNNER=FedPLoRA-main/scripts/RunScripts/run_eval_only_matched_domain.sh
export MD_SUMMARIZER=FedPLoRA-main/scripts/Analysis/summarize_matched_domain_eval.py
mkdir -p "$MD_ROOT/d1" "$MD_ROOT/flowertune_existing" "$MD_ROOT/logs" "$MD_ROOT/pids"

D1_METHOD_DIRS=(OS1_normal OS1_ffa OS1_flora OS1_flexlora OS1_ecolora OS1_fedsa_lora OS1_feddat OS1_yoco OS1_fedalt OS1_hydralora OS1_hilora OS1_fedlease)
FLOWER_METHOD_DIRS=(N9_flower_normal N9_flower_ecolora N9_flower_fedsa_lora N9_flower_fedalt N9_flower_hydralora N9_flower_fedlease)

D1_RESULTS=()
for SEED in 42 43 44; do
  BASE="$GB_RESULT_ROOT/os_20260709_baseline_35c_dir05_r1_finaleval_seed${SEED}/result_logs"
  for METHOD in "${D1_METHOD_DIRS[@]}"; do
    mapfile -t HITS < <(find "$BASE/$METHOD" -maxdepth 1 -name '*.json' | sort)
    [[ "${#HITS[@]}" -eq 1 ]]
    D1_RESULTS+=("${HITS[0]}")
  done
done

FLOWER_RESULTS=()
for SEED in 42 43 44; do
  BASE="$GB_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed${SEED}/result_logs"
  for METHOD in "${FLOWER_METHOD_DIRS[@]}"; do
    mapfile -t HITS < <(find "$BASE/$METHOD" -maxdepth 1 -name '*.json' | sort)
    [[ "${#HITS[@]}" -eq 1 ]]
    FLOWER_RESULTS+=("${HITS[0]}")
  done
done

[[ "${#D1_RESULTS[@]}" -eq 36 ]]
[[ "${#FLOWER_RESULTS[@]}" -eq 18 ]]
printf '%s\n' "${D1_RESULTS[@]}" > "$MD_ROOT/d1_source_results.txt"
printf '%s\n' "${FLOWER_RESULTS[@]}" > "$MD_ROOT/flower_source_results.txt"

nohup env CUDA_VISIBLE_DEVICES=0 EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
  EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$MD_ROOT/d1" \
  bash "$MD_RUNNER" "${D1_RESULTS[@]}" > "$MD_ROOT/logs/d1_baselines.log" 2>&1 &
echo $! > "$MD_ROOT/pids/d1_baselines.pid"

nohup env CUDA_VISIBLE_DEVICES=1 EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
  EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$MD_ROOT/flowertune_existing" \
  bash "$MD_RUNNER" "${FLOWER_RESULTS[@]}" > "$MD_ROOT/logs/flower_existing_baselines.log" 2>&1 &
echo $! > "$MD_ROOT/pids/flower_existing_baselines.pid"
```

这里对 eval-only 使用特殊的“每个数据集一个顺序 runner”：D1 的 36 个 checkpoint 在 GPU 0 依次评估，Flower 的 18 个 checkpoint 在 GPU 1 依次评估。这样仍保持每次只载入一个模型，同时避免 54 个进程争抢 GPU；它不包含训练或聚合。

### 1.2.2 A100 节点：新补 Flower baseline 的 matched-domain

必须先完成 1.1 的训练。每个新结果 JSON 单独启动：

```bash
export NEW_MD_ROOT="$RESULT_ROOT/matched_domain_new_flower"
mkdir -p "$NEW_MD_ROOT" "$RESULT_ROOT/launcher_logs" "$RESULT_ROOT/pids"

launch_new_flower_md () {
  local tag="$1" agg="$2" seed="$3" gpu="$4"
  local method="N9_${tag}_${agg}"
  mapfile -t hits < <(find "$RESULT_ROOT/$tag/result_logs/$method" -maxdepth 1 -name '*.json' | sort)
  [[ "${#hits[@]}" -eq 1 ]] || { echo "[new-md][error] $tag" >&2; return 2; }
  nohup env CUDA_VISIBLE_DEVICES="$gpu" EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
    EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$NEW_MD_ROOT" \
    bash scripts/RunScripts/run_eval_only_matched_domain.sh "${hits[0]}" \
    > "$RESULT_ROOT/launcher_logs/test20260725_md_${tag}.log" 2>&1 &
  echo $! > "$RESULT_ROOT/pids/md_${tag}.pid"
}

launch_new_flower_md flower_yoco_seed42 yoco 42 0
launch_new_flower_md flower_yoco_seed43 yoco 43 1
launch_new_flower_md flower_yoco_seed44 yoco 44 2
launch_new_flower_md flower_ffa_seed42 ffa 42 3
launch_new_flower_md flower_ffa_seed43 ffa 43 4
launch_new_flower_md flower_ffa_seed44 ffa 44 5
launch_new_flower_md flower_flora_seed42 flora 42 6
launch_new_flower_md flower_flora_seed43 flora 43 7
wait

launch_new_flower_md flower_flora_seed44 flora 44 0
launch_new_flower_md flower_flexlora_seed42 flexlora 42 1
launch_new_flower_md flower_flexlora_seed43 flexlora 43 2
launch_new_flower_md flower_flexlora_seed44 flexlora 44 3
launch_new_flower_md flower_feddat_seed42 feddat 42 4
launch_new_flower_md flower_feddat_seed43 feddat 43 5
launch_new_flower_md flower_feddat_seed44 feddat 44 6
launch_new_flower_md flower_hilora_seed42 hilora 42 7
wait

launch_new_flower_md flower_hilora_seed43 hilora 43 0
launch_new_flower_md flower_hilora_seed44 hilora 44 1
```

正式汇总前，baseline 加主方法在每个数据集上必须使用相同的 256 runner。D1 应有 36 个 baseline JSON；Flower 已有 18 个 + 新补 18 个。若只先完成 YOCO，则新补部分为 3 个，其余主表继续写 `--`。

## 1.3 【原编号 9】官方任务 baseline：Normal、FedALT 与 HydraLoRA × 3 seeds

**选择理由：** Normal 是共享全局适配器参考；FedALT 在 FlowerTune 上具有很强的 Local；HydraLoRA 是 D1 上最接近主方法的强个性化 baseline。三者与主方法使用相同 task、zero-shot、dtype、batch、adapter export 和解码设置。

### 1.3.1 checkpoint 解析与 adapter export

```bash
export CKPT_SEARCH_ROOTS="/data2/minghao/model/trained_models_LW /data2/minghao/result/FedPLoRA"
export HF_CACHE_ROOT="$CODE_DIR/data/external_lm_eval_hf_cache"
python scripts/Analysis/prepare_external_lm_eval_hf_cache.py --cache_root "$HF_CACHE_ROOT" \
  --tasks mmlu,pubmedqa,mbpp --verify_only

export_baseline_adapter () {
  local agg="$1" seed="$2"
  local ckpt
  ckpt=$(python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS --resolve \
    --agg_type "$agg" --seed "$seed" --model_contains SmolLM2-135M \
    --benchmark_contains "A100_domain_benchmark_35c_dir05/seed_${seed}")
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python -u tasks/fed_train_sft.py \
    --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_${seed}" --agg_type "$agg" --seed "$seed" \
    --eval_only_from_checkpoint "$ckpt" \
    --metrics_output_dir "$RESULT_ROOT/external_export/${agg}_seed${seed}/metrics" \
    --client_state_dir "$RESULT_ROOT/external_export/${agg}_seed${seed}/scratch" \
    --export_eval_adapter_dir "$RESULT_ROOT/external_adapters/${agg}_seed${seed}" \
    --export_eval_adapter_only --eval_max_batches 0 --batch_size 2 --max_seq_length 256 \
    --torch_dtype bfloat16 --eval_personalization_metrics
}

GPU_ID=0 export_baseline_adapter normal 42
GPU_ID=0 export_baseline_adapter normal 43
GPU_ID=0 export_baseline_adapter normal 44
GPU_ID=0 export_baseline_adapter fedalt 42
GPU_ID=0 export_baseline_adapter fedalt 43
GPU_ID=0 export_baseline_adapter fedalt 44
GPU_ID=0 export_baseline_adapter hydralora 42
GPU_ID=0 export_baseline_adapter hydralora 43
GPU_ID=0 export_baseline_adapter hydralora 44
```

### 1.3.2 external smoke 与六个正式 launcher

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Analysis/run_external_lm_eval.py \
  --adapter_manifest "$RESULT_ROOT/external_adapters/fedalt_seed42/adapter_export_manifest.json" \
  --tasks pubmedqa:medical --mode both --limit 10 --device cuda:0 --batch_size auto \
  --hf_cache_dir "$HF_CACHE_ROOT" --output_dir "$RESULT_ROOT/external_smoke/fedalt_seed42"
```

```bash
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/normal_seed42/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode global --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/normal_seed42" > "$RESULT_ROOT/launcher_logs/test20260725_external_normal_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_normal_seed42.pid"
CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/normal_seed43/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode global --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/normal_seed43" > "$RESULT_ROOT/launcher_logs/test20260725_external_normal_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_normal_seed43.pid"
CUDA_VISIBLE_DEVICES=2 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/normal_seed44/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode global --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/normal_seed44" > "$RESULT_ROOT/launcher_logs/test20260725_external_normal_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_normal_seed44.pid"
CUDA_VISIBLE_DEVICES=3 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/fedalt_seed42/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/fedalt_seed42" > "$RESULT_ROOT/launcher_logs/test20260725_external_fedalt_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_fedalt_seed42.pid"
CUDA_VISIBLE_DEVICES=4 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/fedalt_seed43/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/fedalt_seed43" > "$RESULT_ROOT/launcher_logs/test20260725_external_fedalt_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_fedalt_seed43.pid"
CUDA_VISIBLE_DEVICES=5 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/fedalt_seed44/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/fedalt_seed44" > "$RESULT_ROOT/launcher_logs/test20260725_external_fedalt_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_fedalt_seed44.pid"
wait

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/hydralora_seed42/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/hydralora_seed42" > "$RESULT_ROOT/launcher_logs/test20260725_external_hydralora_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_hydralora_seed42.pid"
CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/hydralora_seed43/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/hydralora_seed43" > "$RESULT_ROOT/launcher_logs/test20260725_external_hydralora_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_hydralora_seed43.pid"
CUDA_VISIBLE_DEVICES=2 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/hydralora_seed44/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/hydralora_seed44" > "$RESULT_ROOT/launcher_logs/test20260725_external_hydralora_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_hydralora_seed44.pid"
```

与主方法相同，MBPP 只在隔离环境运行；FiQA 在稳定 task 注册前不纳入正式比较。

---

# 第二部分：正文实验——最近客户端替代解释与 70-client 对照

## 2.1 【原编号 4】nearest-training-client retrieval baseline（零额外 GPU）

主算法文档 2.1 的 15 个 FlowerTune 作业已经在相同 probe state、相同 held-out fold 和相同评估中同时写入 `subspace`（本文专家池路由）与 `nearest_client_subspace`（最近训练客户端）。这里仅做严格配对汇总，禁止重跑一套不同 split。

```bash
python - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("/data2/minghao/result/FedPLoRA/order_main_20260725")
paths = sorted(root.glob("main_20260725_flower_probe1_seed*/result_logs/X2_flower_probe1_offset*_seed*.json"))
assert len(paths) == 15, len(paths)
rows = []
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    audits = (data.get("strict_held_out") or {}).get("route_audits") or {}
    for key in ("subspace", "nearest_client_subspace"):
        assert key in audits, (path, key)
    rows.append({
        "path": str(path),
        "expert_pool_match": audits["subspace"]["summary"]["oracle_match_rate"],
        "nearest_client_match": audits["nearest_client_subspace"]["summary"]["oracle_match_rate"],
        "expert_pool_margin": audits["subspace"]["summary"]["mean_margin"],
        "nearest_client_margin": audits["nearest_client_subspace"]["summary"]["mean_margin"],
    })
out = root / "analysis" / "flower_nearest_client_paired_20260725.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
for key in ("expert_pool_match", "nearest_client_match"):
    vals = [float(x[key]) for x in rows]
    print(key, statistics.mean(vals), statistics.stdev(vals))
print("[retrieval][ok]", out)
PY
```

正式表还应从各 scheme 的 accuracy 字段提取配对 Local 与 wrong-domain count；字段名以 JSON 实际 schema 为准，不用 grep 猜测或手工录入。

## 2.2 【原编号 6】70-client Normal/FedALT × 3 seeds

依赖主算法文档 2.2.1 已构建并通过 `cmp` 的 70-client split：

```bash
for SEED in 42 43 44; do
  cmp -s "$D1_70C_ROOT/seed_${SEED}/test_domain.jsonl" "$D1_ROOT/seed_${SEED}/test_domain.jsonl"
done

GPU_ID=0 launch_baseline_sft 70c_normal_seed42 "$D1_70C_ROOT/seed_42" 70 normal 42 8 16 0
GPU_ID=1 launch_baseline_sft 70c_normal_seed43 "$D1_70C_ROOT/seed_43" 70 normal 43 8 16 0
GPU_ID=2 launch_baseline_sft 70c_normal_seed44 "$D1_70C_ROOT/seed_44" 70 normal 44 8 16 0
GPU_ID=3 launch_baseline_sft 70c_fedalt_seed42 "$D1_70C_ROOT/seed_42" 70 fedalt 42 8 16 0
GPU_ID=4 launch_baseline_sft 70c_fedalt_seed43 "$D1_70C_ROOT/seed_43" 70 fedalt 43 8 16 0
GPU_ID=5 launch_baseline_sft 70c_fedalt_seed44 "$D1_70C_ROOT/seed_44" 70 fedalt 44 8 16 0
```

35c 对照直接复用主表正式三种子；不能把 70c `min_samples_per_client=25` 的结果当作主表替代协议。

---

# 第三部分：附录实验——Non-IID、r16 与 D1 检索复现

## 3.1 【原编号 5】common-test `alpha=0.5` baseline

比较 Normal、FedSA-LoRA、FedALT；依赖主算法文档 3.1.1 构建的 frozen-test split。

```bash
for SEED in 42 43 44; do
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$IID_ROOT/seed_${SEED}/test_domain.jsonl"
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$DIR01_ROOT/seed_${SEED}/test_domain.jsonl"
done

GPU_ID=0 launch_baseline_sft common_a05_normal_seed42 "$DIR05_COMMON_ROOT/seed_42" 35 normal 42 8 16 0
GPU_ID=1 launch_baseline_sft common_a05_normal_seed43 "$DIR05_COMMON_ROOT/seed_43" 35 normal 43 8 16 0
GPU_ID=2 launch_baseline_sft common_a05_normal_seed44 "$DIR05_COMMON_ROOT/seed_44" 35 normal 44 8 16 0
GPU_ID=3 launch_baseline_sft common_a05_fedsa_seed42 "$DIR05_COMMON_ROOT/seed_42" 35 fedsa_lora 42 8 16 0
GPU_ID=4 launch_baseline_sft common_a05_fedsa_seed43 "$DIR05_COMMON_ROOT/seed_43" 35 fedsa_lora 43 8 16 0
GPU_ID=5 launch_baseline_sft common_a05_fedsa_seed44 "$DIR05_COMMON_ROOT/seed_44" 35 fedsa_lora 44 8 16 0
GPU_ID=6 launch_baseline_sft common_a05_fedalt_seed42 "$DIR05_COMMON_ROOT/seed_42" 35 fedalt 42 8 16 0
GPU_ID=7 launch_baseline_sft common_a05_fedalt_seed43 "$DIR05_COMMON_ROOT/seed_43" 35 fedalt 43 8 16 0
wait

GPU_ID=0 launch_baseline_sft common_a05_fedalt_seed44 "$DIR05_COMMON_ROOT/seed_44" 35 fedalt 44 8 16 0
```

只有 IID/0.5/0.1 三档 test 文件逐 seed 按字节相同，才允许画趋势线。

## 3.2 【原编号 7】LoRA `r=16` Normal/FedALT × 3 seeds

```bash
GPU_ID=0 launch_baseline_sft r16_normal_seed42 "$D1_ROOT/seed_42" 35 normal 42 16 32 0
GPU_ID=1 launch_baseline_sft r16_normal_seed43 "$D1_ROOT/seed_43" 35 normal 43 16 32 0
GPU_ID=2 launch_baseline_sft r16_normal_seed44 "$D1_ROOT/seed_44" 35 normal 44 16 32 0
GPU_ID=3 launch_baseline_sft r16_fedalt_seed42 "$D1_ROOT/seed_42" 35 fedalt 42 16 32 0
GPU_ID=4 launch_baseline_sft r16_fedalt_seed43 "$D1_ROOT/seed_43" 35 fedalt 43 16 32 0
GPU_ID=5 launch_baseline_sft r16_fedalt_seed44 "$D1_ROOT/seed_44" 35 fedalt 44 16 32 0
```

## 3.3 【原编号 8】D1 strict held-out 最近客户端复现（零额外 GPU）

读取主算法文档 3.3 的 15 个同协议 JSON：

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("/data2/minghao/result/FedPLoRA/order_main_20260725")
paths = sorted(root.glob("main_20260725_d1_heldout_seed*/result_logs/X2_d1_heldout_offset*_seed*.json"))
assert len(paths) == 15, len(paths)
total_routes = 0
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    held = data.get("strict_held_out") or {}
    audits = held.get("route_audits") or {}
    assert "subspace" in audits and "nearest_client_subspace" in audits, path
    n = audits["subspace"]["summary"]["num_routed"]
    assert n == 7, (path, n)
    total_routes += n
assert total_routes == 105, total_routes
print("[d1-retrieval][ok] fold-seed=15 routes=105")
PY
```

该对照必须与 FlowerTune 一样报告 route match、margin、Local、wrong-domain count、CPU time；不得只报告路由命中率而省略下游 accuracy。

---

# 4. 完整性检查、优先级与剩余阻塞

## 4.1 baseline 新训练完整性

```bash
python - <<'PY'
import pathlib
root = pathlib.Path("/data2/minghao/result/FedPLoRA/order_baseline_20260725")
required = []
required += [f"flower_yoco_seed{s}" for s in (42,43,44)]
required += [f"70c_{m}_seed{s}" for s in (42,43,44) for m in ("normal","fedalt")]
required += [f"common_a05_{m}_seed{s}" for s in (42,43,44) for m in ("normal","fedsa","fedalt")]
required += [f"r16_{m}_seed{s}" for s in (42,43,44) for m in ("normal","fedalt")]
missing = [x for x in required if not list((root/x/"result_logs").rglob("*.json"))]
assert not missing, missing
print("[baseline][ok] P0/P1 required groups", len(required))
PY
```

## 4.2 建议执行顺序

```text
Stage 0: runner 256 门禁 -> YOCO smoke
Stage 1 (P0): FlowerTune YOCO ×3
Stage 2 (P0): existing D1/Flower baseline Worst In-Domain；YOCO 完成后追加其 matched-domain
Stage 3 (P0/P1): Flower 其余五类缺失 baseline ×3
Stage 4 (P0/P1): Normal/FedALT/HydraLoRA external export + smoke + formal
Stage 5 (P1): 从主方法 15 个 Flower held-out JSON 提取 nearest-client 配对结果
Stage 6 (P1): 70c Normal/FedALT
Stage 7 (P1): common-test alpha=0.5 Normal/FedSA/FedALT
Stage 8 (P2): r16 Normal/FedALT；D1 nearest-client 零 GPU 审计
```

## 4.3 剩余阻塞与诚实边界

1. FiQA 尚无本项目锁定的 lm-eval task/cache 映射；在 task 名与数据版本明确前继续标为未完成。
2. HydraLoRA 必须像 Normal/FedALT 一样完整解析三个正式 checkpoint；任一 seed 缺失时不得只用其余 seeds 填表。
3. FlowerTune 其余五个缺失 baseline 未完成三种子时，主表继续写 `--`；单 seed 不能填 mean±std。
4. matched-domain 的 source JSON、checkpoint 与 benchmark 必须位于同一节点可访问路径；不能只复制 result JSON 而不复制 checkpoint。
5. 所有 matched-domain 日志必须显式出现 `max_seq_length=256`，否则整批无效。

# FedPLoRA 20260725 投稿缺口实验命令（gb 服务器 · 合并版）

######### order_Main_20260725 + order_baseline_20260725 的 gb 单卡适配-20260725 #########

> 由 `order/order_Main_20260725.md` 与 `order/order_baseline_20260725.md` 合并适配 gb。章节与正式协议保持原文；仅换路径、conda、GPU，并写入 gb 已知坑。  
> **主方法训练/checkpoint** → `order_0725/main/`；**baseline 训练** → `order_0725/baseline/`；**matched-domain eval-only** → `order_0725/eval_only_*`。  
> 与 `order_gb_0723new.md` / `order_0723_sup` 重叠的 YOCO-Flower、7.2-A common-test、70c、r16 等：**若 sup 已产出 final JSON，本节标注「跳过」**，不要重复 `--force_retrain`。

## 先看：运行顺序总览（严格先后）

与 `order_Main_20260725.md` / `order_baseline_20260725.md` 一致：**Main 全部在前，Baseline 在后**；每份内部按 **主实验 → 正文 → 附录**。**附录整块必须最后跑**，不得提前启动附录 GPU 作业。

```text
Stage 0  第零部分：共同前置（0.1–0.5）+ Main smoke

Stage 1  第一部分 Main · 主实验
         1.1 Worst In-Domain ours（6 eval-only）
         1.2 external eval ours ×3 seeds

Stage 2  第一部分 Main · 正文实验
         2.1 FlowerTune 五折×三 seed routing/1-shot（15 GPU）
         2.2 70-client FedPLoRA-OS ×3（正文规模扩展；prep 在 2.2-prep）

Stage 3  第二部分 Baseline · 主实验
         B-smoke → 1.1 Flower 缺失 baseline → 1.2 Worst In-Domain baseline（54 eval-only）
         → 1.3 external baseline

Stage 4  第二部分 Baseline · 正文实验
         2.1 nearest-client 配对汇总（0-GPU，依赖 Main 2.1 完成）
         2.2 70-client Normal/FedALT ×6

Stage 5  ⛔ 第一部分 Main · 附录实验（最后一批 Main GPU）
         3.1 common-test α=0.5 v13a ×3
         3.2 LoRA r=16 v13a ×3
         3.3 D1 strict held-out 五折×三 seed ×15

Stage 6  ⛔ 第二部分 Baseline · 附录实验（全文件最后）
         3.1 common-test α=0.5 baseline ×9
         3.2 LoRA r=16 baseline ×6
         3.3 D1 nearest-client 审计（0-GPU，依赖 Main 3.3 完成）

Stage 7  第三部分：总体验收与 summarize 出表
```

**硬依赖：**

| 后续阶段 | 必须先完成 |
|---|---|
| Baseline 2.1 | Main 2.1 共 15 个 flower JSON |
| Baseline 3.3 | Main 3.3 共 15 个 d1 held-out JSON |
| Main 3.1 | `DIR05_COMMON_ROOT` repartition + cmp（或 order_0723 7.2-A 已做过） |
| Main 2.2 / Baseline 2.2 | `D1_70C_ROOT` repartition + cmp（Main 2.2-prep） |
| Main/Baseline external | HF 离线 cache verify_only 通过 |

## 【路径对照（minghao A100 → gb）】

| 项 | order_*_20260725.md | order_gb_0725.md |
|---|---|---|
| conda | `FedRepo2` | `fedplora` |
| 代码 | `/data2/minghao/code/FedPLoRA-main` | `/data/yaominghao/gb/FedPLoRA` |
| 模型 | `/data2/minghao/model/` | `/data/yaominghao/gb/models/` |
| 主方法结果 | `.../order_main_20260725/` | `.../order_0725/main/` |
| baseline 结果 | `.../order_baseline_20260725/` | `.../order_0725/baseline/` |
| D1 数据 | `A100_domain_benchmark_35c_dir05` | `domain_benchmark_35c_dir05` |
| 70c 数据 | `A100_domain_benchmark_70c_dir05_frozen_test` | `domain_benchmark_70c_dir05_frozen_test` |
| GPU | 多卡 0–7 | **单卡串行**；`export GPU_ID=0` 或 `1` |

## 【gb 防坑（必须遵守）】

```text
1. 每条 GPU 命令前：cd /data/yaominghao/gb/FedPLoRA && export PATH=.../fedplora/bin
2. nohup 子 shell 勿 set -u + source conda.sh；用 PATH 指向 fedplora/bin/python
3. gb 无 A100_* 前缀；D1 一律 domain_benchmark_35c_dir05
4. 单卡串行：同 GPU 一次只开一条 nohup；上一条结束再开下一条
5. personalized_eval 不加 --force_retrain；SFT 正式训练必须 --force_retrain
6. matched-domain runner：scripts/RunScripts/run_eval_only_matched_domain.sh（非 FedPLoRA-main/...）
7. summarize_fedplora_results.py 只接受一个 root 位置参数；多目录用 --output 写到同一 summary，或分别汇总
8. HF 不可达：rsync 本地 cache 到 data/external_lm_eval_hf_cache/，并 export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
9. 读历史 checkpoint 的 result JSON 在 FED_RESULT_ROOT/order_0709|0711|0712|0715|0723*；本批次新训练写在 ORDER_ROOT 下
```

| 部分 | 作业 | 数量 | gb 章节 |
|---|---|---:|---|
| Main 主实验 | Worst In-Domain eval-only | 6 | Main-1.1 |
| Main 主实验 | external eval MMLU/PubMedQA/MBPP | 3 launcher | Main-1.2 |
| Main 正文 | FlowerTune 1-example + routing | 15 | Main-2.1 |
| Main 正文 | 70-client FedPLoRA-OS | 3 | Main-2.2 |
| Main **附录** | common-test α=0.5 v13a | 3 | Main-3.1 ⛔ |
| Main **附录** | LoRA r=16 v13a | 3 | Main-3.2 ⛔ |
| Main **附录** | D1 strict held-out | 15 | Main-3.3 ⛔ |
| Baseline 主实验 | Flower 缺失 baseline | 18 | Baseline-1.1 |
| Baseline 主实验 | Worst In-Domain baseline | 54 eval-only | Baseline-1.2 |
| Baseline 主实验 | external baseline | 9 launcher | Baseline-1.3 |
| Baseline 正文 | nearest-client 汇总 | 0 GPU | Baseline-2.1 |
| Baseline 正文 | 70c Normal/FedALT | 6 | Baseline-2.2 |
| Baseline **附录** | common-test α=0.5 baseline | 9 | Baseline-3.1 ⛔ |
| Baseline **附录** | r=16 baseline | 6 | Baseline-3.2 ⛔ |
| Baseline **附录** | D1 nearest-client 审计 | 0 GPU | Baseline-3.3 ⛔ |

## 【与 order_0723_sup 去重】

| 0725 缺口 | gb 上可能已在 sup 完成 | 本文件处理 |
|---|---|---|
| Flower YOCO ×3 | `order_0723_sup/yoco_flower_seed*` | 已完成则跳过 B-YOCO；matched-domain 仍要补 |
| common-test α=0.5 ×6 | `order_0723` 第七部分 7.2-A | v13a/normal/fedsa/fedalt 已完成则跳过 M5/B-common |
| 70c ×9 | `order_0723_sup/70c_*` | 已完成则跳过 M4/B-70c |
| r16 ×9 | `order_0723_sup/r16_*` | 已完成则跳过 M6/B-r16 |
| Flower 1-shot offset1–4 | `order_0723` G1–G12 | 0725 的 15 fold 含 offset0；offset0 若 0723 已有可只补缺项 |

---

# 第零部分：共同前置

## 0.1 代码同步

```bash
cd /data/yaominghao/gb/FedPLoRA && git pull
```

需含：`run_eval_only_matched_domain.sh`、`fed_train_sft.py`、`eval_personalized.py`、`checkpoint_manifest.py`、`run_external_lm_eval.py`、`prepare_external_lm_eval_hf_cache.py`、`repartition_with_frozen_test.py`、`summarize_matched_domain_eval.py`、`summarize_fedplora_results.py`。

## 0.2 环境变量（每个新 shell 粘贴一次）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && exec bash

export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export FED_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0725
export MAIN_RESULT_ROOT="$ORDER_ROOT/main"
export BASELINE_RESULT_ROOT="$ORDER_ROOT/baseline"
export MAIN_MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW/order_0725/main
export BASELINE_MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW/order_0725/baseline
export MODEL_135M=/data/yaominghao/gb/models/SmolLM2-135M
export D1_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir05"
export FLOWER_ROOT="$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05"
export D1_70C_ROOT="$CODE_DIR/data/domain_benchmark_70c_dir05_frozen_test"
export IID_ROOT="$CODE_DIR/data/domain_benchmark_35c_iid"
export DIR01_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir01"
export DIR05_COMMON_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir05_common_test_v2"
export HF_CACHE_ROOT="$CODE_DIR/data/external_lm_eval_hf_cache"
export GPU_ID=${GPU_ID:-0}

mkdir -p "$ORDER_ROOT/launcher_logs" "$ORDER_ROOT/pids" "$ORDER_ROOT/analysis" \
  "$MAIN_RESULT_ROOT/launcher_logs" "$MAIN_RESULT_ROOT/pids" \
  "$BASELINE_RESULT_ROOT/launcher_logs" "$BASELINE_RESULT_ROOT/pids" \
  "$MAIN_MODEL_ROOT" "$BASELINE_MODEL_ROOT"
cd "$CODE_DIR"
```

## 0.3 代码与 256 门禁

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/Analysis/eval_personalized.py \
  scripts/Analysis/checkpoint_manifest.py \
  scripts/Analysis/run_external_lm_eval.py \
  scripts/Analysis/prepare_external_lm_eval_hf_cache.py \
  scripts/Analysis/summarize_matched_domain_eval.py \
  scripts/DataProcessScripts/repartition_with_frozen_test.py
bash -n scripts/RunScripts/run_20260713_one_experiment.sh
bash -n scripts/RunScripts/run_eval_only_matched_domain.sh
grep -q 'EVAL_MAX_SEQ_LENGTH="${EVAL_MAX_SEQ_LENGTH:-256}"' scripts/RunScripts/run_eval_only_matched_domain.sh
grep -q -- '--max_seq_length "${EVAL_MAX_SEQ_LENGTH}"' scripts/RunScripts/run_eval_only_matched_domain.sh
```

任一 `grep` 失败必须停止。

## 0.4 公共 SFT 参数

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && COMMON_SFT_ARGS=(
  --rounds 1 --local_epochs 1 --lr 0.0002 --lora_dropout 0.05
  --batch_size 2 --max_seq_length 256 --torch_dtype bfloat16
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --save_client_state_to_disk --gradient_checkpointing
  --eval_personalization_metrics --eval_final_only --skip_post_agg_snapshots
)
LORA_R8_ARGS=(--lora_r 8 --lora_alpha 16)
LORA_R16_ARGS=(--lora_r 16 --lora_alpha 32)
YOCO_ARGS=(--yoco_sparse_lambda 0.0001 --yoco_pcwa_components 3 --yoco_aggregate_mode conflict --yoco_conflict_method avgm --yoco_sign_lambda 0.01)
FEDDAT_EXTRA=(--feddat_teacher_lambda 0.01)
HILORA_EXTRA=(--hilora_leaf_blend 0.25)
```

## 0.5 数据检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import collections, json, pathlib
checks = [
    (pathlib.Path("/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05"), 35, 7, 5),
    (pathlib.Path("/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05"), 20, 4, 5),
]
for root, n_clients, n_domains, per_domain in checks:
    for seed in (42, 43, 44):
        split = root / f"seed_{seed}"
        rows = json.loads((split / "clients.json").read_text(encoding="utf-8"))
        counts = collections.Counter(str(x["domain"]) for x in rows)
        assert len(rows) == n_clients and len(counts) == n_domains and set(counts.values()) == {per_domain}
        for name in ("train.jsonl", "val.jsonl", "test_local.jsonl", "test_domain.jsonl"):
            assert (split / name).is_file(), split / name
        print("[data][ok]", split, dict(sorted(counts.items())))
PY
```

---

# 第一部分：FedPLoRA-OS 主方法（order_Main_20260725）

> 对应原文：**主实验 → 正文 → 附录**。附录（第三部分）标注 ⛔，**必须 Stage 5 再跑**。

## Main-0. smoke：三条门禁（串行）

### M-S1 主方法训练 smoke

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_42" --num_clients 35 --agg_type fedplora_v13a_os \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" \
  --client_state_dir "$MAIN_RESULT_ROOT/smoke_v13a_seed42/result_files/client_states/N7_smoke_v13a_seed42" \
  --metrics_output_dir "$MAIN_RESULT_ROOT/smoke_v13a_seed42/result_logs/N7_smoke_v13a_seed42" \
  --save_run_checkpoint_dir "$MAIN_MODEL_ROOT/smoke_v13a_seed42/N7_smoke_v13a_seed42" \
  --trained_models_root "$MAIN_MODEL_ROOT/smoke_v13a_seed42" \
  --eval_max_batches 1 --seed 42 --force_retrain \
  --train_max_steps_per_client 1 --max_train_samples_per_client 10 \
  > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_main_smoke_v13a_seed42.log" 2>&1 &
echo $! > "$MAIN_RESULT_ROOT/pids/smoke_v13a_seed42.pid"
```

### M-S2 personalized routing smoke

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" \
BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 \
MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=1 \
nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval --method X2_smoke_flower_route_probe1_seed42 \
  --seed 42 --split-seed 42 --run-id-prefix main_20260725_smoke_route --gpu "${GPU_ID:-0}" -- \
  --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 \
  --few_shot_caps 1 --held_out_route_probe_samples 1 \
  --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle \
  --onboarding_accounting --schemes base,global,coldstart,coldstart_geom \
  --select_candidates global,coldstart,coldstart_geom \
  --max_steps 1 --max_train_samples_per_client 10 \
  > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_main_smoke_route.launch.log" 2>&1 &
```

### M-S3 matched-domain smoke（256 协议）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && \
SMOKE_JSON=$(find "$FED_RESULT_ROOT/order_0712" -path '*NX0_v13a_os_split42_train42/*.json' 2>/dev/null | head -n 1)
test -n "$SMOKE_JSON" || SMOKE_JSON=$(find "$FED_RESULT_ROOT" -path '*NX0_v13a_os_split42_train42/*.json' 2>/dev/null | head -n 1)
test -n "$SMOKE_JSON"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" EVAL_MAX_BATCHES=1 EVAL_MAX_SEQ_LENGTH=256 \
MATCHED_DOMAIN_OUTPUT_ROOT="$ORDER_ROOT/eval_only_main_20260725/smoke" \
bash scripts/RunScripts/run_eval_only_matched_domain.sh "$SMOKE_JSON"
grep -R 'max_seq_length=256' "$ORDER_ROOT/eval_only_main_20260725/smoke" || grep -R 'max_seq_length=256' "$ORDER_ROOT/eval_only_main_20260725/smoke/../logs" 2>/dev/null || true
```

---

# Main-第一部分：主实验——协议一致的核心有效性

## Main-1.1 【原 1、10】FedPLoRA-OS Worst In-Domain ×3 seeds（D1 + FlowerTune）

在原训练节点 gb 执行；读历史 v13a JSON，写 `order_0725/eval_only_main_20260725/`。

```bash
set -euo pipefail
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}"
export MD_ROOT="$ORDER_ROOT/eval_only_main_20260725"
export MD_RUNNER=scripts/RunScripts/run_eval_only_matched_domain.sh
export MD_SUMMARIZER=scripts/Analysis/summarize_matched_domain_eval.py
mkdir -p "$MD_ROOT/d1" "$MD_ROOT/flowertune" "$MD_ROOT/logs" "$MD_ROOT/pids"

find_one_json () {
  local dir="$1"
  mapfile -t hits < <(find "$dir" -maxdepth 1 -type f -name '*.json' | sort)
  [[ "${#hits[@]}" -eq 1 ]] || { echo "[source][error] expected one JSON: $dir, got ${#hits[@]}" >&2; return 2; }
  printf '%s\n' "${hits[0]}"
}

D1_OURS_42=$(find_one_json "$FED_RESULT_ROOT/order_0712/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42" 2>/dev/null || find_one_json "$FED_RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42")
D1_OURS_43=$(find_one_json "$FED_RESULT_ROOT/order_0711/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43" 2>/dev/null || find_one_json "$FED_RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43")
D1_OURS_44=$(find_one_json "$FED_RESULT_ROOT/order_0711/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44" 2>/dev/null || find_one_json "$FED_RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44")
FLOWER_OURS_42=$(find_one_json "$FED_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed42/result_logs/N7_ours_flower_v13a")
FLOWER_OURS_43=$(find_one_json "$FED_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed43/result_logs/N7_ours_flower_v13a")
FLOWER_OURS_44=$(find_one_json "$FED_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed44/result_logs/N7_ours_flower_v13a")

launch_md () {
  local tag="$1" source_json="$2" output="$3"
  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
    EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$output" \
    bash "$MD_RUNNER" "$source_json" > "$MD_ROOT/logs/${tag}.log" 2>&1 &
  echo $! > "$MD_ROOT/pids/${tag}.pid"
  wait $(cat "$MD_ROOT/pids/${tag}.pid") 2>/dev/null || true
}

launch_md d1_ours_seed42 "$D1_OURS_42" "$MD_ROOT/d1"
launch_md d1_ours_seed43 "$D1_OURS_43" "$MD_ROOT/d1"
launch_md d1_ours_seed44 "$D1_OURS_44" "$MD_ROOT/d1"
launch_md flower_ours_seed42 "$FLOWER_OURS_42" "$MD_ROOT/flowertune"
launch_md flower_ours_seed43 "$FLOWER_OURS_43" "$MD_ROOT/flowertune"
launch_md flower_ours_seed44 "$FLOWER_OURS_44" "$MD_ROOT/flowertune"

[[ "$(find "$MD_ROOT/d1" -name '*_matched_domain.json' | wc -l)" -eq 3 ]]
[[ "$(find "$MD_ROOT/flowertune" -name '*_matched_domain.json' | wc -l)" -eq 3 ]]
python "$MD_SUMMARIZER" "$MD_ROOT/d1" | tee "$MD_ROOT/d1_ours_summary.tsv"
python "$MD_SUMMARIZER" "$MD_ROOT/flowertune" | tee "$MD_ROOT/flower_ours_summary.tsv"
grep -R 'max_seq_length=256' "$MD_ROOT/logs"
```

---

## Main-1.2 【原 9】external eval：FedPLoRA-OS ×3 seeds（gb 离线 cache）

### Main-1.2-E0 任务与 cache 门禁

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && \
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets" \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/Analysis/prepare_external_lm_eval_hf_cache.py --cache_root "$HF_CACHE_ROOT" --tasks mmlu,pubmedqa,mbpp --verify_only
python -m lm_eval ls tasks > "$MAIN_RESULT_ROOT/analysis/lm_eval_tasks.txt"
for TASK in mmlu pubmedqa mbpp; do
  grep -Eq "(^|[[:space:]])${TASK}([[:space:]]|$)" "$MAIN_RESULT_ROOT/analysis/lm_eval_tasks.txt" \
    || { echo "[external][error] task not registered: $TASK" >&2; exit 1; }
done
```

### Main-1.2-E1 adapter export（读 gb 上 v13a checkpoint）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && \
export CKPT_SEARCH_ROOTS="$MAIN_MODEL_ROOT $FED_RESULT_ROOT $MAIN_RESULT_ROOT"
export_ours_adapter () {
  local seed="$1"
  local ckpt
  ckpt=$(python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS --resolve \
    --agg_type fedplora_v13a_os --seed "$seed" --model_contains SmolLM2-135M \
    --benchmark_contains "domain_benchmark_35c_dir05/seed_${seed}")
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python -u tasks/fed_train_sft.py \
    --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_${seed}" --agg_type fedplora_v13a_os --seed "$seed" \
    --eval_only_from_checkpoint "$ckpt" \
    --metrics_output_dir "$MAIN_RESULT_ROOT/external_export/ours_seed${seed}/metrics" \
    --client_state_dir "$MAIN_RESULT_ROOT/external_export/ours_seed${seed}/scratch" \
    --export_eval_adapter_dir "$MAIN_RESULT_ROOT/external_adapters/ours_seed${seed}" \
    --export_eval_adapter_only --eval_max_batches 0 --batch_size 2 --max_seq_length 256 \
    --torch_dtype bfloat16 --eval_personalization_metrics
}
export_ours_adapter 42; export_ours_adapter 43; export_ours_adapter 44
```

### Main-1.2-E2 正式 lm-eval（串行三 seed）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && \
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py \
    --adapter_manifest "$MAIN_RESULT_ROOT/external_adapters/ours_seed${SEED}/adapter_export_manifest.json" \
    --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto \
    --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code \
    --output_dir "$MAIN_RESULT_ROOT/external_eval/ours_seed${SEED}" \
    > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_external_ours_seed${SEED}.log" 2>&1 &
  wait $!
done
```

---

# Main-第二部分：正文实验——路由解释与参与者扩展

## Main-2.1 【原 3、4】FlowerTune 五折×三 seed 1-example + 六类路由（15 条 GPU，串行）

每条命令等前一条结束再跑。`offset0` 若 `order_0723` 已有完整 JSON 可跳过对应三行。

```bash
# offset0 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset0_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset0_seed42.launch.log" 2>&1 &
```

```bash
# offset0 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset0_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset0_seed43.launch.log" 2>&1 &
```

```bash
# offset0 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset0_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset0_seed44.launch.log" 2>&1 &
```

```bash
# offset1 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset1_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset1_seed42.launch.log" 2>&1 &
```

```bash
# offset1 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset1_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset1_seed43.launch.log" 2>&1 &
```

```bash
# offset1 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset1_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset1_seed44.launch.log" 2>&1 &
```

```bash
# offset2 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset2_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset2_seed42.launch.log" 2>&1 &
```

```bash
# offset2 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset2_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset2_seed43.launch.log" 2>&1 &
```

```bash
# offset2 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset2_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset2_seed44.launch.log" 2>&1 &
```

```bash
# offset3 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset3_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset3_seed42.launch.log" 2>&1 &
```

```bash
# offset3 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset3_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset3_seed43.launch.log" 2>&1 &
```

```bash
# offset3 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset3_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset3_seed44.launch.log" 2>&1 &
```

```bash
# offset4 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset4_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset4_seed42.launch.log" 2>&1 &
```

```bash
# offset4 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset4_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset4_seed43.launch.log" 2>&1 &
```

```bash
# offset4 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_offset4_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_flower_probe1 --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_flower_probe1_offset4_seed44.launch.log" 2>&1 &
```

### Main-2.1-check

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import json, pathlib
root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0725/main")
paths = sorted(root.glob("main_20260725_flower_probe1_seed*/result_logs/X2_flower_probe1_offset*_seed*.json"))
assert len(paths) == 15, len(paths)
required = {"flat_b_cosine", "subspace", "relative_l2", "delta_w_cosine", "nearest_client_subspace", "random", "oracle"}
for path in paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    audits = (row.get("strict_held_out") or {}).get("route_audits") or {}
    assert required <= set(audits), (path, sorted(audits))
print("[flower-route][ok]", len(paths))
PY
```

---

## Main-2.2-prep. 70-client frozen split（0-GPU，正文 2.2 前置，只执行一次）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && for SEED in 42 43 44; do
  python scripts/DataProcessScripts/repartition_with_frozen_test.py \
    --reference_split "$D1_ROOT/seed_${SEED}" --output_dir "$D1_70C_ROOT" \
    --num_clients_per_domain 10 --min_samples_per_client 25 --seed "$SEED" \
    --partition dirichlet --dirichlet_alpha 0.5 --subtopic kmeans --n_subtopics 10
  cmp -s "$D1_70C_ROOT/seed_${SEED}/test_domain.jsonl" "$D1_ROOT/seed_${SEED}/test_domain.jsonl"
done
```

## Main-2.2 【原 6】70-client FedPLoRA-OS ×3 seeds

若 `$FED_RESULT_ROOT/order_0723_sup/70c_v13a_seed42/result_logs` 已有 final JSON，**跳过**。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_42" --num_clients 70 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$MAIN_RESULT_ROOT/70c_v13a_seed42/result_files/client_states/N7_70c_v13a_seed42" --metrics_output_dir "$MAIN_RESULT_ROOT/70c_v13a_seed42/result_logs/N7_70c_v13a_seed42" --save_run_checkpoint_dir "$MAIN_MODEL_ROOT/70c_v13a_seed42/N7_70c_v13a_seed42" --trained_models_root "$MAIN_MODEL_ROOT/70c_v13a_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_main_70c_v13a_seed42.log" 2>&1 &
# seed43/44 同上，改 seed 与路径
```

---

# Main-第三部分：附录实验 ⛔ Stage 5 —— 稳健性与跨数据集复现（最后跑）

> **不要提前启动本节 GPU 作业。** 须 Main 主实验+正文、Baseline 主实验+正文 均完成后，再进入 Stage 5。

## Main-3.1-prep. common-test α=0.5 split（0-GPU）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && for SEED in 42 43 44; do
  python scripts/DataProcessScripts/repartition_with_frozen_test.py \
    --reference_split "$IID_ROOT/seed_${SEED}" --output_dir "$DIR05_COMMON_ROOT" \
    --num_clients_per_domain 5 --seed "$SEED" --partition dirichlet \
    --dirichlet_alpha 0.5 --subtopic kmeans --n_subtopics 10
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$IID_ROOT/seed_${SEED}/test_domain.jsonl"
done
```

## Main-3.1 【原 5】common-test α=0.5 FedPLoRA-OS ×3

若 `order_0723` 7.2-A 已有 v13a final JSON，**跳过**。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$DIR05_COMMON_ROOT/seed_42" --num_clients 35 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$MAIN_RESULT_ROOT/common_a05_v13a_seed42/result_files/client_states/N7_common_a05_v13a_seed42" --metrics_output_dir "$MAIN_RESULT_ROOT/common_a05_v13a_seed42/result_logs/N7_common_a05_v13a_seed42" --save_run_checkpoint_dir "$MAIN_MODEL_ROOT/common_a05_v13a_seed42/N7_common_a05_v13a_seed42" --trained_models_root "$MAIN_MODEL_ROOT/common_a05_v13a_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_main_common_a05_v13a_seed42.log" 2>&1 &
# seed43/44 同上
```

## Main-3.2 【原 7】LoRA r=16 FedPLoRA-OS ×3

若 `order_0723_sup/r16_v13a_seed*` 已有 final JSON，**跳过**。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_42" --num_clients 35 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$MAIN_RESULT_ROOT/r16_v13a_seed42/result_files/client_states/N7_r16_v13a_seed42" --metrics_output_dir "$MAIN_RESULT_ROOT/r16_v13a_seed42/result_logs/N7_r16_v13a_seed42" --save_run_checkpoint_dir "$MAIN_MODEL_ROOT/r16_v13a_seed42/N7_r16_v13a_seed42" --trained_models_root "$MAIN_MODEL_ROOT/r16_v13a_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_main_r16_v13a_seed42.log" 2>&1 &
# seed43/44 同上
```

## Main-3.3 【原 8】D1 strict held-out 五折×三 seed（15 条 GPU，串行）

probe headline = 10 examples；`few_shot_caps 1,5,10`。每条等前一条结束再跑。

```bash
# offset0 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset0_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset0_seed42.launch.log" 2>&1 &
```

```bash
# offset0 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset0_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset0_seed43.launch.log" 2>&1 &
```

```bash
# offset0 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset0_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset0_seed44.launch.log" 2>&1 &
```

```bash
# offset1 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset1_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset1_seed42.launch.log" 2>&1 &
```

```bash
# offset1 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset1_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset1_seed43.launch.log" 2>&1 &
```

```bash
# offset1 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset1_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset1_seed44.launch.log" 2>&1 &
```

```bash
# offset2 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset2_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset2_seed42.launch.log" 2>&1 &
```

```bash
# offset2 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset2_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset2_seed43.launch.log" 2>&1 &
```

```bash
# offset2 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset2_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset2_seed44.launch.log" 2>&1 &
```

```bash
# offset3 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset3_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset3_seed42.launch.log" 2>&1 &
```

```bash
# offset3 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset3_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset3_seed43.launch.log" 2>&1 &
```

```bash
# offset3 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset3_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset3_seed44.launch.log" 2>&1 &
```

```bash
# offset4 seed42
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset4_seed42 --seed 42 --split-seed 42 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset4_seed42.launch.log" 2>&1 &
```

```bash
# offset4 seed43
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset4_seed43 --seed 43 --split-seed 43 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset4_seed43.launch.log" 2>&1 &
```

```bash
# offset4 seed44
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && RESULT_ROOT="$MAIN_RESULT_ROOT" MODEL_ROOT="$MAIN_MODEL_ROOT" MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_d1_heldout_offset4_seed44 --seed 44 --split-seed 44 --run-id-prefix main_20260725_d1_heldout --gpu "${GPU_ID:-0}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$MAIN_RESULT_ROOT/launcher_logs/test20260725_d1_heldout_offset4_seed44.launch.log" 2>&1 &
```

### Main-3.3-check

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import json, pathlib
root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0725/main")
paths = sorted(root.glob("main_20260725_d1_heldout_seed*/result_logs/X2_d1_heldout_offset*_seed*.json"))
assert len(paths) == 15, len(paths)
total = 0
for path in paths:
    audits = (json.loads(path.read_text()) .get("strict_held_out") or {}).get("route_audits") or {}
    assert "subspace" in audits and "nearest_client_subspace" in audits
    n = audits["subspace"]["summary"]["num_routed"]
    assert n == 7, (path, n)
    total += n
assert total == 105
print("[d1-heldout][ok]", len(paths), "routes=", total)
PY
```

---

# 第二部分：baseline（order_baseline_20260725）

> 对应原文：**主实验 → 正文 → 附录**。须在 **Main 主实验+正文（Stage 1–2）** 之后跑 Baseline 主实验+正文（Stage 3–4）；**Baseline 附录（Stage 6）** 为全文件最后一批 GPU。

## Baseline-0. smoke：YOCO FlowerTune

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$FLOWER_ROOT/seed_42" --num_clients 20 --agg_type yoco "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" --client_state_dir "$BASELINE_RESULT_ROOT/smoke_flower_yoco_seed42/result_files/client_states/N9_smoke_flower_yoco_seed42" --metrics_output_dir "$BASELINE_RESULT_ROOT/smoke_flower_yoco_seed42/result_logs/N9_smoke_flower_yoco_seed42" --save_run_checkpoint_dir "$BASELINE_MODEL_ROOT/smoke_flower_yoco_seed42/N9_smoke_flower_yoco_seed42" --trained_models_root "$BASELINE_MODEL_ROOT/smoke_flower_yoco_seed42" --eval_max_batches 1 --seed 42 --force_retrain --train_max_steps_per_client 1 --max_train_samples_per_client 10 > "$BASELINE_RESULT_ROOT/launcher_logs/test20260725_baseline_smoke_yoco.log" 2>&1 &
```

---

# Baseline-第一部分：主实验——主表可比性与任务级外部有效性

## Baseline-1.1 【原 2】FlowerTune 缺失 baseline

### B-YOCO ×3

若 `$FED_RESULT_ROOT/order_0723_sup/yoco_flower_seed42/result_logs` 已有 final JSON，**跳过训练**，仅做 B1-md matched-domain。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$FLOWER_ROOT/seed_42" --num_clients 20 --agg_type yoco "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" --client_state_dir "$BASELINE_RESULT_ROOT/flower_yoco_seed42/result_files/client_states/N9_flower_yoco_seed42" --metrics_output_dir "$BASELINE_RESULT_ROOT/flower_yoco_seed42/result_logs/N9_flower_yoco_seed42" --save_run_checkpoint_dir "$BASELINE_MODEL_ROOT/flower_yoco_seed42/N9_flower_yoco_seed42" --trained_models_root "$BASELINE_MODEL_ROOT/flower_yoco_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$BASELINE_RESULT_ROOT/launcher_logs/test20260725_baseline_flower_yoco_seed42.log" 2>&1 &
# seed43/44 同上
```

### B-OTHER ×15（ffa/flora/flexlora/feddat/hilora ×3 seeds，串行）

以 ffa seed42 为例；feddat/hilora 追加对应 EXTRA 数组：

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$FLOWER_ROOT/seed_42" --num_clients 20 --agg_type ffa "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$BASELINE_RESULT_ROOT/flower_ffa_seed42/result_files/client_states/N9_flower_ffa_seed42" --metrics_output_dir "$BASELINE_RESULT_ROOT/flower_ffa_seed42/result_logs/N9_flower_ffa_seed42" --save_run_checkpoint_dir "$BASELINE_MODEL_ROOT/flower_ffa_seed42/N9_flower_ffa_seed42" --trained_models_root "$BASELINE_MODEL_ROOT/flower_ffa_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$BASELINE_RESULT_ROOT/launcher_logs/test20260725_baseline_flower_ffa_seed42.log" 2>&1 &
```

---

## Baseline-1.2 【原 1、10】baseline Worst In-Domain（gb 顺序 runner）

```bash
set -euo pipefail
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}"
export MD_ROOT="$ORDER_ROOT/eval_only_baseline_20260725"
export MD_RUNNER=scripts/RunScripts/run_eval_only_matched_domain.sh
mkdir -p "$MD_ROOT/d1" "$MD_ROOT/flowertune_existing" "$MD_ROOT/logs" "$MD_ROOT/pids"

D1_METHOD_DIRS=(OS1_normal OS1_ffa OS1_flora OS1_flexlora OS1_ecolora OS1_fedsa_lora OS1_feddat OS1_yoco OS1_fedalt OS1_hydralora OS1_hilora OS1_fedlease)
FLOWER_METHOD_DIRS=(N9_flower_normal N9_flower_ecolora N9_flower_fedsa_lora N9_flower_fedalt N9_flower_hydralora N9_flower_fedlease)

D1_RESULTS=()
for SEED in 42 43 44; do
  BASE="$FED_RESULT_ROOT/order_0709/os_20260709_baseline_35c_dir05_r1_finaleval_seed${SEED}/result_logs"
  for METHOD in "${D1_METHOD_DIRS[@]}"; do
    mapfile -t HITS < <(find "$BASE/$METHOD" -maxdepth 1 -name '*.json' | sort)
    [[ "${#HITS[@]}" -eq 1 ]]
    D1_RESULTS+=("${HITS[0]}")
  done
done

FLOWER_RESULTS=()
for SEED in 42 43 44; do
  BASE="$FED_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed${SEED}/result_logs"
  for METHOD in "${FLOWER_METHOD_DIRS[@]}"; do
    mapfile -t HITS < <(find "$BASE/$METHOD" -maxdepth 1 -name '*.json' | sort)
    [[ "${#HITS[@]}" -eq 1 ]]
    FLOWER_RESULTS+=("${HITS[0]}")
  done
done

[[ "${#D1_RESULTS[@]}" -eq 36 ]]
[[ "${#FLOWER_RESULTS[@]}" -eq 18 ]]

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
  EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$MD_ROOT/d1" \
  bash "$MD_RUNNER" "${D1_RESULTS[@]}" > "$MD_ROOT/logs/d1_baselines.log" 2>&1 &
echo $! > "$MD_ROOT/pids/d1_baselines.pid"
wait $(cat "$MD_ROOT/pids/d1_baselines.pid")

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
  EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$MD_ROOT/flowertune_existing" \
  bash "$MD_RUNNER" "${FLOWER_RESULTS[@]}" > "$MD_ROOT/logs/flower_existing_baselines.log" 2>&1 &
echo $! > "$MD_ROOT/pids/flower_existing_baselines.pid"
wait $(cat "$MD_ROOT/pids/flower_existing_baselines.pid")
```

新补 Flower baseline（YOCO 等）训练完成后，对每个 JSON 单独跑 matched-domain（串行）：

```bash
launch_new_flower_md () {
  local tag="$1" agg="$2"
  mapfile -t hits < <(find "$BASELINE_RESULT_ROOT/$tag/result_logs/N9_${tag}_${agg}" -maxdepth 1 -name '*.json' | sort)
  [[ "${#hits[@]}" -eq 1 ]]
  env CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" EVAL_MAX_BATCHES=0 EVAL_MAX_SEQ_LENGTH=256 \
    EVAL_BATCH_SIZE=2 EVAL_TORCH_DTYPE=bfloat16 MATCHED_DOMAIN_OUTPUT_ROOT="$BASELINE_RESULT_ROOT/matched_domain_new_flower" \
    bash scripts/RunScripts/run_eval_only_matched_domain.sh "${hits[0]}"
}
# launch_new_flower_md flower_yoco_seed42 yoco  # 逐 tag 调用
```

---

## Baseline-1.3 【原 9】external eval：Normal / FedALT / HydraLoRA ×3 seeds

结构与 Main-1.2 对称，结果根换 `$BASELINE_RESULT_ROOT` / `$BASELINE_MODEL_ROOT`；`export_baseline_adapter` 的 `--agg_type` 分别为 `normal`/`fedalt`/`hydralora`。lm-eval 时 Normal 用 `--mode global`，FedALT/HydraLoRA 用 `--mode both`。须 HF 离线 cache（同 Main-1.2-E0）。

---

# Baseline-第二部分：正文实验——最近客户端替代解释与 70-client 对照

> **前置：** Main-2.1 共 15 个 flower JSON 必须齐全，再跑 Baseline-2.1。

## Baseline-2.1 【原 4】nearest-training-client retrieval（0-GPU，读 Main-2.1）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0725/main")
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

## Baseline-2.2 【原 6】70-client Normal/FedALT ×6

依赖 Main-2.2-prep 已构建的 `D1_70C_ROOT`。若 `order_0723_sup/70c_*` 已有 JSON，**跳过**。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && for SEED in 42 43 44; do
  cmp -s "$D1_70C_ROOT/seed_${SEED}/test_domain.jsonl" "$D1_ROOT/seed_${SEED}/test_domain.jsonl"
done
# 逐 seed 串行 launch：70c_normal_seed42/43/44、70c_fedalt_seed42/43/44（agg normal/fedalt，格式同 Main-2.2，结果根换 BASELINE_*）
```

---

# Baseline-第三部分：附录实验 ⛔ Stage 6 —— 异构与容量对照（全文件最后跑）

> **前置：** Main-3.3 共 15 个 d1 held-out JSON 齐全后，再跑 Baseline-3.3。

## Baseline-3.1 【原 5】common-test α=0.5：Normal / FedSA-LoRA / FedALT ×9

依赖 Main-3.1-prep 的 `DIR05_COMMON_ROOT`。若 `order_0723` 7.2-A 已有对应 JSON，**跳过**。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && for SEED in 42 43 44; do
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$IID_ROOT/seed_${SEED}/test_domain.jsonl"
done
# common_a05_{normal,fedsa,fedalt}_seed{42,43,44} ×9，结果根 BASELINE_RESULT_ROOT
```

## Baseline-3.2 【原 7】LoRA r=16：Normal / FedALT ×6

若 `order_0723_sup/r16_*` 已有 JSON，**跳过**。

```bash
# r16_{normal,fedalt}_seed{42,43,44} ×6，LORA_R16_ARGS，结果根 BASELINE_RESULT_ROOT
```

## Baseline-3.3 【原 8】D1 strict held-out 最近客户端审计（0-GPU，读 Main-3.3）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import json, pathlib
root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0725/main")
paths = sorted(root.glob("main_20260725_d1_heldout_seed*/result_logs/X2_d1_heldout_offset*_seed*.json"))
assert len(paths) == 15, len(paths)
total_routes = 0
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    audits = (data.get("strict_held_out") or {}).get("route_audits") or {}
    assert "subspace" in audits and "nearest_client_subspace" in audits, path
    n = audits["subspace"]["summary"]["num_routed"]
    assert n == 7, (path, n)
    total_routes += n
assert total_routes == 105, total_routes
print("[d1-retrieval][ok] fold-seed=15 routes=105")
PY
```

---

# 第三部分：总体验收

## Z1. 主方法完整性

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import pathlib
root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0725/main")
for pat, n in (
    ("main_20260725_flower_probe1_seed*/result_logs/X2_*.json", 15),
    ("main_20260725_d1_heldout_seed*/result_logs/X2_*.json", 15),
):
    got = len(list(root.glob(pat)))
    assert got == n, (pat, got)
print("[main][ok]")
PY
```

## Z2. summarize 出表（正确 CLI）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && \
python scripts/Analysis/summarize_fedplora_results.py "$MAIN_RESULT_ROOT" \
  --output "$ORDER_ROOT/summary_main_20260725.md" --exclude_smoke

python scripts/Analysis/summarize_fedplora_results.py "$BASELINE_RESULT_ROOT" \
  --output "$ORDER_ROOT/summary_baseline_20260725.md" --exclude_smoke
```

若只汇总 YOCO Flower（例如结果在 `order_0723_sup`）：

```bash
python scripts/Analysis/summarize_fedplora_results.py "$FED_RESULT_ROOT/order_0723_sup" \
  --output "$ORDER_ROOT/yoco_flower_summary_20260725.md" --kind sft --exclude_smoke
```

## Z3. 停止条件

1. matched-domain 日志未出现 `max_seq_length=256` → 整批无效。  
2. common-test / 70c 的 `cmp` 失败 → 停止 Non-IID / scale 实验。  
3. Flower 15 fold-seed 不全或 route audit 缺字段 → 不得汇总 1-example 表。  
4. external：cache 未 verify_only 通过 → 不得开 E2；FiQA 无稳定 task 前标未完成。  
5. MBPP 非隔离环境 → 删 `mbpp:code` 与 `--confirm_run_unsafe_code`，并在 summary 注明。

## Z4. 执行顺序（与 Stage 0–7 一致，禁止跳阶段）

```text
Stage 0:  第零部分 0.1–0.5 + Main-0 smoke
Stage 1:  Main-第一部分 主实验（1.1 Worst In-Domain → 1.2 external）
Stage 2:  Main-第二部分 正文（2.1 Flower 15 → 2.2-prep → 2.2 70c v13a）
Stage 3:  Baseline-0 smoke + Baseline-第一部分 主实验（1.1 Flower → 1.2 Worst MD → 1.3 external）
Stage 4:  Baseline-第二部分 正文（2.1 nearest 0-GPU → 2.2 70c baseline）
Stage 5:  ⛔ Main-第三部分 附录（3.1 common → 3.2 r16 → 3.3 D1 held-out 15）
Stage 6:  ⛔ Baseline-第三部分 附录（3.1 common → 3.2 r16 → 3.3 d1 audit 0-GPU）
Stage 7:  第三部分总体验收 Z1–Z2
```

**禁止：** 在 Stage 1–4 未完成时启动 Main-3.x / Baseline-3.x 的 GPU 训练；禁止 Baseline-2.1 在 Main-2.1 未完成时汇总。

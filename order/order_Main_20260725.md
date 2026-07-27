# FedPLoRA-OS 主算法缺口实验命令（2026-07-25）

######### FedPLoRA-OS 主方法：主实验、正文实验与附录实验补齐-20260725 #########

> 本文件只安排 **FedPLoRA-OS 主算法及其内部路由诊断**，不重复启动 external baseline。baseline 训练与评估见 `order_baseline_20260725.md`。原始缺口编号严格对应 `Paper/AAAI-2027/Result/codex_CN_Result_20260725.md` 的 1--10 项。

## 【命令介绍】

本文件按论文位置分成三部分：

1. **主实验：协议一致的核心有效性**——补主方法在 D1/FlowerTune 上的 Worst In-Domain（原编号 1、10）及官方任务级外部评估（原编号 9）。
2. **正文实验：路由解释与规模扩展**——统一比较六类核心路由度量并附加隐式 `Delta W` 诊断，补齐 FlowerTune 五折三种子 1-example onboarding（原编号 3、4），并运行 70-client 主方法（原编号 6）。
3. **附录实验：稳健性与跨数据集复现**——补统一测试集的 Non-IID `alpha=0.5`（原编号 5）、LoRA `r=16`（原编号 7）和 D1 严格 held-out 五折三种子（原编号 8）。

主方法新增的正式 GPU 作业为：

| 部分 | 作业 | 数量 |
|---|---|---:|
| 主实验 | Worst In-Domain eval-only：D1/FlowerTune × 3 seeds | 6 |
| 主实验 | external eval：MMLU/PubMedQA/MBPP × 3 seeds | 3 个 launcher；内部顺序评估 |
| 正文 | FlowerTune 1-example + routing audit：5 offsets × 3 seeds | 15 |
| 正文 | 70-client FedPLoRA-OS | 3 |
| 附录 | common-test `alpha=0.5` FedPLoRA-OS | 3 |
| 附录 | LoRA `r=16` FedPLoRA-OS | 3 |
| 附录 | D1 strict held-out：5 offsets × 3 seeds | 15 |

每个 launcher 只对应一个 seed/offset/method 单元，使用独立日志和结果目录。不要把同一批命令全部指向 GPU 0。

## 【命令目的】

- 锁定正式协议为 `max_seq_length=256`，废弃此前默认 2048 的 matched-domain 输出。
- 证明路由优势来自 LoRA-​`B` 的 gauge-insensitive 子空间结构，而不是任意 raw-entry 距离、最近客户端或随机选择。
- 检验极低样本新客户端接入、参与者扩展、LoRA rank 扩展及 Non-IID 强度变化时，主方法结论是否保持。
- 将 token accuracy 的内部基准结论外推到社区任务指标；不使用 best-client cherry-picking。

## 【统一实验设置】

```text
服务器（训练/大多数评估）: 172.26.191.30, minghao
代码目录: /data2/minghao/code/FedPLoRA-main
环境: FedRepo2
基础模型: /data2/minghao/model/SmolLM2-135M

正式主协议:
rounds=1, local_epochs=1, lr=2e-4
LoRA r=8, alpha=16, dropout=0.05
batch_size=2, max_seq_length=256, bfloat16
eval_max_batches=0（完整测试集）
seeds={42,43,44}

D1: 35 clients / 7 domains / 5 clients per domain
FlowerTune-Mixed: 20 clients / 4 domains / 5 clients per domain
70-client: 70 clients / 7 domains / 10 clients per domain，冻结 D1 test
```

## 【实验产物位置说明】

```text
A100 主方法结果:
/data2/minghao/result/FedPLoRA/order_main_20260725/

A100 主方法 checkpoint:
/data2/minghao/model/trained_models_LW/order_main_20260725/

原训练节点 matched-domain 结果:
/data/yaominghao/gb/result/FedPLoRA/eval_only_main_20260725/

每个新训练单元:
<RESULT_ROOT>/<tag>/run_logs/*.log
<RESULT_ROOT>/<tag>/result_logs/<method>/*.json
<RESULT_ROOT>/<tag>/result_files/client_states/<method>/
<MODEL_ROOT>/<tag>/<method>/
```

---

# 0. 共同前置、协议修复与 smoke

## 0.1 将本轮需要的最新脚本同步到 A100 节点

在本地执行。external-eval 的三个文件必须一起同步，不能只覆盖入口脚本。

```bash
scp -o RemoteCommand=none -o RequestTTY=no \
  /Users/hawaiii/codex/FedPLoRA/FedPLoRA-main/scripts/RunScripts/run_eval_only_matched_domain.sh \
  /Users/hawaiii/codex/FedPLoRA/FedPLoRA-main/scripts/RunScripts/launch_eval_only_matched_domain_one.sh \
  /Users/hawaiii/codex/FedPLoRA/FedPLoRA-main/scripts/RunScripts/check_eval_only_matched_domain_jobs.sh \
  minghao@172.26.191.30:/data2/minghao/code/FedPLoRA-main/scripts/RunScripts/

scp -o RemoteCommand=none -o RequestTTY=no \
  /Users/hawaiii/codex/FedPLoRA/FedPLoRA-clean-sync-20260723/scripts/Analysis/external_lm_eval_datasets.py \
  /Users/hawaiii/codex/FedPLoRA/FedPLoRA-clean-sync-20260723/scripts/Analysis/prepare_external_lm_eval_hf_cache.py \
  /Users/hawaiii/codex/FedPLoRA/FedPLoRA-clean-sync-20260723/scripts/Analysis/run_external_lm_eval.py \
  minghao@172.26.191.30:/data2/minghao/code/FedPLoRA-main/scripts/Analysis/
```

原训练节点 `/data/yaominghao/gb/FedPLoRA` 也必须同步修复后的 `run_eval_only_matched_domain.sh`、`launch_eval_only_matched_domain_one.sh`、`check_eval_only_matched_domain_jobs.sh`、`fed_train_sft.py` 与 `summarize_matched_domain_eval.py`。该节点地址未写入现有实验记录，因此这里不伪造 SSH host；登录原节点后将文件放到相同相对路径再执行第 1.1 节。

## 0.2 A100 环境变量与代码硬检查

```bash
ssh minghao@172.26.191.30
exec bash
source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate FedRepo2

export CODE_DIR=/data2/minghao/code/FedPLoRA-main
export RESULT_ROOT=/data2/minghao/result/FedPLoRA/order_main_20260725
export MODEL_ROOT=/data2/minghao/model/trained_models_LW/order_main_20260725
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

python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/Analysis/eval_personalized.py \
  scripts/Analysis/checkpoint_manifest.py \
  scripts/Analysis/external_lm_eval_datasets.py \
  scripts/Analysis/prepare_external_lm_eval_hf_cache.py \
  scripts/Analysis/run_external_lm_eval.py \
  scripts/Analysis/summarize_matched_domain_eval.py \
  scripts/DataProcessScripts/repartition_with_frozen_test.py
bash -n scripts/RunScripts/run_20260713_one_experiment.sh
bash -n scripts/RunScripts/run_eval_only_matched_domain.sh
bash -n scripts/RunScripts/launch_eval_only_matched_domain_one.sh
bash -n scripts/RunScripts/check_eval_only_matched_domain_jobs.sh

grep -q 'EVAL_MAX_SEQ_LENGTH="${EVAL_MAX_SEQ_LENGTH:-256}"' scripts/RunScripts/run_eval_only_matched_domain.sh
grep -q -- '--max_seq_length "${EVAL_MAX_SEQ_LENGTH}"' scripts/RunScripts/run_eval_only_matched_domain.sh
```

最后两条 `grep` 任一失败都必须停止。它们是原编号 10 的修复门禁，防止再次以默认 2048 评估。

## 0.3 数据结构与正式 fingerprint 审计

```bash
python - <<'PY'
import collections
import json
import pathlib

checks = [
    (pathlib.Path("/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05"), 35, 7, 5),
    (pathlib.Path("/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05"), 20, 4, 5),
]
for root, n_clients, n_domains, per_domain in checks:
    for seed in (42, 43, 44):
        split = root / f"seed_{seed}"
        rows = json.loads((split / "clients.json").read_text(encoding="utf-8"))
        counts = collections.Counter(str(x["domain"]) for x in rows)
        assert len(rows) == n_clients, (split, len(rows))
        assert len(counts) == n_domains and set(counts.values()) == {per_domain}, (split, counts)
        for name in ("train.jsonl", "val.jsonl", "test_local.jsonl", "test_domain.jsonl"):
            assert (split / name).is_file(), split / name
        print("[data][ok]", split, dict(sorted(counts.items())))
PY

for SEED in 42 43 44; do
  python utilities/benchmark_fingerprint.py "$D1_ROOT/seed_${SEED}" \
    --output "$RESULT_ROOT/analysis/d1_fingerprint_seed${SEED}.json"
  python utilities/benchmark_fingerprint.py "$FLOWER_ROOT/seed_${SEED}" \
    --output "$RESULT_ROOT/analysis/flower_fingerprint_seed${SEED}.json"
done
```

论文回填只接受 D1/FlowerTune 正式 fingerprint 族；历史记录中的前缀分别为 `43f0ac1c` 与 `86603887`。若本机工具计算口径得到不同前缀，必须先核对 fingerprint 版本和 split 文件，不能通过修改表注掩盖差异。

## 0.4 新训练统一 launcher

```bash
launch_main_sft () {
  local tag="$1" benchmark="$2" n_clients="$3" seed="$4" rank="$5" alpha="$6" eval_cap="$7"
  shift 7
  local method="N7_${tag}"
  mkdir -p "$RESULT_ROOT/$tag/run_logs" \
           "$RESULT_ROOT/$tag/result_logs/$method" \
           "$RESULT_ROOT/$tag/result_files/client_states/$method" \
           "$MODEL_ROOT/$tag/$method"
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
    --model "$MODEL_135M" \
    --benchmark_dir "$benchmark" \
    --num_clients "$n_clients" \
    --agg_type fedplora_v13a_os \
    --rounds 1 --local_epochs 1 --lr 0.0002 \
    --lora_r "$rank" --lora_alpha "$alpha" --lora_dropout 0.05 \
    --batch_size 2 --max_seq_length 256 --torch_dtype bfloat16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --save_client_state_to_disk --gradient_checkpointing \
    --eval_personalization_metrics --eval_final_only --skip_post_agg_snapshots \
    --client_state_dir "$RESULT_ROOT/$tag/result_files/client_states/$method" \
    --metrics_output_dir "$RESULT_ROOT/$tag/result_logs/$method" \
    --save_run_checkpoint_dir "$MODEL_ROOT/$tag/$method" \
    --trained_models_root "$MODEL_ROOT/$tag" \
    --eval_max_batches "$eval_cap" --seed "$seed" --force_retrain "$@" \
    > "$RESULT_ROOT/$tag/run_logs/test20260725_main_${tag}.log" 2>&1 &
  echo $! > "$RESULT_ROOT/pids/${tag}.pid"
  echo "[launch] tag=$tag pid=$(cat "$RESULT_ROOT/pids/${tag}.pid") gpu=${GPU_ID:-0}"
}
```

## 0.5 smoke：训练、held-out 与 matched-domain 三条门禁

### S1. 主方法训练 smoke

```bash
GPU_ID=0 launch_main_sft smoke_v13a_seed42 "$D1_ROOT/seed_42" 35 42 8 16 1 \
  --train_max_steps_per_client 1 --max_train_samples_per_client 10
```

### S2. personalized routing smoke

```bash
RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_135M" \
BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 \
RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=1 \
nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_smoke_flower_route_probe1_seed42 \
  --seed 42 --split-seed 42 --run-id-prefix main_20260725_smoke_route \
  --gpu 0 -- \
  --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 \
  --few_shot_caps 1 --held_out_route_probe_samples 1 \
  --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle \
  --onboarding_accounting --schemes base,global,coldstart,coldstart_geom \
  --select_candidates global,coldstart,coldstart_geom \
  > "$RESULT_ROOT/launcher_logs/test20260725_main_smoke_route.launch.log" 2>&1 &
```

### S3. matched-domain runner smoke

该 smoke 在原训练节点执行，选择一个正式 `max_seq_length=256` 的主方法 result JSON；`EVAL_MAX_BATCHES=1` 仅验证流程，不进入论文。

```bash
cd /data/yaominghao/gb/FedPLoRA
SMOKE_JSON=$(find /data/yaominghao/gb/result/FedPLoRA/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42 -maxdepth 1 -name '*.json' | head -n 1)
test -n "$SMOKE_JSON"
CUDA_VISIBLE_DEVICES=0 EVAL_MAX_BATCHES=1 EVAL_MAX_SEQ_LENGTH=256 \
MATCHED_DOMAIN_OUTPUT_ROOT=/data/yaominghao/gb/result/FedPLoRA/eval_only_main_20260725/smoke \
bash FedPLoRA-main/scripts/RunScripts/run_eval_only_matched_domain.sh "$SMOKE_JSON"
```

S1/S2 必须有最终 JSON 且无 traceback；S2 JSON 必须含 `strict_held_out.route_audits.subspace`；S3 日志必须显式打印 `max_seq_length=256`。三项通过后再跑正式命令。

---

# 第一部分：主实验——协议一致的核心有效性

## 1.1 【原编号 1、10】FedPLoRA-OS Worst In-Domain，D1/FlowerTune × 3 seeds

**为何放主实验：** In-Domain 衡量平均匹配领域效果，Worst In-Domain 衡量最弱领域是否仍可靠。它不能与 broad-domain Worst 混用。

在原训练节点执行：

```bash
set -euo pipefail
cd /data/yaominghao/gb/FedPLoRA

export GB_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MD_ROOT="$GB_RESULT_ROOT/eval_only_main_20260725"
export MD_RUNNER="${MD_RUNNER:-scripts/RunScripts/run_eval_only_matched_domain.sh}"
export MD_LAUNCHER="${MD_LAUNCHER:-scripts/RunScripts/launch_eval_only_matched_domain_one.sh}"
export MD_STATUS="${MD_STATUS:-scripts/RunScripts/check_eval_only_matched_domain_jobs.sh}"
export MD_SUMMARIZER="${MD_SUMMARIZER:-scripts/Analysis/summarize_matched_domain_eval.py}"
mkdir -p "$MD_ROOT/d1" "$MD_ROOT/flowertune" "$MD_ROOT/logs" "$MD_ROOT/pids" "$MD_ROOT/meta"

find_one_json () {
  local dir="$1"
  mapfile -t hits < <(find "$dir" -maxdepth 1 -type f -name '*.json' | sort)
  [[ "${#hits[@]}" -eq 1 ]] || { echo "[source][error] expected one JSON: $dir, got ${#hits[@]}" >&2; return 2; }
  printf '%s\n' "${hits[0]}"
}

D1_OURS_42=$(find_one_json "$GB_RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42")
D1_OURS_43=$(find_one_json "$GB_RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43")
D1_OURS_44=$(find_one_json "$GB_RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44")
FLOWER_OURS_42=$(find_one_json "$GB_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed42/result_logs/N7_ours_flower_v13a")
FLOWER_OURS_43=$(find_one_json "$GB_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed43/result_logs/N7_ours_flower_v13a")
FLOWER_OURS_44=$(find_one_json "$GB_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed44/result_logs/N7_ours_flower_v13a")

bash "$MD_LAUNCHER" d1_ours_seed42 "$D1_OURS_42" "$MD_ROOT/d1" 0
bash "$MD_LAUNCHER" d1_ours_seed43 "$D1_OURS_43" "$MD_ROOT/d1" 1
bash "$MD_LAUNCHER" d1_ours_seed44 "$D1_OURS_44" "$MD_ROOT/d1" 2
bash "$MD_LAUNCHER" flower_ours_seed42 "$FLOWER_OURS_42" "$MD_ROOT/flowertune" 3
bash "$MD_LAUNCHER" flower_ours_seed43 "$FLOWER_OURS_43" "$MD_ROOT/flowertune" 4
bash "$MD_LAUNCHER" flower_ours_seed44 "$FLOWER_OURS_44" "$MD_ROOT/flowertune" 5
```

上面六条只负责启动，正常情况下每条只打印 `pid/log` 后立刻返回；不要在这个启动块里写 `wait`。之后可以关闭 SSH 或继续复制其它实验命令，日志与 pid 会保留在 `$MD_ROOT/logs` 和 `$MD_ROOT/pids`。

随时查看状态（不阻塞）：

```bash
bash "$MD_STATUS" "$MD_ROOT"
```

全部完成后再汇总（只在 `bash "$MD_STATUS" "$MD_ROOT"` 显示 `running=0 failed=0 unknown=0` 后执行）：

```bash
[[ "$(find "$MD_ROOT/d1" -name '*_matched_domain.json' | wc -l)" -eq 3 ]]
[[ "$(find "$MD_ROOT/flowertune" -name '*_matched_domain.json' | wc -l)" -eq 3 ]]
python "$MD_SUMMARIZER" "$MD_ROOT/d1" | tee "$MD_ROOT/d1_ours_summary.tsv"
python "$MD_SUMMARIZER" "$MD_ROOT/flowertune" | tee "$MD_ROOT/flower_ours_summary.tsv"
grep -R 'max_seq_length=256' "$MD_ROOT/logs"
```

论文使用 `in_domain_domain_test_token_accuracy` 与 `in_domain_domain_test_worst_token_accuracy`；不得回填旧 2048 输出。

## 1.2 【原编号 9】官方任务级外部评估：FedPLoRA-OS × 3 seeds

**为何放主实验：** 该表验证 token-accuracy 结论能否迁移到 MMLU、PubMedQA、MBPP 的官方指标。对每个 task，`domain_clients` 是该声明领域全部路由客户端 adapter 的无权宏平均，禁止挑选最佳客户端。

### 1.2.1 任务与离线数据门禁

```bash
python -m pip show lm_eval >/dev/null 2>&1 || python -m pip install 'lm_eval[hf]>=0.4.8,<0.5'
export HF_CACHE_ROOT="$CODE_DIR/data/external_lm_eval_hf_cache"
python scripts/Analysis/prepare_external_lm_eval_hf_cache.py \
  --cache_root "$HF_CACHE_ROOT" --tasks mmlu,pubmedqa,mbpp --verify_only
echo "[external][ok] offline cache verified for mmlu,pubmedqa,mbpp"
echo "[external][note] skip 'lm_eval ls tasks' grep: this lm-eval version may not list repo-local overrides/aliases."
```

若缓存尚未准备好，优先让合作者 `git pull` 后用统一脚本准备 cache；默认走 `hf-mirror.com`，准备完会自动离线复验：

```bash
cd "$CODE_DIR"
bash scripts/RunScripts/prepare_external_lm_eval_cache.sh probe
bash scripts/RunScripts/prepare_external_lm_eval_cache.sh prepare mmlu,pubmedqa,mbpp
bash scripts/RunScripts/prepare_external_lm_eval_cache.sh verify mmlu,pubmedqa,mbpp
```

`verify` 时出现 `couldn't be found on the Hugging Face Hub (offline mode is enabled)` 是正常离线提示，不是失败；只要最后出现 `[hf-cache][ok] offline cache ready` 就说明 cache 可用。

若镜像也不可达，在任意能联网机器上用同一脚本准备 `data/external_lm_eval_hf_cache/`，再 `rsync` 到服务器同一路径；正式 E2 评测继续使用 `--hf_cache_dir "$HF_CACHE_ROOT"` 离线读取。不要直接改用 ModelScope 版数据集替代 MMLU/PubMedQA/MBPP，否则 lm-eval task YAML、dataset config 与论文可复现口径会变化。当前脚本没有 FiQA 的稳定 task/cache 映射，因此本轮**不伪造 FiQA alias**；只有服务器版本先通过 cache verify 与 smoke 后才能另加。

注意：不要再执行旧版 `python -m lm_eval ls tasks | grep mmlu/pubmedqa/mbpp` 门禁。该命令只反映 lm-eval 内置 task 表，可能不包含本仓库动态生成的 PubMedQA override，也可能不暴露兼容 alias；只要本节 `verify_only` 末尾出现 `[hf-cache][ok] offline cache ready`，就进入下一步 adapter export / smoke。

### 1.2.2 解析 checkpoint 并导出三种子部署 adapter

```bash
export CKPT_SEARCH_ROOTS="/data2/minghao/model/trained_models_LW /data2/minghao/result/FedPLoRA"
python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS \
  --output "$RESULT_ROOT/analysis/checkpoint_manifest_external.json"

export_ours_adapter_async () {
  local seed="$1" gpu="$2"
  local ckpt
  ckpt=$(python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS --resolve \
    --agg_type fedplora_v13a_os --seed "$seed" --model_contains SmolLM2-135M \
    --benchmark_contains "A100_domain_benchmark_35c_dir05/seed_${seed}")
  nohup env CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/time -v python -u tasks/fed_train_sft.py \
    --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_${seed}" \
    --agg_type fedplora_v13a_os --seed "$seed" \
    --eval_only_from_checkpoint "$ckpt" \
    --metrics_output_dir "$RESULT_ROOT/external_export/ours_seed${seed}/metrics" \
    --client_state_dir "$RESULT_ROOT/external_export/ours_seed${seed}/scratch" \
    --export_eval_adapter_dir "$RESULT_ROOT/external_adapters/ours_seed${seed}" \
    --export_eval_adapter_only --eval_max_batches 0 --batch_size 2 \
    --max_seq_length 256 --torch_dtype bfloat16 --eval_personalization_metrics \
    > "$RESULT_ROOT/launcher_logs/test20260725_export_ours_seed${seed}.log" 2>&1 &
  echo $! > "$RESULT_ROOT/pids/export_ours_seed${seed}.pid"
  echo "[export][launch] seed=${seed} gpu=${gpu} pid=$(cat "$RESULT_ROOT/pids/export_ours_seed${seed}.pid") log=$RESULT_ROOT/launcher_logs/test20260725_export_ours_seed${seed}.log"
}

export_ours_adapter_async 42 0
export_ours_adapter_async 43 1
export_ours_adapter_async 44 2
```

查看 adapter export 状态（不阻塞）：

```bash
for p in "$RESULT_ROOT"/pids/export_ours_seed*.pid; do
  pid=$(cat "$p")
  tag=$(basename "$p" .pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "[export][running] $tag pid=$pid"
  elif grep -Eiq 'Traceback|CUDA out of memory|\\[error\\]|Exit status: [1-9]' "$RESULT_ROOT/launcher_logs/test20260725_${tag}.log"; then
    echo "[export][failed] $tag log=$RESULT_ROOT/launcher_logs/test20260725_${tag}.log"
  else
    echo "[export][exited] $tag"
  fi
done
for SEED in 42 43 44; do
  test -f "$RESULT_ROOT/external_adapters/ours_seed${SEED}/adapter_export_manifest.json" \
    && echo "[export][ok] seed=${SEED}" \
    || echo "[export][pending] seed=${SEED}"
done
```

只有三份 `adapter_export_manifest.json` 都显示 `[export][ok]` 后，才执行 1.2.3 的 smoke 与正式 external eval。

### 1.2.3 smoke 与正式评估

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Analysis/run_external_lm_eval.py \
  --adapter_manifest "$RESULT_ROOT/external_adapters/ours_seed42/adapter_export_manifest.json" \
  --tasks pubmedqa:medical --mode both --limit 10 --device cuda:0 --batch_size auto \
  --hf_cache_dir "$HF_CACHE_ROOT" --output_dir "$RESULT_ROOT/external_smoke/ours_seed42"
```

```bash
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/ours_seed42/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/ours_seed42" > "$RESULT_ROOT/launcher_logs/test20260725_external_ours_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_ours_seed42.pid"
```

```bash
CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/ours_seed43/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/ours_seed43" > "$RESULT_ROOT/launcher_logs/test20260725_external_ours_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_ours_seed43.pid"
```

```bash
CUDA_VISIBLE_DEVICES=2 nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/ours_seed44/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --hf_cache_dir "$HF_CACHE_ROOT" --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/ours_seed44" > "$RESULT_ROOT/launcher_logs/test20260725_external_ours_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/external_ours_seed44.pid"
```

MBPP 会执行生成代码，必须在隔离环境运行；若节点不满足该条件，删去 `mbpp:code` 和 `--confirm_run_unsafe_code`，不得绕过安全门。验收产物为每个 seed 的 `external_eval_summary.json`。

---

# 第二部分：正文实验——路由解释、极低样本接入与参与者扩展

## 2.1 【原编号 3、4】FlowerTune 五折三种子 1-example + 六类核心路由度量

**作用：** 同一作业同时给出 principal-angle/canonical-correlation 子空间、raw-​`B` cosine、flattened-​`B` distance、nearest-client、random、oracle 六类核心对照，并附加隐式 `Delta W` cosine 诊断；指标包括 Route match、margin、Local、错误领域数和 CPU route time，同时补齐 5 offsets × 3 seeds 的完整 1-example 结果。`subspace` 是本文主张的 gauge-insensitive 度量；`nearest_client_subspace` 是检索 baseline，不另起 GPU 作业。

```bash
launch_flower_route () {
  local offset="$1" seed="$2" gpu="$3"
  local tag="flower_probe1_offset${offset}_seed${seed}"
  RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_135M" \
  BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 \
  RUN_TAG_DATASET=flowertune_mixed_20c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 \
  nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh \
    --kind personalized_eval --method "X2_${tag}" --seed "$seed" --split-seed "$seed" \
    --run-id-prefix main_20260725_flower_probe1 --gpu "$gpu" -- \
    --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset "$offset" \
    --few_shot_caps 1 --held_out_route_probe_samples 1 \
    --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle \
    --onboarding_accounting --schemes base,global,coldstart,coldstart_geom \
    --select_candidates global,coldstart,coldstart_geom \
    > "$RESULT_ROOT/launcher_logs/test20260725_${tag}.launch.log" 2>&1 &
  echo $! > "$RESULT_ROOT/pids/${tag}.pid"
}
```

每一行是一条独立正式作业；按实际空闲卡修改第三个参数。

```bash
launch_flower_route 0 42 0
launch_flower_route 0 43 1
launch_flower_route 0 44 2
launch_flower_route 1 42 3
launch_flower_route 1 43 4
launch_flower_route 1 44 5
launch_flower_route 2 42 6
launch_flower_route 2 43 7
```

查看第一波状态；全部结束且无失败后启动第二波：

```bash
for p in "$RESULT_ROOT"/pids/flower_probe1_*.pid; do
  pid=$(cat "$p")
  tag=$(basename "$p" .pid)
  log="$RESULT_ROOT/launcher_logs/test20260725_${tag}.launch.log"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[flower-route][running] $tag pid=$pid"
  elif grep -Eiq 'Traceback|CUDA out of memory|\\[error\\]|Exit status: [1-9]' "$log"; then
    echo "[flower-route][failed] $tag log=$log"
  else
    echo "[flower-route][exited] $tag"
  fi
done
```

第二波：

```bash
launch_flower_route 2 44 0
launch_flower_route 3 42 1
launch_flower_route 3 43 2
launch_flower_route 3 44 3
launch_flower_route 4 42 4
launch_flower_route 4 43 5
launch_flower_route 4 44 6
```

完整性与字段门禁：

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("/data2/minghao/result/FedPLoRA/order_main_20260725")
paths = sorted(root.glob("main_20260725_flower_probe1_seed*/result_logs/X2_flower_probe1_offset*_seed*.json"))
assert len(paths) == 15, len(paths)
required = {"flat_b_cosine", "subspace", "relative_l2", "delta_w_cosine", "nearest_client_subspace", "random", "oracle"}
for path in paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    held = row.get("strict_held_out") or {}
    audits = held.get("route_audits") or {}
    assert required <= set(audits), (path, sorted(audits))
    assert (row.get("onboarding_accounting") or {}).get("enabled") is True, path
print("[flower-route][ok]", len(paths), "fold-seed units")
PY
```

## 2.2 【原编号 6】70-client FedPLoRA-OS × 3 seeds

### 2.2.1 构建冻结测试集的 70-client split（0-GPU，只执行一次）

```bash
set -euo pipefail
for SEED in 42 43 44; do
  python scripts/DataProcessScripts/repartition_with_frozen_test.py \
    --reference_split "$D1_ROOT/seed_${SEED}" --output_dir "$D1_70C_ROOT" \
    --num_clients_per_domain 10 --min_samples_per_client 25 --seed "$SEED" \
    --partition dirichlet --dirichlet_alpha 0.5 --subtopic kmeans --n_subtopics 10
  cmp -s "$D1_70C_ROOT/seed_${SEED}/test_domain.jsonl" "$D1_ROOT/seed_${SEED}/test_domain.jsonl"
done
```

### 2.2.2 三个正式作业

```bash
GPU_ID=0 launch_main_sft 70c_v13a_seed42 "$D1_70C_ROOT/seed_42" 70 42 8 16 0
GPU_ID=1 launch_main_sft 70c_v13a_seed43 "$D1_70C_ROOT/seed_43" 70 43 8 16 0
GPU_ID=2 launch_main_sft 70c_v13a_seed44 "$D1_70C_ROOT/seed_44" 70 44 8 16 0
```

与 baseline 文档中的 70c Normal/FedALT 共同报告 Local、In-Domain、通信量，以及 v13a JSON 中的簇数/路由纯度。35c 对照使用正式主表结果，不重复训练。

---

# 第三部分：附录实验——稳健性、容量与跨数据集 held-out

## 3.1 【原编号 5】统一测试集 Non-IID `alpha=0.5`：FedPLoRA-OS × 3 seeds

**作用：** 与既有 IID、`alpha=0.1` 端点组成同一 frozen test 下的三档异构曲线。旧的不同 test-fingerprint `alpha=0.5` 结果作废。

### 3.1.1 重建共同测试集 split（0-GPU，只执行一次）

```bash
set -euo pipefail
for SEED in 42 43 44; do
  python scripts/DataProcessScripts/repartition_with_frozen_test.py \
    --reference_split "$IID_ROOT/seed_${SEED}" --output_dir "$DIR05_COMMON_ROOT" \
    --num_clients_per_domain 5 --seed "$SEED" --partition dirichlet \
    --dirichlet_alpha 0.5 --subtopic kmeans --n_subtopics 10
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$IID_ROOT/seed_${SEED}/test_domain.jsonl"
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$DIR01_ROOT/seed_${SEED}/test_domain.jsonl"
done
```

任一 `cmp` 失败即停止；不能采用旧 checkpoint 与相交测试子集的 eval-only 修补方案。

### 3.1.2 三个正式作业

```bash
GPU_ID=0 launch_main_sft common_a05_v13a_seed42 "$DIR05_COMMON_ROOT/seed_42" 35 42 8 16 0
GPU_ID=1 launch_main_sft common_a05_v13a_seed43 "$DIR05_COMMON_ROOT/seed_43" 35 43 8 16 0
GPU_ID=2 launch_main_sft common_a05_v13a_seed44 "$DIR05_COMMON_ROOT/seed_44" 35 44 8 16 0
```

## 3.2 【原编号 7】LoRA rank `r=16`：FedPLoRA-OS × 3 seeds

保持 `alpha/r=2`，即 `r=16, alpha=32`。与 baseline 文档的 Normal/FedALT 同协议比较 Local、Macro、双向通信量和相对通信占比。

```bash
GPU_ID=0 launch_main_sft r16_v13a_seed42 "$D1_ROOT/seed_42" 35 42 16 32 0
GPU_ID=1 launch_main_sft r16_v13a_seed43 "$D1_ROOT/seed_43" 35 43 16 32 0
GPU_ID=2 launch_main_sft r16_v13a_seed44 "$D1_ROOT/seed_44" 35 44 16 32 0
```

## 3.3 【原编号 8】D1 strict held-out：5 offsets × 3 seeds

**作用：** 使用与 FlowerTune 相同的每域排除一个参与客户端、probe、路由、专家构造和评估逻辑，验证新客户端专家复用不是 Finance 领域特例。headline probe 为 10 examples，同时保留 Local 1/5/10 上界。

```bash
launch_d1_heldout () {
  local offset="$1" seed="$2" gpu="$3"
  local tag="d1_heldout_offset${offset}_seed${seed}"
  RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" MODEL_PATH="$MODEL_135M" \
  BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 \
  RUN_TAG_DATASET=a1009k_35c_dir05 MAX_SEQ_LENGTH=256 PIPELINE_EVAL_MAX_BATCHES=0 \
  nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh \
    --kind personalized_eval --method "X2_${tag}" --seed "$seed" --split-seed "$seed" \
    --run-id-prefix main_20260725_d1_heldout --gpu "$gpu" -- \
    --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset "$offset" \
    --few_shot_caps 1,5,10 --held_out_route_probe_samples 10 \
    --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle \
    --onboarding_accounting --schemes base,global,coldstart,coldstart_geom \
    --select_candidates global,coldstart,coldstart_geom \
    > "$RESULT_ROOT/launcher_logs/test20260725_${tag}.launch.log" 2>&1 &
  echo $! > "$RESULT_ROOT/pids/${tag}.pid"
}

launch_d1_heldout 0 42 0
launch_d1_heldout 0 43 1
launch_d1_heldout 0 44 2
launch_d1_heldout 1 42 3
launch_d1_heldout 1 43 4
launch_d1_heldout 1 44 5
launch_d1_heldout 2 42 6
launch_d1_heldout 2 43 7
```

查看第一波状态；全部结束且无失败后启动第二波：

```bash
for p in "$RESULT_ROOT"/pids/d1_heldout_*.pid; do
  pid=$(cat "$p")
  tag=$(basename "$p" .pid)
  log="$RESULT_ROOT/launcher_logs/test20260725_${tag}.launch.log"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[d1-heldout][running] $tag pid=$pid"
  elif grep -Eiq 'Traceback|CUDA out of memory|\\[error\\]|Exit status: [1-9]' "$log"; then
    echo "[d1-heldout][failed] $tag log=$log"
  else
    echo "[d1-heldout][exited] $tag"
  fi
done
```

第二波：

```bash
launch_d1_heldout 2 44 0
launch_d1_heldout 3 42 1
launch_d1_heldout 3 43 2
launch_d1_heldout 3 44 3
launch_d1_heldout 4 42 4
launch_d1_heldout 4 43 5
launch_d1_heldout 4 44 6
```

验收必须得到 15 个 JSON、每个 JSON 7 个 held-out clients，共 105 条路由；同时检查 `nearest_client_subspace` 和 `subspace` 均存在。baseline 文档只读取这些相同输出做检索对照，不重复跑 GPU。

---

# 4. 总体验收、回填字段与执行顺序

## 4.1 新训练完整性

```bash
python - <<'PY'
import pathlib
root = pathlib.Path("/data2/minghao/result/FedPLoRA/order_main_20260725")
expected = []
expected += [f"70c_v13a_seed{s}" for s in (42,43,44)]
expected += [f"common_a05_v13a_seed{s}" for s in (42,43,44)]
expected += [f"r16_v13a_seed{s}" for s in (42,43,44)]
missing = [name for name in expected if not list((root/name/"result_logs").rglob("*.json"))]
assert not missing, missing
assert len(list(root.glob("main_20260725_flower_probe1_seed*/result_logs/X2_*.json"))) == 15
assert len(list(root.glob("main_20260725_d1_heldout_seed*/result_logs/X2_*.json"))) == 15
print("[main][ok] formal groups complete")
PY
```

## 4.2 建议执行顺序

```text
Stage 0: 代码同步 -> py_compile/bash -n -> fingerprint -> 三类 smoke
Stage 1 (P0): FlowerTune routing/1-example 15 jobs
Stage 2 (P0): 正确协议的 Worst In-Domain 6 jobs
Stage 3 (P0/P1): external adapter export -> external smoke -> 3-seed formal eval
Stage 4 (P1): 70-client split -> v13a 3 jobs
Stage 5 (P1): common-test alpha=0.5 split -> v13a 3 jobs
Stage 6 (P1/P2): D1 strict held-out 15 jobs
Stage 7 (P2): r16 v13a 3 jobs
```

## 4.3 停止条件

1. matched-domain 日志未打印 `max_seq_length=256`：立即停止，结果无效。
2. shared-test 任一 `cmp` 失败或发现 train/test overlap：停止 Non-IID 扫描。
3. FlowerTune 15 个 fold-seed 单元不全、任一 route audit 字段缺失：不得汇总 1-example 或路由度量表。
4. external task alias/离线缓存检查失败：不得改名绕过；FiQA 在稳定注册前继续列为未完成。
5. MBPP 非隔离执行：本轮只跑 MMLU/PubMedQA，并在结果文档中明确 MBPP 未执行。

## 4.4 论文回填指标

| 缺口 | 回填指标 |
|---|---|
| 原 1 | In-Domain、Worst In-Domain，mean±std over 3 seeds |
| 原 3 | Route match、mean margin、Local、wrong-domain count、CPU route time |
| 原 4 | 1-example Local/Worst/ΔGlobal、route match、payload、wall time |
| 原 5 | IID/0.5/0.1 的 Local 与相对 Normal 增益 |
| 原 6 | 35c/70c Local、通信量、簇数与路由纯度 |
| 原 7 | r8/r16 Local、Macro、Comm 与通信折扣 |
| 原 8 | D1 15 fold-seed、105 held-out routes 的 Local/Worst/route match |
| 原 9 | MMLU acc、PubMedQA acc、MBPP pass@1；global 与 routed macro 分列 |

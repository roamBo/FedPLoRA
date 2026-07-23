# FedPLoRA 20260723 投稿前统一实验命令（gb 服务器 · 合并修正版）

######### FedPLoRA 1-shot、router、common-test修复、YOCO与补充实验统一入口（gb 单卡 GPU1）-20260723 #########

> 由 `order/order_20260723.md`（合并 sup 后的最新版）适配 gb。章节与命令格式保持原文；仅换服务器路径、conda、GPU，并写入 gb 已知坑。  
> 原则：已完成结果不重复训练；每条正式命令只运行一个实验、产生一个 PID 和一个独立产物根。  
> **旧 C0–C7 intersection eval-only 已停用**；common-test 请走 `7.2-A0 → A1–A6`（gb 上 dir05 train + IID test 会 ~2100 overlap，不要强行 C1）。  
> YOCO、70-client、r16 等 sup 实验在第九部分之后；先完成 P0 主线与 7.2-A，再按资源追加。

## 先看：运行顺序总览

```text
0. 已完成本地 0-GPU 审计：不要重复跑。
1. 服务器前置：同步新增脚本、设置环境、检查数据与模型。
2. GPU smoke：先跑 S1-S3，确认无 traceback 和 JSON 正常。
3. P0 主线：FlowerTune full-fold 1-shot G1-G12，补齐 60-client 1-example route probe。
4. P0 主线：Qwen3B seed42 probe=1 G13，对齐 3B cold-start 协议。
5. P1 机制：canonical 9k states G14-G16，然后 CPU spectrum。
6. P0 router 闭环：R1-R3 补 offset0 router/onboarding，与 G1-G12 合并验收。
7. 报错修复：只跑 7.2-A frozen-test repartition + A1-A6 重训；旧 C0-C7 intersection 已停用。
8. P2 external eval：adapter export 后跑 lm-eval。
9. 额外新增实验：YOCO-FlowerTune、YOCO-scale、70-client、r16，按资源放到最后跑。
```

## 如果已经卡在 common-test 报错

如果日志出现类似：

```text
[common-test][error] source train/val intersects common test: 2099 examples
Rebuild all heterogeneity splits from one frozen shared test set; do not use eval-only on this pair.
```

处理方式：

1. 停止继续执行旧 `C1/C2-C7` eval-only 路径。
2. 确认服务器已经同步 `scripts/DataProcessScripts/repartition_with_frozen_test.py`。
3. 直接从本文第七部分 `7.2-A0` 开始，先构建 `domain_benchmark_35c_dir05_common_test_v2`。
4. 三个 seed 的 `cmp` 和 35-client 检查全部通过后，再跑 `A1-A6` 六条 GPU 重训。
5. 如果 reference IID split 自身泄漏，说明旧数据不能修补，必须重建 IID/dir0.1/dir0.5 后重训；不要删 guard。

## 【命令介绍】

本文件包含以下任务：

1. 已完成的 0-GPU 证据审计：nested-fold 统计、60-client margin、probe-size、fingerprint、通信和 router reliability。
2. 3 条 GPU smoke：FlowerTune full-fold 1-shot、Qwen3B probe 对齐、canonical 9k client-state 生成。
3. P0：补 FlowerTune offset1–4 × seeds42/43/44 的 1-example route probe，共 12 条 GPU probe/eval 任务。offset0 三 seed 已完成，不重复运行。
4. P0：补 Qwen2.5-3B seed42 的 probe=1，共 1 条 GPU probe/eval 任务，使三 seed 均为 probe=1。
5. P1：重新生成 D1 canonical 9k 的三 seed client states，共 3 条 GPU 训练；随后执行 CPU-only layer×module spectrum。

原主线 GPU 任务共 **16 条**：12 + 1 + 3。第七部分另补 3 条 offset0 router、**7.2-A** 6 条 full-common-test dir0.5 重训练、2 条 external-task eval；**旧 C0–C7 intersection eval-only 已停用**。第九部分以后为 YOCO/70c/r16 等 sup 队列（`order_0723_sup` 结果根）。

## 【命令目的】

- 判断 `as few as one example` 是否能从 offset0 的 12/12 扩展到完整 60 clients。
- 统一 Qwen3B cold-start 的 probe-size，消除 seed42=10、seed43/44=1 的混合协议。
- 为 canonical 9k 的 layer×module spectrum 补齐真实输入状态，避免继续使用旧 19k states。
- 保持训练算法、v13a 聚合和既有 baseline 不变。

## 【重要口径】

1. 当前 `eval_personalized.py` 中名为 `coldstart_geom` 的 held-out route，实际实现是**拼平后的 LoRA-B cosine**，不是 principal-angle router。下列命令验证的是当前部署 evaluator，不能把结果改写成“principal-angle router 60/60”。
2. `coldstart` 使用真实 domain，是 oracle-domain 上界；`coldstart_geom` 使用带 response 的本地 SFT probe 样本，是 domain-label-free，但不是 zero-data。
3. 每个 held-out fold 同时留出每个域一个 client；应写 `client-held-out fold, one client per domain`，不写经典 leave-one-client-out。
4. full-fold 1-shot 的预注册通过条件：route match ≥95%、15/15 fold-seed 的 ΔGlobal 为正、Local 相对 10-shot 无实质退化。
5. spectrum 的 CPU 分析不需要 GPU；gb 需先用 G14–G16 生成 dir05 的 35-client states。旧 19k / minghao A100 states 不得混用。

## 【命令设置】

```text
服务器: gb / yaominghao
conda: fedplora
代码目录: /data/yaominghao/gb/FedPLoRA

结果根:
/data/yaominghao/gb/result/FedPLoRA/order_0723

模型根:
/data/yaominghao/gb/models/trained_models_LW/order_0723

模型:
/data/yaominghao/gb/models/SmolLM2-135M
/data/yaominghao/gb/models/Qwen2.5-3B
/data/yaominghao/gb/models/SmolLM2-1.7B  # sup/YOCO-scale；缺则先下载

D1 (gb，非 A100 9k；不生成 A100_*):
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05/seed_{42,43,44}
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_iid/seed_{42,43,44}
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir01/seed_{42,43,44}
common-test 重训输出:
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05_common_test_v2/seed_{42,43,44}

FlowerTune-Mixed:
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_{42,43,44}

70-client frozen-test (sup):
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_70c_dir05_frozen_test/seed_{42,43,44}

GPU: 默认物理 1 号卡（export GPU_ID=1）；单卡串行，勿同卡堆 nohup

训练/评测:
rounds=1, local_epochs=1, lr=2e-4
LoRA r=8, alpha=16, dropout=0.05（r16: r=16, alpha=32）
batch=2, max_seq_length=256, dtype=bfloat16
formal eval_max_batches=0
smoke eval_max_batches=1, max_steps=1, max_train_samples_per_client=10
```

【路径对照（minghao → gb）】

| 项 | order_20260723.md | order_gb_0723new.md |
| --- | --- | --- |
| conda | `FedRepo2` | `fedplora` |
| 代码 | `/data2/minghao/code/FedPLoRA-main` | `/data/yaominghao/gb/FedPLoRA` |
| 模型根 | `/data2/minghao/model/` | `/data/yaominghao/gb/models/` |
| 结果 | `/data2/minghao/result/FedPLoRA/` | `/data/yaominghao/gb/result/FedPLoRA/` |
| D1 | `A100_domain_benchmark_35c_dir05` | `domain_benchmark_35c_dir05` |
| GPU | `GPU_ID=0/1` | 默认 `GPU_ID=1` |

【gb / preflight 防坑（必须遵守）】

```text
1. 每条 GPU 命令前先：cd /data/yaominghao/gb/FedPLoRA && export GPU_ID=1
   （或 export PATH=.../fedplora/bin；勿在 result/order_* 下裸粘贴相对路径）。
2. nohup 子壳禁止 set -u + source conda.sh（PS1 unbound）；用 PATH=.../fedplora/bin。
3. gb 无 A100_*；D1 一律 domain_benchmark_35c_dir05。缺 seed43/44 先 build。
4. 单卡串行：G/R/A1–A6/sup 正式训练同卡一次只开一条；3B 独占 GPU1。
5. personalized-eval 不加 --force_retrain；G14–G16 / A1–A6 / sup 必须 --force_retrain。
6. HF 不可达：export HF_ENDPOINT=https://hf-mirror.com。
7. 旧 C1 build_common_test_benchmark（dir05 train + IID test）在 gb 会因 ~2.1k overlap 失败；改跑 7.2-A0 repartition。
8. 新脚本用 git pull；需含 repartition_with_frozen_test.py。
9. 7.2-A0 第二个 cmp（dir05 vs dir01 test）失败说明 iid/dir01 未共 test，需先对齐三套 benchmark 再 repartition。
```


## 【实验产物位置说明】

```text
run_logs:
/data/yaominghao/gb/result/FedPLoRA/order_0723/<RUN_ID>/run_logs/

result_logs:
/data/yaominghao/gb/result/FedPLoRA/order_0723/<RUN_ID>/result_logs/

result_files/client_states:
/data/yaominghao/gb/result/FedPLoRA/order_0723/<RUN_ID>/result_files/client_states/

pids:
/data/yaominghao/gb/result/FedPLoRA/order_0723/<RUN_ID>/pids/

launcher logs:
/data/yaominghao/gb/result/FedPLoRA/order_0723/launcher_logs/

9k spectrum:
/data/yaominghao/gb/result/FedPLoRA/order_0723/spectrum_9k/
```

## 【实验运行涉及场景】

```text
P0-A: FlowerTune-Mixed / 20 clients / 4 domains / 5 held-out offsets / 3 seeds / 1-example probe
P0-B: Qwen2.5-3B / FlowerTune offset0 / 3-seed aligned 1-example probe
P1: D1 canonical 9k / 35 clients / 7 domains / v13a one-round states / CPU layer spectrum
```

---

# 第一部分：已直接执行的 0-GPU 项

以下工作已经在本地完成，不要在服务器重复运行：

```text
输入:
/Users/hawaiii/codex/1_experiment/Result/FedPLoRA/

输出:
/Users/hawaiii/codex/1_experiment/Result/FedPLoRA/analysis/20260723_zero_gpu/

主 artifact:
paper_evidence_audit_20260723.md/json
heldout_fold_seed_units.csv
heldout_client_units.csv
probe_offset0_units.csv
benchmark_fingerprint_audit.csv
communication_audit.csv
router_reliability.md/json
```

已核结果：

- FlowerTune 10-example：15/15 fold-seed 为正，60/60 route；ΔGlobal = 8.616±0.991 pp（按 3 个 seed 均值统计）。
- seed-level t 95% CI：[6.154, 11.077] pp。
- seed-cluster bootstrap 95% CI：[7.530, 9.472] pp。
- two-level hierarchical bootstrap 95% CI：[7.539, 9.599] pp。
- 60-client margin–gain：Pearson r=0.944，Spearman ρ=0.825；仅作描述性分析，不能作因果/校准结论。
- offset0 的 probe 1/2/5 均为 12/12；这仍不能替代 full-fold 1-shot。
- 新增 `scripts/Analysis/l1_layer_spectrum.py` 已用旧 states 完成 CPU 功能验证，但未产生 canonical 9k 论文结果。

---

# 第二部分：服务器前置命令

## 0.1 代码同步（gb）

```bash
cd /data/yaominghao/gb/FedPLoRA && git pull
```

确保含：`repartition_with_frozen_test.py`、`l1_layer_spectrum.py`、`pt_reader.py`、
`build_common_test_benchmark.py`、`checkpoint_manifest.py`、`run_external_lm_eval.py`、
`eval_personalized.py`、`fed_train_sft.py`、`summarize_fedplora_results.py`、`print_sft_comm_profile.py`。

## 0.2 服务器环境

```bash
# 登录 gb 后执行（用户: yaominghao）
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && exec bash
# 交互壳可用 conda；若已 set -u：export PS1="${PS1-}"
source /data/yaominghao/miniconda3/etc/profile.d/conda.sh
conda activate fedplora

export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW/order_0723
export MODEL_135M=/data/yaominghao/gb/models/SmolLM2-135M
export MODEL_3B=/data/yaominghao/gb/models/Qwen2.5-3B
export D1_ROOT=/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05
export FLOWER_ROOT=/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05
export GPU_ID=${GPU_ID:-1}

cd "$CODE_DIR"
mkdir -p "$RESULT_ROOT/launcher_logs" "$RESULT_ROOT/spectrum_9k" "$MODEL_ROOT"

python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/Analysis/eval_personalized.py \
  scripts/Analysis/build_common_test_benchmark.py \
  scripts/Analysis/checkpoint_manifest.py \
  scripts/Analysis/run_external_lm_eval.py \
  scripts/DataProcessScripts/repartition_with_frozen_test.py \
  scripts/Analysis/l1_layer_spectrum.py \
  scripts/Analysis/pt_reader.py
bash -n scripts/RunScripts/run_20260713_one_experiment.sh
```

## 0.3 数据与模型检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import collections
import json
import pathlib

checks = [
    (pathlib.Path("/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05"), 35, 7),
    (pathlib.Path("/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05"), 20, 4),
]
for root, expected_clients, expected_domains in checks:
    for seed in (42, 43, 44):
        path = root / f"seed_{seed}" / "clients.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        counts = collections.Counter(str(row["domain"]) for row in rows)
        assert len(rows) == expected_clients, (path, len(rows))
        assert len(counts) == expected_domains, (path, counts)
        assert set(counts.values()) == {5}, (path, counts)
        print("[preflight][ok]", path.parent, dict(sorted(counts.items())))

for model in ("/data/yaominghao/gb/models/SmolLM2-135M", "/data/yaominghao/gb/models/Qwen2.5-3B"):
    path = pathlib.Path(model) / "config.json"
    assert path.is_file(), path
    print("[preflight][ok] model", model)
PY
```

---

# 第三部分：GPU smoke（与正式命令分开）

## S1. FlowerTune offset1 / seed42 / 1-example

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset1_smoke_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260723_probe1_smoke --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom --max_steps 1 --max_train_samples_per_client 10 > "$RESULT_ROOT/launcher_logs/test20260723_smoke_flower_probe1_offset1_seed42.launch.log" 2>&1 &
```

## S2. Qwen3B / seed42 / probe=1

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_3B" RUN_TAG_MODEL=Qwen2.5-3B BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_qwen3b_flower_probe1_smoke_seed42 --seed 42 --split-seed 42 --run-id-prefix qwen3b_20260723_probe1_smoke --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom --max_steps 1 --max_train_samples_per_client 10 > "$RESULT_ROOT/launcher_logs/test20260723_smoke_qwen3b_probe1_seed42.launch.log" 2>&1 &
```

## S3. D1-9k state generation smoke

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb19k_35c_dir05 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_spectrum9k_v13a_smoke_seed42 --agg fedplora_v13a_os --seed 42 --split-seed 42 --run-id-prefix spectrum9k_20260723_smoke --gpu "${GPU_ID:-1}" -- --force_retrain --train_max_steps_per_client 1 --max_train_samples_per_client 10 > "$RESULT_ROOT/launcher_logs/test20260723_smoke_spectrum9k_state_seed42.launch.log" 2>&1 &
```

## S4. smoke 完成检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && find "$RESULT_ROOT" -path '*smoke*/run_logs/*.log' -type f -print -exec tail -n 8 {} \;
find "$RESULT_ROOT" -path '*smoke*/result_logs/*.json' -type f -print
find "$RESULT_ROOT" -path '*spectrum9k*smoke*client_states*client_*.pt' -type f | wc -l
```

只有三条 smoke 都无 traceback、JSON 可解析且 spectrum smoke 生成 client states 后，才进入正式任务。

---

# 第四部分：P0-A FlowerTune full-fold 1-shot（12 条 GPU probe/eval）

offset0 × seeds42/43/44 已完成并同步到本地，因此仅补 offset1–4。

## G1. offset1 / seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset1_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260723_probe1_offset1 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset1_seed42.launch.log" 2>&1 &
```

## G2. offset1 / seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset1_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260723_probe1_offset1 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset1_seed43.launch.log" 2>&1 &
```

## G3. offset1 / seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset1_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260723_probe1_offset1 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset1_seed44.launch.log" 2>&1 &
```

## G4. offset2 / seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset2_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260723_probe1_offset2 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset2_seed42.launch.log" 2>&1 &
```

## G5. offset2 / seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset2_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260723_probe1_offset2 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset2_seed43.launch.log" 2>&1 &
```

## G6. offset2 / seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset2_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260723_probe1_offset2 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset2_seed44.launch.log" 2>&1 &
```

## G7. offset3 / seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset3_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260723_probe1_offset3 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset3_seed42.launch.log" 2>&1 &
```

## G8. offset3 / seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset3_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260723_probe1_offset3 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset3_seed43.launch.log" 2>&1 &
```

## G9. offset3 / seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset3_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260723_probe1_offset3 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset3_seed44.launch.log" 2>&1 &
```

## G10. offset4 / seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset4_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260723_probe1_offset4 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset4_seed42.launch.log" 2>&1 &
```

## G11. offset4 / seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset4_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260723_probe1_offset4 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset4_seed43.launch.log" 2>&1 &
```

## G12. offset4 / seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_fullfold_probe1_offset4_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260723_probe1_offset4 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_flower_probe1_offset4_seed44.launch.log" 2>&1 &
```

## G1–G12 完成检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import json
import pathlib

root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0723")
seen = set()
route_ok = route_total = 0
deltas = []
for path in root.glob("flowertune_20260723_probe1_offset*_seed*/result_logs/*.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data["config"]
    strict = data["strict_held_out"]
    seed = int(cfg["seed"])
    offset = int(strict["selection_offset"])
    assert int(cfg["held_out_route_probe_samples"]) == 1, path
    assert len(strict["held_out_clients"]) == 4, path
    geom = data["results"]["coldstart_geom"]
    global_row = data["results"]["global"]
    matches = geom["geom_route_oracle_match_by_client"]
    route_ok += sum(bool(v) for v in matches.values())
    route_total += len(matches)
    deltas.append(100.0 * (float(geom["macro_acc"]) - float(global_row["macro_acc"])))
    seen.add((offset, seed))
expected = {(offset, seed) for offset in (1, 2, 3, 4) for seed in (42, 43, 44)}
assert seen == expected, ("missing", sorted(expected - seen), "extra", sorted(seen - expected))
print("[fullfold-1shot][ok] units=", len(seen), "routes=", f"{route_ok}/{route_total}", "positive=", sum(v > 0 for v in deltas), "/12")
PY
```

最终论文统计需把这 12 个单元与既有 offset0 三 seed 合并，形成 15 fold-seed / 60 clients。

---

# 第五部分：P0-B Qwen3B probe 配置对齐

仅补 seed42 probe=1；seed43/44 probe=1 已完成。不要重复运行 D1 3B 的 Normal/FedALT/v13a。

## G13. Qwen3B / FlowerTune / offset0 / seed42 / probe=1

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_3B" RUN_TAG_MODEL=Qwen2.5-3B BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_qwen3b_flower_probe1_seed42 --seed 42 --split-seed 42 --run-id-prefix qwen3b_20260723_flower_probe1 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom --select_candidates global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_qwen3b_flower_probe1_seed42.launch.log" 2>&1 &
```

完成后只把三 seed 的 `coldstart_geom` 合并；`coldstart` 仍作为 oracle-domain 上界单列。

---

# 第六部分：P1 canonical 9k spectrum 输入状态生成

这三条命令会重新训练 35 个 client LoRA，是真正的 GPU 训练。目的仅是生成 canonical 9k states；它们不替代现有主表结果。

## G14. D1-9k / seed42 states

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb19k_35c_dir05 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_spectrum9k_v13a_seed42 --agg fedplora_v13a_os --seed 42 --split-seed 42 --run-id-prefix spectrum9k_20260723_v13a --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_spectrum9k_state_seed42.launch.log" 2>&1 &
```

## G15. D1-9k / seed43 states

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb19k_35c_dir05 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_spectrum9k_v13a_seed43 --agg fedplora_v13a_os --seed 43 --split-seed 43 --run-id-prefix spectrum9k_20260723_v13a --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_spectrum9k_state_seed43.launch.log" 2>&1 &
```

## G16. D1-9k / seed44 states

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$D1_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=gb19k_35c_dir05 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_spectrum9k_v13a_seed44 --agg fedplora_v13a_os --seed 44 --split-seed 44 --run-id-prefix spectrum9k_20260723_v13a --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_spectrum9k_state_seed44.launch.log" 2>&1 &
```

## G14–G16 之后的 CPU-only postprocess

下面三条不使用 GPU，但依赖 G14–G16 先生成 states：

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES='' python scripts/Analysis/l1_layer_spectrum.py --state_dir "$RESULT_ROOT/spectrum9k_20260723_v13a_seed42/result_files/client_states/N7_ours_spectrum9k_v13a_seed42/seed_42" --clients_json "$D1_ROOT/seed_42/clients.json" --output_json "$RESULT_ROOT/spectrum_9k/l1_spectrum_v13a_seed42.json" --output_npz "$RESULT_ROOT/spectrum_9k/l1_spectrum_v13a_seed42.npz" --seed 42
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES='' python scripts/Analysis/l1_layer_spectrum.py --state_dir "$RESULT_ROOT/spectrum9k_20260723_v13a_seed43/result_files/client_states/N7_ours_spectrum9k_v13a_seed43/seed_43" --clients_json "$D1_ROOT/seed_43/clients.json" --output_json "$RESULT_ROOT/spectrum_9k/l1_spectrum_v13a_seed43.json" --output_npz "$RESULT_ROOT/spectrum_9k/l1_spectrum_v13a_seed43.npz" --seed 43
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES='' python scripts/Analysis/l1_layer_spectrum.py --state_dir "$RESULT_ROOT/spectrum9k_20260723_v13a_seed44/result_files/client_states/N7_ours_spectrum9k_v13a_seed44/seed_44" --clients_json "$D1_ROOT/seed_44/clients.json" --output_json "$RESULT_ROOT/spectrum_9k/l1_spectrum_v13a_seed44.json" --output_npz "$RESULT_ROOT/spectrum_9k/l1_spectrum_v13a_seed44.npz" --seed 44
```

NPZ 中包含每个 projection 的 35×35 similarity、client/domain 顺序、layer、module 和 signal ratio，可直接生成 A15，而不再用旧 19k 文件。

---

# 第七部分：补全实验闭环

本部分已经有对应代码，不再是占位项。新 flag 默认关闭，因此不改变旧 evaluator 和已有算法：

- `--held_out_route_metrics`：同一 probe state 对照 flat-B cosine、B 子空间、relative-L2、隐式 ΔW cosine、nearest-client subspace、random、oracle。
- `--onboarding_accounting`：记录 probe 本地训练 wall time、server route time、B-signature 上传字节、A/B 下载字节；字节为 tensor payload，不含序列化和网络协议开销。
- `build_common_test_benchmark.py`：保留原训练划分与 client size，只替换共享测试集，避免 eval-only 重新聚合污染。
- `checkpoint_manifest.py`：只接受 `checkpoint_ok=true && phase=final` 且唯一匹配的 bundle。
- `--export_eval_adapter_dir`：从 checkpoint 物化全局 PEFT adapter 和每个客户端实际部署 adapter。
- `run_external_lm_eval.py`：调用 lm-evaluation-harness；routed 结果对该任务域所有客户端 adapter 做宏平均，不挑最好客户端。

## 7.1 P0：补 offset0 的 router/onboarding（3 条 GPU probe/eval）

G1–G12 已加入全部 router flag 和 onboarding 埋点；以下三条补 offset0，使完整协议为 5 offsets × 3 seeds。三条同级，可跨卡并行。

### R1. FlowerTune offset0 / seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_routeraudit_probe1_offset0_seed42 --seed 42 --split-seed 42 --run-id-prefix routeraudit_20260723_offset0 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_routeraudit_offset0_seed42.launch.log" 2>&1 &
```

### R2. FlowerTune offset0 / seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_routeraudit_probe1_offset0_seed43 --seed 43 --split-seed 43 --run-id-prefix routeraudit_20260723_offset0 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_routeraudit_offset0_seed43.launch.log" 2>&1 &
```

### R3. FlowerTune offset0 / seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$FLOWER_ROOT/seed_42" EXPECTED_NUM_CLIENTS=20 RUN_TAG_DATASET=flowertune_mixed_20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_routeraudit_probe1_offset0_seed44 --seed 44 --split-seed 44 --run-id-prefix routeraudit_20260723_offset0 --gpu "${GPU_ID:-1}" -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 0 --few_shot_caps 1 --held_out_route_probe_samples 1 --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle --onboarding_accounting --schemes base,global,coldstart,coldstart_geom > "$RESULT_ROOT/launcher_logs/test20260723_routeraudit_offset0_seed44.launch.log" 2>&1 &
```

每个 JSON 必须同时出现 `strict_held_out.route_audits`、`onboarding_accounting.enabled=true` 和 8 个 `coldstart_route_*` 结果。论文中分别报告 route accuracy 与下游 accuracy；oracle 只作上界，random 只作 sanity baseline。

## 7.2 P1：修复 heterogeneity shared-test（原 C1 不可直接使用完整 IID test）

服务器实测表明，完整 IID test 与旧 dir0.5 train/val 每个 seed 重叠约 2.1k 条，因此旧 C1 的“完整 IID test + 旧 dir0.5 checkpoint”在统计上无效。当前只保留一个正文可用方案：

- **7.2-A（必须执行）**：冻结现有 IID 的完整 test，只把 IID 的 train/val/test_local 非测试池重新分配为 dir0.5，再重跑 dir0.5 Normal/v13a 共 6 条 GPU 训练。测试文件按字节复制，必须与 IID/dir0.1 完全一致。
- **7.2-B（停用）**：旧 checkpoint intersection eval-only 不再作为默认运行路径，不要运行 C0/C1/C2-C7。

### 7.2-A0. 0-GPU 重建 full-common-test dir0.5

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && set -eo pipefail
export IID_19K_ROOT="$CODE_DIR/data/domain_benchmark_35c_iid"
export DIR01_19K_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir01"
export DIR05_COMMON_ROOT="$CODE_DIR/data/domain_benchmark_35c_dir05_common_test_v2"

for SEED in 42 43 44; do
  python scripts/DataProcessScripts/repartition_with_frozen_test.py \
    --reference_split "$IID_19K_ROOT/seed_${SEED}" \
    --output_dir "$DIR05_COMMON_ROOT" \
    --num_clients_per_domain 5 \
    --seed "$SEED" \
    --partition dirichlet \
    --dirichlet_alpha 0.5 \
    --subtopic kmeans \
    --n_subtopics 10

  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$IID_19K_ROOT/seed_${SEED}/test_domain.jsonl" || { echo "[common-test][error] dir05 vs IID test mismatch seed=$SEED" >&2; exit 1; }
  cmp -s "$DIR05_COMMON_ROOT/seed_${SEED}/test_domain.jsonl" "$DIR01_19K_ROOT/seed_${SEED}/test_domain.jsonl" || { echo "[common-test][error] dir05 vs dir01 test mismatch seed=$SEED" >&2; exit 1; }
  python - "$DIR05_COMMON_ROOT/seed_${SEED}/clients.json" <<'PY'
import collections, json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = collections.Counter(str(row["domain"]) for row in rows)
assert len(rows) == 35 and len(counts) == 7 and set(counts.values()) == {5}, (len(rows), counts)
print("[common-test][ok]", sys.argv[1], dict(sorted(counts.items())))
PY
done
```

该脚本会先验证 reference 的 train/val 与 frozen test 零泄漏；若 IID reference 自身不干净会直接停止。只有三个 seed 的两次 `cmp` 和 35-client 检查全部通过，才运行下面 6 条。若第二个 `cmp` 失败，说明 IID 与 dir0.1 本来也不是同一测试集，不能继续画三档曲线。

### 7.2-A1–A6. full-common-test dir0.5 重新训练（6 条 GPU）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$DIR05_COMMON_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=dir05_common_test_v2 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N9_common_dir05_normal_seed42 --agg normal --seed 42 --split-seed 42 --run-id-prefix common_test_20260723_dir05 --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_common_dir05_normal_seed42.log" 2>&1 &
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$DIR05_COMMON_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=dir05_common_test_v2 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_common_dir05_v13a_seed42 --agg fedplora_v13a_os --seed 42 --split-seed 42 --run-id-prefix common_test_20260723_dir05 --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_common_dir05_v13a_seed42.log" 2>&1 &
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$DIR05_COMMON_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=dir05_common_test_v2 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N9_common_dir05_normal_seed43 --agg normal --seed 43 --split-seed 43 --run-id-prefix common_test_20260723_dir05 --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_common_dir05_normal_seed43.log" 2>&1 &
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$DIR05_COMMON_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=dir05_common_test_v2 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_common_dir05_v13a_seed43 --agg fedplora_v13a_os --seed 43 --split-seed 43 --run-id-prefix common_test_20260723_dir05 --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_common_dir05_v13a_seed43.log" 2>&1 &
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$DIR05_COMMON_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=dir05_common_test_v2 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N9_common_dir05_normal_seed44 --agg normal --seed 44 --split-seed 44 --run-id-prefix common_test_20260723_dir05 --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_common_dir05_normal_seed44.log" 2>&1 &
```

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && MODEL_PATH="$MODEL_135M" BENCHMARK_DIR_MAIN="$DIR05_COMMON_ROOT/seed_42" EXPECTED_NUM_CLIENTS=35 RUN_TAG_DATASET=dir05_common_test_v2 PIPELINE_ROUNDS=1 PIPELINE_EVAL_MAX_BATCHES=0 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind sft --method N7_ours_common_dir05_v13a_seed44 --agg fedplora_v13a_os --seed 44 --split-seed 44 --run-id-prefix common_test_20260723_dir05 --gpu "${GPU_ID:-1}" -- --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_common_dir05_v13a_seed44.log" 2>&1 &
```

这 6 条是 GPU 训练，不是 eval-only。完成后可把新 dir0.5 与既有 IID/dir0.1 的同方法三 seed放入 shared-test heterogeneity 表；前提是上面的 full test `cmp` 完全相同。

### 7.2-B（停用）：旧 checkpoint intersection eval-only 不再作为默认运行路径

这一路径保留为历史说明，但不要继续运行。合作者刚才遇到的 `source train/val intersects common test` 正是该类 eval-only 修补的风险信号：旧 dir0.5 checkpoint 的 train/val 与完整 IID test 有约 2.1k 条重叠，不能通过删 guard 或改阈值解决。

当前唯一正文可用路径是上面的 `7.2-A0 -> A1-A6`：冻结共同 test，重建 dir0.5 非测试池，再重训 Normal/v13a。若只是为了快速 sanity check，可以另开临时文件做 intersection 子集分析，但不得写入主表或 F5 heterogeneity 曲线。

## 7.3 P2：标准外部任务（checkpoint adapter，不重新训练）

### E-prep. checkpoint 解析 + seed42 adapter export（不训练）

旧 C2/C3 已停用，所以 external eval 不再依赖 common-test eval-only 的副产物。这里直接从正式 D1 seed42 checkpoint 导出 PEFT adapter；只做 adapter 物化，不重新训练、不重评 common-test。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && export CKPT_SEARCH_ROOTS="/data/yaominghao/gb/models/trained_models_LW /data/yaominghao/gb/result/FedPLoRA"
export CKPT_MANIFEST_EXT="$RESULT_ROOT/checkpoint_manifest_external_20260723.json"
python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS --output "$CKPT_MANIFEST_EXT"

export CKPT_NORMAL_EXT_42="$(python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS --resolve --agg_type normal --seed 42 --model_contains SmolLM2-135M --benchmark_contains "domain_benchmark_35c_dir05/seed_42")"
export CKPT_V13A_EXT_42="$(python scripts/Analysis/checkpoint_manifest.py --roots $CKPT_SEARCH_ROOTS --resolve --agg_type fedplora_v13a_os --seed 42 --model_contains SmolLM2-135M --benchmark_contains "domain_benchmark_35c_dir05/seed_42")"

CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --agg_type normal \
  --eval_only_from_checkpoint "$CKPT_NORMAL_EXT_42" \
  --metrics_output_dir "$RESULT_ROOT/external_adapter_export/normal_seed42" \
  --client_state_dir "$RESULT_ROOT/external_adapter_export/scratch_normal_seed42" \
  --export_eval_adapter_dir "$RESULT_ROOT/external_adapters/normal_seed42" \
  --export_eval_adapter_only \
  --eval_max_batches 0 \
  --batch_size 2 \
  --max_seq_length 256 \
  --torch_dtype bfloat16 \
  --eval_personalization_metrics

CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --agg_type fedplora_v13a_os \
  --eval_only_from_checkpoint "$CKPT_V13A_EXT_42" \
  --metrics_output_dir "$RESULT_ROOT/external_adapter_export/v13a_seed42" \
  --client_state_dir "$RESULT_ROOT/external_adapter_export/scratch_v13a_seed42" \
  --export_eval_adapter_dir "$RESULT_ROOT/external_adapters/v13a_seed42" \
  --export_eval_adapter_only \
  --eval_max_batches 0 \
  --batch_size 2 \
  --max_seq_length 256 \
  --torch_dtype bfloat16 \
  --eval_personalization_metrics

test -f "$RESULT_ROOT/external_adapters/normal_seed42/adapter_export_manifest.json"
test -f "$RESULT_ROOT/external_adapters/v13a_seed42/adapter_export_manifest.json"
```

### E0. 0-GPU 安装/任务注册硬检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python -m pip show lm_eval >/dev/null 2>&1 || python -m pip install 'lm_eval[hf]>=0.4.8,<0.5'
python -m lm_eval ls tasks > "$RESULT_ROOT/lm_eval_tasks_20260723.txt"
for TASK in mmlu pubmedqa mbpp; do
  grep -Eq "(^|[[:space:]])${TASK}([[:space:]]|$)" "$RESULT_ROOT/lm_eval_tasks_20260723.txt" || { echo "[external][error] task not registered: $TASK" >&2; exit 1; }
done
```

FiQA 在 lm-eval 版本间没有稳定内置 task 名，本批不伪造 task alias。若 `lm_eval ls tasks` 明确列出服务器版本的 FiQA task，再把该准确名称映射为 `:finance` 单独补跑。

### E-smoke. 10 examples（GPU eval，不训练）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/v13a_seed42/adapter_export_manifest.json" --tasks pubmedqa:medical --mode both --limit 10 --device cuda:0 --batch_size auto --output_dir "$RESULT_ROOT/external_eval_smoke/v13a_seed42" > "$RESULT_ROOT/launcher_logs/test20260723_external_smoke_v13a_seed42.log" 2>&1 &
```

### E1. Normal global / MMLU + PubMedQA + MBPP

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/normal_seed42/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode global --device cuda:0 --batch_size auto --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/normal_seed42" > "$RESULT_ROOT/launcher_logs/test20260723_external_normal_seed42.log" 2>&1 &
```

### E2. v13a global + declared-domain routed-client macro

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python scripts/Analysis/run_external_lm_eval.py --adapter_manifest "$RESULT_ROOT/external_adapters/v13a_seed42/adapter_export_manifest.json" --tasks mmlu:general,pubmedqa:medical,mbpp:code --mode both --device cuda:0 --batch_size auto --confirm_run_unsafe_code --output_dir "$RESULT_ROOT/external_eval/v13a_seed42" > "$RESULT_ROOT/launcher_logs/test20260723_external_v13a_seed42.log" 2>&1 &
```

E2 会对每个 task 顺序跑 global + 该域全部 client adapters，并在 `external_eval_summary.json` 做无权宏平均。MBPP 会执行生成代码，只能在隔离环境运行；若服务器不是隔离执行环境，去掉 MBPP 和 `--confirm_run_unsafe_code`，不要绕过安全门。

---

# 第八部分：串行与并行逻辑

```text
Stage 0（已完成，0 GPU）
  nested statistics + margin audit + fingerprint/comm + router reliability

Stage 1（串行 smoke）
  S1 -> 检查 JSON
  S2 -> 检查 3B JSON
  S3 -> 检查 client-state 文件

Stage 2（P0，优先）
  G1–G12 同级，可并行；每张 GPU 同时只跑 1 条
  同一 GPU 上必须串行，禁止 12 条全部指向 GPU 0

Stage 3（P0，可与 Stage 2 跨卡并行）
  G13 Qwen3B probe=1
  3B 显存占用高，单卡独占

Stage 4（P1，主张锁定后再跑）
  G14–G16 同级，可一 seed/一卡并行
  每个 seed 完成后立即串行运行对应 CPU spectrum postprocess

Stage 5（P0 router/onboarding 闭环）
  G1–G12 已内联新 router；R1–R3 补 offset0，可同 Stage 2 跨卡并行
  等 15 fold-seed 全齐后，统一比较 route accuracy、下游增益与 payload/time

Stage 6（P1 common-test；正文方案需重训练）
  正文方案：7.2-A0 frozen-test repartition -> test cmp -> A1–A6 GPU 重训练
  A1–A6 同级，可跨卡并行；每张卡一次一条
  旧 C0/C1/C2–C7 intersection eval-only 已停用；不要运行
  任一 test cmp 或泄漏检查失败，则对应阶段停止

Stage 7（P2 external；不训练）
  E-prep seed42 adapter export -> E0 task check -> E-smoke 串行
  smoke 通过后 E1 与 E2 可跨卡并行；E2 内部各 adapter 串行

优先停止条件
  若 full-fold 1-shot route <95%，或任一 fold-seed ΔGlobal≤0，
  不再把 1-example 放 headline；正文保持已成立的 10-example 60/60。
```

## 【注意事项】

1. 每复制一条命令前先设置 `GPU_ID`，例如 `export GPU_ID=1`；不要让多个正式任务挤在同一卡。
2. 不要给 personalized-eval 命令加 `--force_retrain`；它没有 run-checkpoint resume 语义，输出路径已按 offset/seed 隔离。
3. G14–G16 必须保留 `--force_retrain`，否则已有同 stem bundle 可能触发 resume 跳过，最终仍没有新 states。
4. FlowerTune-Mixed 是 public-source custom federation，不是官方 FlowerTune leaderboard protocol。
5. 所有论文通信量使用 effective upload+download 双向总量。
6. 不运行 v14、μ/rank 大扫参、第三训练 benchmark 或 7–8B 单点；这些不影响当前主线闭环。
7. `subspace` 是逐层 B 列空间 canonical-correlation；`delta_w_cosine` 用恒等式计算 BA 余弦而不物化稠密 ΔW。两者与 legacy `coldstart_geom` 分开命名。
8. onboarding 的 wall time 依赖硬件，只能在同机同负载条件比较；论文同时报告不依赖硬件的 tensor bytes。
9. external eval 的 PEFT adapter 必须由 `--export_eval_adapter_dir` 生成；不得直接对 base model 跑 lm-eval 后标成 FedPLoRA。
---

# 第九部分以后：额外新增实验队列

以下内容来自原 `order_20260723_sup.md`，已经合并到本文档末尾。它们不是 common-test 报错修复的前置条件；资源紧张时先跑 YOCO FlowerTune，再考虑 YOCO scale、70-client、r16。

## 【额外新增实验覆盖评估】

| require 项 | 本次判断 | 命令落地 |
|---|---|---|
| G1 YOCO on FlowerTune ×3 | 缺，最高优先 | 本文第十一部分 F1-F3 |
| G2 YOCO on 1.7B/3B ×3 | 缺，scale one-shot 对照 | 本文第十二部分 S1-S6 |
| R1 检索型 cold-start baseline | 已落地，无需新训练 | `order_gb_0723new.md` 的 `--held_out_route_metrics ... nearest_client_subspace ...` |
| W1-W10 诚实性改稿 | 写作/制图项，不是服务器实验 | 不放 GPU 命令；写稿时按清单改 |
| H1 non-IID common-test | 已修正为 frozen-test repartition + 重训 | `order_gb_0723new.md` 第 7.2-A |
| I1 matched-domain eval-only | 已单独落地，60 个正式 checkpoint | `order_eval_only_worst_indomain_20260723.md` |
| N1 70-client | 缺，P1 加分 | 本文第十三部分 N1.1-N1.9 |
| N2 rank r=16 | 缺，P1 加分 | 本文第十四部分 N2.1-N2.9 |
| E4 per-client Local | 当前训练 JSON 已有 `client_local_macro_*`，但未落盘逐 client/逐域正式表 | 暂不伪造命令；若要正文逐域 Local，需要再补代码输出 per-client artifact |
| X1 official task eval-only | 已落地 | `order_gb_0723new.md` 第 7.3 |
| L1-9k spectrum | 已落地 | `order_gb_0723new.md` 第 6 节 |

## 【额外新增实验命令设置】

```text
服务器: gb（/data/yaominghao/gb/FedPLoRA）
用户名: minghao
代码目录: /data/yaominghao/gb/FedPLoRA

结果根:
/data/yaominghao/gb/result/FedPLoRA/order_0723_sup

模型根:
/data/yaominghao/gb/models/trained_models_LW/order_0723_sup

模型:
/data/yaominghao/gb/models/SmolLM2-135M
/data/yaominghao/gb/models/SmolLM2-1.7B
/data/yaominghao/gb/models/Qwen2.5-3B

D1 (gb dir05，非 A100 9k):
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05/seed_{42,43,44}

FlowerTune-Mixed:
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_{42,43,44}

70-client frozen-test split:
/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_70c_dir05_frozen_test/seed_{42,43,44}

训练/评测:
rounds=1, local_epochs=1, lr=2e-4
主表 LoRA: r=8, alpha=16, dropout=0.05
r16: r=16, alpha=32, dropout=0.05
batch=2, max_seq_length=256, dtype=bfloat16
formal eval_max_batches=0
smoke eval_max_batches=1, max_steps=1, max_train_samples_per_client=10
```

## 【额外新增实验产物位置说明】

```text
run_logs:
/data/yaominghao/gb/result/FedPLoRA/order_0723_sup/launcher_logs/test20260723_sup_*.log

result_logs:
/data/yaominghao/gb/result/FedPLoRA/order_0723_sup/<run_id>/result_logs/<method>/

result_files/client_states:
/data/yaominghao/gb/result/FedPLoRA/order_0723_sup/<run_id>/result_files/client_states/<method>/

checkpoints:
/data/yaominghao/gb/models/trained_models_LW/order_0723_sup/<run_id>/<method>/

pids:
/data/yaominghao/gb/result/FedPLoRA/order_0723_sup/pids/*.pid
```

## 【额外新增实验运行涉及场景】

```text
P0-a/G1: FlowerTune-Mixed / 20 clients / 4 domains / YOCO one-shot external baseline
P0-a/G2: D1 canonical 9k / 35 clients / SmolLM2-1.7B and Qwen2.5-3B / YOCO scale baseline
P1/N1: D1 frozen-test repartition / 70 clients / participant-scale robustness
P1/N2: D1 canonical 9k / LoRA rank r=16 / rank-scale robustness and communication ratio
```

---

# 第九部分：额外新增实验服务器前置命令

## 0.1 登录与环境变量

```bash
# gb 登录后
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && exec bash
source /data/yaominghao/miniconda3/etc/profile.d/conda.sh
conda activate fedplora

export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723_sup
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW/order_0723_sup
export MODEL_135M=/data/yaominghao/gb/models/SmolLM2-135M
export MODEL_17B=/data/yaominghao/gb/models/SmolLM2-1.7B
export MODEL_3B=/data/yaominghao/gb/models/Qwen2.5-3B
export D1_ROOT=/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05
export FLOWER_ROOT=/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05
export D1_70C_ROOT=/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_70c_dir05_frozen_test
export GPU_ID=${GPU_ID:-1}

cd "$CODE_DIR"
mkdir -p "$RESULT_ROOT/launcher_logs" "$RESULT_ROOT/pids" "$RESULT_ROOT/comm_profile" "$MODEL_ROOT"
```

## 0.2 公共参数

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && COMMON_SFT_ARGS=(
  --rounds 1
  --local_epochs 1
  --lr 0.0002
  --lora_dropout 0.05
  --batch_size 2
  --max_seq_length 256
  --torch_dtype bfloat16
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --save_client_state_to_disk
  --gradient_checkpointing
  --eval_personalization_metrics
  --eval_final_only
  --skip_post_agg_snapshots
)

LORA_R8_ARGS=(--lora_r 8 --lora_alpha 16)
LORA_R16_ARGS=(--lora_r 16 --lora_alpha 32)

YOCO_ARGS=(
  --yoco_sparse_lambda 0.0001
  --yoco_pcwa_components 3
  --yoco_aggregate_mode conflict
  --yoco_conflict_method avgm
  --yoco_sign_lambda 0.01
)
```

## 0.3 代码、模型和数据检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/DataProcessScripts/repartition_with_frozen_test.py \
  scripts/Analysis/summarize_fedplora_results.py \
  scripts/RunScripts/print_sft_comm_profile.py

python - <<'PY'
import collections
import json
import pathlib

checks = [
    (pathlib.Path("/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05"), 35, 7, 5),
    (pathlib.Path("/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_flowertune_mixed_20c_dir05"), 20, 4, 5),
]
for root, expected_clients, expected_domains, expected_per_domain in checks:
    for seed in (42, 43, 44):
        path = root / f"seed_{seed}" / "clients.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        counts = collections.Counter(str(row["domain"]) for row in rows)
        assert len(rows) == expected_clients, (path, len(rows))
        assert len(counts) == expected_domains, (path, counts)
        assert set(counts.values()) == {expected_per_domain}, (path, counts)
        print("[preflight][ok]", path.parent, dict(sorted(counts.items())))

for model in (
    "/data/yaominghao/gb/models/SmolLM2-135M",
    "/data/yaominghao/gb/models/SmolLM2-1.7B",
    "/data/yaominghao/gb/models/Qwen2.5-3B",
):
    cfg = pathlib.Path(model) / "config.json"
    assert cfg.is_file(), cfg
    print("[preflight][ok] model", model)
PY
```

## 0.4 构建 70-client frozen-test split（0-GPU）

说明：70-client 用 canonical D1 的 full test 冻结为同一测试集，只重分配非测试池。`--min_samples_per_client 25` 是为了避免小域在 10 clients/domain 下样本下限不可达；该实验只作 participant-scale robustness，不替代主表。

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && set -eo pipefail

for SEED in 42 43 44; do
  python scripts/DataProcessScripts/repartition_with_frozen_test.py \
    --reference_split "$D1_ROOT/seed_${SEED}" \
    --output_dir "$D1_70C_ROOT" \
    --num_clients_per_domain 10 \
    --min_samples_per_client 25 \
    --seed "$SEED" \
    --partition dirichlet \
    --dirichlet_alpha 0.5 \
    --subtopic kmeans \
    --n_subtopics 10

  cmp -s "$D1_70C_ROOT/seed_${SEED}/test_domain.jsonl" "$D1_ROOT/seed_${SEED}/test_domain.jsonl" || { echo "[70c][error] test_domain mismatch seed=$SEED" >&2; exit 1; }
  python - "$D1_70C_ROOT/seed_${SEED}/clients.json" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
counts = collections.Counter(str(row["domain"]) for row in rows)
assert len(rows) == 70 and len(counts) == 7 and set(counts.values()) == {10}, (len(rows), counts)
print("[70c][ok]", path.parent, dict(sorted(counts.items())))
PY
done
```

## 0.5 r16 通信公式输出（0-GPU/CPU）

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES='' python scripts/RunScripts/print_sft_comm_profile.py \
  --model "$MODEL_135M" \
  --lora_r 16 \
  --lora_alpha 32 \
  --agg_types normal,fedalt,fedplora_v13a_os,yoco \
  --json > "$RESULT_ROOT/comm_profile/r16_comm_profile_smol135m.json"
```

---

# 第十部分：额外新增实验 GPU smoke（与正式命令分开）

所有 smoke 都是 1 step + 10 samples/client + `eval_max_batches=1`，只检查导入、训练、聚合、评测、checkpoint 写入；不能进论文表。

## SM1. YOCO FlowerTune seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$FLOWER_ROOT/seed_42" \
  --num_clients 20 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/smoke_yoco_flower_seed42/result_files/client_states/N9_flower_yoco_smoke_seed42" \
  --metrics_output_dir "$RESULT_ROOT/smoke_yoco_flower_seed42/result_logs/N9_flower_yoco_smoke_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/smoke_yoco_flower_seed42/N9_flower_yoco_smoke_seed42" \
  --trained_models_root "$MODEL_ROOT/smoke_yoco_flower_seed42" \
  --eval_max_batches 1 \
  --seed 42 \
  --train_max_steps_per_client 1 \
  --max_train_samples_per_client 10 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_smoke_yoco_flower_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/smoke_yoco_flower_seed42.pid"
```

## SM2. YOCO SmolLM2-1.7B seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_17B" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/smoke_yoco_17b_seed42/result_files/client_states/N7_baseline_yoco_17b_smoke_seed42" \
  --metrics_output_dir "$RESULT_ROOT/smoke_yoco_17b_seed42/result_logs/N7_baseline_yoco_17b_smoke_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/smoke_yoco_17b_seed42/N7_baseline_yoco_17b_smoke_seed42" \
  --trained_models_root "$MODEL_ROOT/smoke_yoco_17b_seed42" \
  --eval_max_batches 1 \
  --seed 42 \
  --train_max_steps_per_client 1 \
  --max_train_samples_per_client 10 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_smoke_yoco_17b_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/smoke_yoco_17b_seed42.pid"
```

## SM3. YOCO Qwen2.5-3B seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_3B" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/smoke_yoco_3b_seed42/result_files/client_states/N7_baseline_yoco_3b_smoke_seed42" \
  --metrics_output_dir "$RESULT_ROOT/smoke_yoco_3b_seed42/result_logs/N7_baseline_yoco_3b_smoke_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/smoke_yoco_3b_seed42/N7_baseline_yoco_3b_smoke_seed42" \
  --trained_models_root "$MODEL_ROOT/smoke_yoco_3b_seed42" \
  --eval_max_batches 1 \
  --seed 42 \
  --train_max_steps_per_client 1 \
  --max_train_samples_per_client 10 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_smoke_yoco_3b_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/smoke_yoco_3b_seed42.pid"
```

## SM4. 70-client v13a seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$D1_70C_ROOT/seed_42" \
  --num_clients 70 \
  --agg_type fedplora_v13a_os \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/smoke_70c_v13a_seed42/result_files/client_states/N1_70c_v13a_smoke_seed42" \
  --metrics_output_dir "$RESULT_ROOT/smoke_70c_v13a_seed42/result_logs/N1_70c_v13a_smoke_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/smoke_70c_v13a_seed42/N1_70c_v13a_smoke_seed42" \
  --trained_models_root "$MODEL_ROOT/smoke_70c_v13a_seed42" \
  --eval_max_batches 1 \
  --seed 42 \
  --train_max_steps_per_client 1 \
  --max_train_samples_per_client 10 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_smoke_70c_v13a_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/smoke_70c_v13a_seed42.pid"
```

## SM5. r16 v13a seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --num_clients 35 \
  --agg_type fedplora_v13a_os \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/smoke_r16_v13a_seed42/result_files/client_states/N2_r16_v13a_smoke_seed42" \
  --metrics_output_dir "$RESULT_ROOT/smoke_r16_v13a_seed42/result_logs/N2_r16_v13a_smoke_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/smoke_r16_v13a_seed42/N2_r16_v13a_smoke_seed42" \
  --trained_models_root "$MODEL_ROOT/smoke_r16_v13a_seed42" \
  --eval_max_batches 1 \
  --seed 42 \
  --train_max_steps_per_client 1 \
  --max_train_samples_per_client 10 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_smoke_r16_v13a_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/smoke_r16_v13a_seed42.pid"
```

## SM6. smoke 完成检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && find "$RESULT_ROOT/launcher_logs" -type f -name 'test20260723_sup_smoke_*.log' -print -exec tail -n 12 {} \;
find "$RESULT_ROOT" -path '*smoke*/result_logs/*.json' -type f -print
```

只有 5 条 smoke 均无 traceback、均能写出 final metrics JSON 后，才运行正式命令。

---

# 第十一部分：P0-a/G1 YOCO on FlowerTune-Mixed（3 条 GPU）

## F1. FlowerTune YOCO seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$FLOWER_ROOT/seed_42" \
  --num_clients 20 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_flower_seed42/result_files/client_states/N9_flower_yoco_seed42" \
  --metrics_output_dir "$RESULT_ROOT/yoco_flower_seed42/result_logs/N9_flower_yoco_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_flower_seed42/N9_flower_yoco_seed42" \
  --trained_models_root "$MODEL_ROOT/yoco_flower_seed42" \
  --eval_max_batches 0 \
  --seed 42 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_flower_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_flower_seed42.pid"
```

## F2. FlowerTune YOCO seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$FLOWER_ROOT/seed_43" \
  --num_clients 20 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_flower_seed43/result_files/client_states/N9_flower_yoco_seed43" \
  --metrics_output_dir "$RESULT_ROOT/yoco_flower_seed43/result_logs/N9_flower_yoco_seed43" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_flower_seed43/N9_flower_yoco_seed43" \
  --trained_models_root "$MODEL_ROOT/yoco_flower_seed43" \
  --eval_max_batches 0 \
  --seed 43 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_flower_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_flower_seed43.pid"
```

## F3. FlowerTune YOCO seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_135M" \
  --benchmark_dir "$FLOWER_ROOT/seed_44" \
  --num_clients 20 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_flower_seed44/result_files/client_states/N9_flower_yoco_seed44" \
  --metrics_output_dir "$RESULT_ROOT/yoco_flower_seed44/result_logs/N9_flower_yoco_seed44" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_flower_seed44/N9_flower_yoco_seed44" \
  --trained_models_root "$MODEL_ROOT/yoco_flower_seed44" \
  --eval_max_batches 0 \
  --seed 44 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_flower_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_flower_seed44.pid"
```

## F-check

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python scripts/Analysis/summarize_fedplora_results.py \
  "$RESULT_ROOT/yoco_flower_seed42/result_logs" \
  "$RESULT_ROOT/yoco_flower_seed43/result_logs" \
  "$RESULT_ROOT/yoco_flower_seed44/result_logs" \
  > "$RESULT_ROOT/yoco_flower_summary_20260723.md"
```

---

# 第十二部分：P0-a/G2 YOCO on scale axis（6 条 GPU）

说明：scale axis 采用 D1 canonical 9k，补齐 1.7B/3B 的外部 one-shot baseline。3B 任务显存高，建议独占 GPU。

## S1. SmolLM2-1.7B YOCO seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_17B" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_17b_seed42/result_files/client_states/N7_baseline_yoco_17b_seed42" \
  --metrics_output_dir "$RESULT_ROOT/yoco_17b_seed42/result_logs/N7_baseline_yoco_17b_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_17b_seed42/N7_baseline_yoco_17b_seed42" \
  --trained_models_root "$MODEL_ROOT/yoco_17b_seed42" \
  --eval_max_batches 0 \
  --seed 42 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_17b_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_17b_seed42.pid"
```

## S2. SmolLM2-1.7B YOCO seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_17B" \
  --benchmark_dir "$D1_ROOT/seed_43" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_17b_seed43/result_files/client_states/N7_baseline_yoco_17b_seed43" \
  --metrics_output_dir "$RESULT_ROOT/yoco_17b_seed43/result_logs/N7_baseline_yoco_17b_seed43" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_17b_seed43/N7_baseline_yoco_17b_seed43" \
  --trained_models_root "$MODEL_ROOT/yoco_17b_seed43" \
  --eval_max_batches 0 \
  --seed 43 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_17b_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_17b_seed43.pid"
```

## S3. SmolLM2-1.7B YOCO seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_17B" \
  --benchmark_dir "$D1_ROOT/seed_44" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_17b_seed44/result_files/client_states/N7_baseline_yoco_17b_seed44" \
  --metrics_output_dir "$RESULT_ROOT/yoco_17b_seed44/result_logs/N7_baseline_yoco_17b_seed44" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_17b_seed44/N7_baseline_yoco_17b_seed44" \
  --trained_models_root "$MODEL_ROOT/yoco_17b_seed44" \
  --eval_max_batches 0 \
  --seed 44 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_17b_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_17b_seed44.pid"
```

## S4. Qwen2.5-3B YOCO seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_3B" \
  --benchmark_dir "$D1_ROOT/seed_42" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_3b_seed42/result_files/client_states/N7_baseline_yoco_3b_seed42" \
  --metrics_output_dir "$RESULT_ROOT/yoco_3b_seed42/result_logs/N7_baseline_yoco_3b_seed42" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_3b_seed42/N7_baseline_yoco_3b_seed42" \
  --trained_models_root "$MODEL_ROOT/yoco_3b_seed42" \
  --eval_max_batches 0 \
  --seed 42 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_3b_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_3b_seed42.pid"
```

## S5. Qwen2.5-3B YOCO seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_3B" \
  --benchmark_dir "$D1_ROOT/seed_43" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_3b_seed43/result_files/client_states/N7_baseline_yoco_3b_seed43" \
  --metrics_output_dir "$RESULT_ROOT/yoco_3b_seed43/result_logs/N7_baseline_yoco_3b_seed43" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_3b_seed43/N7_baseline_yoco_3b_seed43" \
  --trained_models_root "$MODEL_ROOT/yoco_3b_seed43" \
  --eval_max_batches 0 \
  --seed 43 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_3b_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_3b_seed43.pid"
```

## S6. Qwen2.5-3B YOCO seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py \
  --model "$MODEL_3B" \
  --benchmark_dir "$D1_ROOT/seed_44" \
  --num_clients 35 \
  --agg_type yoco \
  "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" "${YOCO_ARGS[@]}" \
  --client_state_dir "$RESULT_ROOT/yoco_3b_seed44/result_files/client_states/N7_baseline_yoco_3b_seed44" \
  --metrics_output_dir "$RESULT_ROOT/yoco_3b_seed44/result_logs/N7_baseline_yoco_3b_seed44" \
  --save_run_checkpoint_dir "$MODEL_ROOT/yoco_3b_seed44/N7_baseline_yoco_3b_seed44" \
  --trained_models_root "$MODEL_ROOT/yoco_3b_seed44" \
  --eval_max_batches 0 \
  --seed 44 \
  --force_retrain \
  > "$RESULT_ROOT/launcher_logs/test20260723_sup_yoco_3b_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/yoco_3b_seed44.pid"
```

---

# 第十三部分：P1/N1 70-client participant-scale（9 条 GPU）

## N1.1 70c Normal seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_42" --num_clients 70 --agg_type normal "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_normal_seed42/result_files/client_states/N1_70c_normal_seed42" --metrics_output_dir "$RESULT_ROOT/70c_normal_seed42/result_logs/N1_70c_normal_seed42" --save_run_checkpoint_dir "$MODEL_ROOT/70c_normal_seed42/N1_70c_normal_seed42" --trained_models_root "$MODEL_ROOT/70c_normal_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_normal_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_normal_seed42.pid"
```

## N1.2 70c FedALT seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_42" --num_clients 70 --agg_type fedalt "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_fedalt_seed42/result_files/client_states/N1_70c_fedalt_seed42" --metrics_output_dir "$RESULT_ROOT/70c_fedalt_seed42/result_logs/N1_70c_fedalt_seed42" --save_run_checkpoint_dir "$MODEL_ROOT/70c_fedalt_seed42/N1_70c_fedalt_seed42" --trained_models_root "$MODEL_ROOT/70c_fedalt_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_fedalt_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_fedalt_seed42.pid"
```

## N1.3 70c v13a seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_42" --num_clients 70 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_v13a_seed42/result_files/client_states/N1_70c_v13a_seed42" --metrics_output_dir "$RESULT_ROOT/70c_v13a_seed42/result_logs/N1_70c_v13a_seed42" --save_run_checkpoint_dir "$MODEL_ROOT/70c_v13a_seed42/N1_70c_v13a_seed42" --trained_models_root "$MODEL_ROOT/70c_v13a_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_v13a_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_v13a_seed42.pid"
```

## N1.4 70c Normal seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_43" --num_clients 70 --agg_type normal "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_normal_seed43/result_files/client_states/N1_70c_normal_seed43" --metrics_output_dir "$RESULT_ROOT/70c_normal_seed43/result_logs/N1_70c_normal_seed43" --save_run_checkpoint_dir "$MODEL_ROOT/70c_normal_seed43/N1_70c_normal_seed43" --trained_models_root "$MODEL_ROOT/70c_normal_seed43" --eval_max_batches 0 --seed 43 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_normal_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_normal_seed43.pid"
```

## N1.5 70c FedALT seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_43" --num_clients 70 --agg_type fedalt "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_fedalt_seed43/result_files/client_states/N1_70c_fedalt_seed43" --metrics_output_dir "$RESULT_ROOT/70c_fedalt_seed43/result_logs/N1_70c_fedalt_seed43" --save_run_checkpoint_dir "$MODEL_ROOT/70c_fedalt_seed43/N1_70c_fedalt_seed43" --trained_models_root "$MODEL_ROOT/70c_fedalt_seed43" --eval_max_batches 0 --seed 43 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_fedalt_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_fedalt_seed43.pid"
```

## N1.6 70c v13a seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_43" --num_clients 70 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_v13a_seed43/result_files/client_states/N1_70c_v13a_seed43" --metrics_output_dir "$RESULT_ROOT/70c_v13a_seed43/result_logs/N1_70c_v13a_seed43" --save_run_checkpoint_dir "$MODEL_ROOT/70c_v13a_seed43/N1_70c_v13a_seed43" --trained_models_root "$MODEL_ROOT/70c_v13a_seed43" --eval_max_batches 0 --seed 43 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_v13a_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_v13a_seed43.pid"
```

## N1.7 70c Normal seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_44" --num_clients 70 --agg_type normal "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_normal_seed44/result_files/client_states/N1_70c_normal_seed44" --metrics_output_dir "$RESULT_ROOT/70c_normal_seed44/result_logs/N1_70c_normal_seed44" --save_run_checkpoint_dir "$MODEL_ROOT/70c_normal_seed44/N1_70c_normal_seed44" --trained_models_root "$MODEL_ROOT/70c_normal_seed44" --eval_max_batches 0 --seed 44 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_normal_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_normal_seed44.pid"
```

## N1.8 70c FedALT seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_44" --num_clients 70 --agg_type fedalt "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_fedalt_seed44/result_files/client_states/N1_70c_fedalt_seed44" --metrics_output_dir "$RESULT_ROOT/70c_fedalt_seed44/result_logs/N1_70c_fedalt_seed44" --save_run_checkpoint_dir "$MODEL_ROOT/70c_fedalt_seed44/N1_70c_fedalt_seed44" --trained_models_root "$MODEL_ROOT/70c_fedalt_seed44" --eval_max_batches 0 --seed 44 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_fedalt_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_fedalt_seed44.pid"
```

## N1.9 70c v13a seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_70C_ROOT/seed_44" --num_clients 70 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R8_ARGS[@]}" --client_state_dir "$RESULT_ROOT/70c_v13a_seed44/result_files/client_states/N1_70c_v13a_seed44" --metrics_output_dir "$RESULT_ROOT/70c_v13a_seed44/result_logs/N1_70c_v13a_seed44" --save_run_checkpoint_dir "$MODEL_ROOT/70c_v13a_seed44/N1_70c_v13a_seed44" --trained_models_root "$MODEL_ROOT/70c_v13a_seed44" --eval_max_batches 0 --seed 44 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_70c_v13a_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/70c_v13a_seed44.pid"
```

---

# 第十四部分：P1/N2 rank r=16 单点（9 条 GPU）

说明：r16 采用 `lora_r=16,lora_alpha=32`，保持 alpha/r=2，与主表 r8 alpha16 的 LoRA scale 对齐。

## N2.1 r16 Normal seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_42" --num_clients 35 --agg_type normal "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_normal_seed42/result_files/client_states/N2_r16_normal_seed42" --metrics_output_dir "$RESULT_ROOT/r16_normal_seed42/result_logs/N2_r16_normal_seed42" --save_run_checkpoint_dir "$MODEL_ROOT/r16_normal_seed42/N2_r16_normal_seed42" --trained_models_root "$MODEL_ROOT/r16_normal_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_normal_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_normal_seed42.pid"
```

## N2.2 r16 FedALT seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_42" --num_clients 35 --agg_type fedalt "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_fedalt_seed42/result_files/client_states/N2_r16_fedalt_seed42" --metrics_output_dir "$RESULT_ROOT/r16_fedalt_seed42/result_logs/N2_r16_fedalt_seed42" --save_run_checkpoint_dir "$MODEL_ROOT/r16_fedalt_seed42/N2_r16_fedalt_seed42" --trained_models_root "$MODEL_ROOT/r16_fedalt_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_fedalt_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_fedalt_seed42.pid"
```

## N2.3 r16 v13a seed42

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_42" --num_clients 35 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_v13a_seed42/result_files/client_states/N2_r16_v13a_seed42" --metrics_output_dir "$RESULT_ROOT/r16_v13a_seed42/result_logs/N2_r16_v13a_seed42" --save_run_checkpoint_dir "$MODEL_ROOT/r16_v13a_seed42/N2_r16_v13a_seed42" --trained_models_root "$MODEL_ROOT/r16_v13a_seed42" --eval_max_batches 0 --seed 42 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_v13a_seed42.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_v13a_seed42.pid"
```

## N2.4 r16 Normal seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_43" --num_clients 35 --agg_type normal "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_normal_seed43/result_files/client_states/N2_r16_normal_seed43" --metrics_output_dir "$RESULT_ROOT/r16_normal_seed43/result_logs/N2_r16_normal_seed43" --save_run_checkpoint_dir "$MODEL_ROOT/r16_normal_seed43/N2_r16_normal_seed43" --trained_models_root "$MODEL_ROOT/r16_normal_seed43" --eval_max_batches 0 --seed 43 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_normal_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_normal_seed43.pid"
```

## N2.5 r16 FedALT seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_43" --num_clients 35 --agg_type fedalt "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_fedalt_seed43/result_files/client_states/N2_r16_fedalt_seed43" --metrics_output_dir "$RESULT_ROOT/r16_fedalt_seed43/result_logs/N2_r16_fedalt_seed43" --save_run_checkpoint_dir "$MODEL_ROOT/r16_fedalt_seed43/N2_r16_fedalt_seed43" --trained_models_root "$MODEL_ROOT/r16_fedalt_seed43" --eval_max_batches 0 --seed 43 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_fedalt_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_fedalt_seed43.pid"
```

## N2.6 r16 v13a seed43

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_43" --num_clients 35 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_v13a_seed43/result_files/client_states/N2_r16_v13a_seed43" --metrics_output_dir "$RESULT_ROOT/r16_v13a_seed43/result_logs/N2_r16_v13a_seed43" --save_run_checkpoint_dir "$MODEL_ROOT/r16_v13a_seed43/N2_r16_v13a_seed43" --trained_models_root "$MODEL_ROOT/r16_v13a_seed43" --eval_max_batches 0 --seed 43 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_v13a_seed43.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_v13a_seed43.pid"
```

## N2.7 r16 Normal seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_44" --num_clients 35 --agg_type normal "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_normal_seed44/result_files/client_states/N2_r16_normal_seed44" --metrics_output_dir "$RESULT_ROOT/r16_normal_seed44/result_logs/N2_r16_normal_seed44" --save_run_checkpoint_dir "$MODEL_ROOT/r16_normal_seed44/N2_r16_normal_seed44" --trained_models_root "$MODEL_ROOT/r16_normal_seed44" --eval_max_batches 0 --seed 44 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_normal_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_normal_seed44.pid"
```

## N2.8 r16 FedALT seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_44" --num_clients 35 --agg_type fedalt "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_fedalt_seed44/result_files/client_states/N2_r16_fedalt_seed44" --metrics_output_dir "$RESULT_ROOT/r16_fedalt_seed44/result_logs/N2_r16_fedalt_seed44" --save_run_checkpoint_dir "$MODEL_ROOT/r16_fedalt_seed44/N2_r16_fedalt_seed44" --trained_models_root "$MODEL_ROOT/r16_fedalt_seed44" --eval_max_batches 0 --seed 44 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_fedalt_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_fedalt_seed44.pid"
```

## N2.9 r16 v13a seed44

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" nohup /usr/bin/time -v python -u tasks/fed_train_sft.py --model "$MODEL_135M" --benchmark_dir "$D1_ROOT/seed_44" --num_clients 35 --agg_type fedplora_v13a_os "${COMMON_SFT_ARGS[@]}" "${LORA_R16_ARGS[@]}" --client_state_dir "$RESULT_ROOT/r16_v13a_seed44/result_files/client_states/N2_r16_v13a_seed44" --metrics_output_dir "$RESULT_ROOT/r16_v13a_seed44/result_logs/N2_r16_v13a_seed44" --save_run_checkpoint_dir "$MODEL_ROOT/r16_v13a_seed44/N2_r16_v13a_seed44" --trained_models_root "$MODEL_ROOT/r16_v13a_seed44" --eval_max_batches 0 --seed 44 --force_retrain > "$RESULT_ROOT/launcher_logs/test20260723_sup_r16_v13a_seed44.log" 2>&1 &
echo $! > "$RESULT_ROOT/pids/r16_v13a_seed44.pid"
```

---

# 第十五部分：已落地 require 项的验收入口

## 7.1 R1 retrieval cold-start baseline

在 `order_gb_0723new.md` 中，G1-G12 与 R1-R3 已包含：

```text
--held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,largest_domain,random,oracle
```

验收命令：

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import json
import pathlib

root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0723")
paths = sorted(root.glob("**/result_logs/X2_flower_*probe1*seed*.json"))
if not paths:
    raise SystemExit("[R1][error] no router audit JSON found")
ok = 0
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    audits = (data.get("strict_held_out") or {}).get("route_audits") or {}
    if "nearest_client_subspace" in audits:
        ok += 1
print(f"[R1][ok] nearest_client_subspace present in {ok}/{len(paths)} files")
if ok == 0:
    raise SystemExit("[R1][error] retrieval baseline not found")
PY
```

## 7.2 H1 non-IID common-test

执行 `order_gb_0723new.md` 第 7.2-A：`repartition_with_frozen_test.py` 构建 `domain_benchmark_35c_dir05_common_test_v2`，再跑 Normal/v13a × 3 seeds。旧完整 IID-test eval-only 方案已因 2.1k train/test overlap 判定无效，不再使用。

## 7.3 I1 matched-domain eval-only

完整 60-job 命令已放在：

```text
/Users/hawaiii/codex/FedPLoRA/order/order_eval_only_worst_indomain_20260723.md
```

该任务必须在原始训练节点 `/data/yaominghao/gb/FedPLoRA` 上运行，因为正式 checkpoint 根在 `/data/yaominghao/gb/models/trained_models_LW`。验收标准是：

```text
D1: 39 个 *_matched_domain.json
FlowerTune: 21 个 *_matched_domain.json
summary: d1_summary.tsv 和 flowertune_summary.tsv
```

## 7.4 X1 external task eval-only

执行 `order_gb_0723new.md` 第 7.3。MBPP 会执行生成代码；若服务器不是隔离环境，去掉 MBPP 和 `--confirm_run_unsafe_code`。

## 7.5 L1-9k spectrum

执行 `order_gb_0723new.md` 第 6 节 G14-G16 与 CPU postprocess。它依赖 canonical 9k 的 v13a client states，不使用本文件的 70c/r16 states。

---

# 第十六部分：额外新增实验正式结果汇总

## 8.1 汇总本文额外新增部分所有正式训练

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python scripts/Analysis/summarize_fedplora_results.py \
  "$RESULT_ROOT/yoco_flower_seed42/result_logs" \
  "$RESULT_ROOT/yoco_flower_seed43/result_logs" \
  "$RESULT_ROOT/yoco_flower_seed44/result_logs" \
  "$RESULT_ROOT/yoco_17b_seed42/result_logs" \
  "$RESULT_ROOT/yoco_17b_seed43/result_logs" \
  "$RESULT_ROOT/yoco_17b_seed44/result_logs" \
  "$RESULT_ROOT/yoco_3b_seed42/result_logs" \
  "$RESULT_ROOT/yoco_3b_seed43/result_logs" \
  "$RESULT_ROOT/yoco_3b_seed44/result_logs" \
  "$RESULT_ROOT/70c_normal_seed42/result_logs" \
  "$RESULT_ROOT/70c_fedalt_seed42/result_logs" \
  "$RESULT_ROOT/70c_v13a_seed42/result_logs" \
  "$RESULT_ROOT/70c_normal_seed43/result_logs" \
  "$RESULT_ROOT/70c_fedalt_seed43/result_logs" \
  "$RESULT_ROOT/70c_v13a_seed43/result_logs" \
  "$RESULT_ROOT/70c_normal_seed44/result_logs" \
  "$RESULT_ROOT/70c_fedalt_seed44/result_logs" \
  "$RESULT_ROOT/70c_v13a_seed44/result_logs" \
  "$RESULT_ROOT/r16_normal_seed42/result_logs" \
  "$RESULT_ROOT/r16_fedalt_seed42/result_logs" \
  "$RESULT_ROOT/r16_v13a_seed42/result_logs" \
  "$RESULT_ROOT/r16_normal_seed43/result_logs" \
  "$RESULT_ROOT/r16_fedalt_seed43/result_logs" \
  "$RESULT_ROOT/r16_v13a_seed43/result_logs" \
  "$RESULT_ROOT/r16_normal_seed44/result_logs" \
  "$RESULT_ROOT/r16_fedalt_seed44/result_logs" \
  "$RESULT_ROOT/r16_v13a_seed44/result_logs" \
  > "$RESULT_ROOT/order_20260723_sup_summary.md"
```

## 8.2 完整性检查

```bash
cd /data/yaominghao/gb/FedPLoRA && export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}" && python - <<'PY'
import pathlib

root = pathlib.Path("/data/yaominghao/gb/result/FedPLoRA/order_0723_sup")
expected = [
    "yoco_flower_seed42", "yoco_flower_seed43", "yoco_flower_seed44",
    "yoco_17b_seed42", "yoco_17b_seed43", "yoco_17b_seed44",
    "yoco_3b_seed42", "yoco_3b_seed43", "yoco_3b_seed44",
]
expected += [f"70c_{m}_seed{s}" for s in (42, 43, 44) for m in ("normal", "fedalt", "v13a")]
expected += [f"r16_{m}_seed{s}" for s in (42, 43, 44) for m in ("normal", "fedalt", "v13a")]
missing = []
for name in expected:
    if not list((root / name / "result_logs").rglob("*.json")):
        missing.append(name)
if missing:
    raise SystemExit("[sup][missing] " + ", ".join(missing))
print(f"[sup][ok] completed formal result groups={len(expected)}")
PY
```

---

# 第十七部分：全量串行与并行逻辑（含额外新增实验）

```text
Stage 0（0-GPU）
  0.1-0.3 preflight -> 0.4 构建 70c frozen-test -> 0.5 r16 comm profile

Stage 1（串行 smoke）
  SM1 -> SM2 -> SM3 -> SM4 -> SM5
  任一 smoke traceback、OOM 或无 JSON，则停止，不跑正式命令。

Stage 2（P0，最高信息增益）
  F1-F3 YOCO FlowerTune，可一 seed/一卡并行。
  这三条优先于 N1/N2/G2，因为它直接补主叙事的外部 one-shot baseline。

Stage 3（P0/P1，可与 eval-only 跨机并行）
  在原训练节点运行 I1 matched-domain eval-only。
  在 gb（/data/yaominghao/gb/FedPLoRA） 按 order_20260723 跑 H1 common-test 与 R1 router/onboarding 验收。

Stage 4（P1）
  N1.1-N1.9 70c 与 N2.1-N2.9 r16。
  两组都是同级附录加分；资源紧时先 N1 再 N2。

Stage 5（P0-a scale 轴补强）
  S1-S3 1.7B YOCO 可并行。
  S4-S6 3B YOCO 单卡独占，建议夜间跑。

并行原则
  每张 GPU 同时只跑一条正式训练。
  复制命令前先 export GPU_ID=<空闲卡号>。
  3B 不与其它 3B 或长 eval 挤同一卡。
```

## 【注意事项】

1. 本文件的 baseline 命令全部直接调用 `tasks/fed_train_sft.py`，不走 `run_20260713_one_experiment.sh`，避免 main/baseline preflight guard 混用时误拦截 YOCO/Normal/FedALT。
2. 所有正式训练都带 `--force_retrain`，防止旧 checkpoint bundle 触发 resume 跳过。
3. 70c 使用 frozen D1 test，可与 D1 35c 同 test 比较；但 `min_samples_per_client=25` 与主表 35c 不同，论文中应写成 participant-scale appendix。
4. r16 采用 alpha32 保持 alpha/r=2；若审稿补问“固定 alpha 的纯 rank 敏感性”，那是另一组实验，不与本组混表。
5. E4 per-client/per-domain Local 当前不是命令问题，而是 artifact 粒度问题；不要把现有 `client_local_macro_*` 强行写成逐域 Local 表。

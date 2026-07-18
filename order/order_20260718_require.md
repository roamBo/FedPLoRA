# FedCCE 20260718 投稿前补齐实验命令

######### FedCCE 顶会投稿前缺口补齐实验命令-20260718 #########

【命令介绍】

本文件根据 `/Users/hawaiii/codex/FedCCE/claude/claude_Experiment_20260718.md` 和
`/Users/hawaiii/codex/FedCCE/claude/claude_require_20260718.md` 重整。核心原则是：

1. 已完成的 M1/M3 主结果、预算曲线、审计定位、历史消融只标注和检查，不重复训练。
2. 真正待运行的 GPU 实验按 P0/P1 排序：N1-N6 为投稿前最高优先级；P1 为附录增强。
3. 0-GPU 的 Z1-Z6 属于离线装配/统计，不占 GPU，但必须与 GPU 实验并行完成。
4. 所有正式训练命令都带已有结果跳过逻辑；同一块 GPU 每次只启动一条正式 `nohup`。

【评估结论】

`claude_Experiment_20260718.md` 的实验矩阵整体完整，已经覆盖顶会审稿最常问的三类问题：

- 主方法是否显著强于强 baseline：Table 1 四数据集三 seed 共轨主表已经成立，但 CIFAR-10 对 naive2l_random 的 seed-level 稳定性还需要 N1 seed3/4 修补。
- 模块是否必要：已有 add-one-in 与 edge-pathway 消融，但缺少 final profile 的 LOO 视角；N5 需要补 `-cov` 和 `-edge-anchor`，`late-half -> uniform` 可离线重聚合，`no-HT` 当前没有可靠命令行开关，需后续代码开关或 posthoc 脚本后再写入论文。
- 泛化/鲁棒/规模是否充分：M1/M3 已有；还需要 N2 的 N=50 分层 baseline 对照、N3 污染鲁棒性、N4 下游有效性、N6 edge 数量敏感性。

`claude_require_20260718.md` 基本可以视为当前全部待补清单，但要区分“待训练”和“待离线装配”：

- 待训练：N1、N2、N3、N4、N5 的可执行部分、N6、P1-1、P1-2、P1-3。
- 待离线：Z1-Z6、N5 的 late-half/uniform posthoc、主表统计包、成本表、motivation 图。
- 暂不能直接声称已闭环：N4 的 DataSize/naive2l 加权聚合、N5 的 no-HT、P1-3 的精确 6/3/2/1 edge size 指定。当前代码可跑的是 proxy，不应在论文中写成精确实现。

【命令目的】

在不重复已有 M1/M3 的前提下，补齐投稿前最容易被审稿人追问的实验缺口：

- N1：补 CIFAR-10 seed3/4，解决 C10 单数据集 seed-level 显著性软点。
- N2：补 N=50 的分层 baseline 对照，避免规模端只有 FedCCE 一行。
- N3：补污染/坏参与者场景下的贡献保真与 bottom-k 命中。
- N4：补 contribution-aware aggregation 的下游有用性证据。
- N5：补 final profile 组件 LOO 的可执行部分。
- N6：补 edge 数量敏感性，形成 E=2/3/4/6/8/10 曲线。
- P1：补 alpha、部分参与、拓扑/edge group imbalance proxy 附录增强。

【命令设置】

```text
代码目录: /data2/minghao/code/FedRepo-FedCCE
主算法: fedcce_v8403, final profile
主数据集: CIFAR-10
主设置: N=12, Dirichlet alpha=0.1, local_epoch=1, max_round=40, eval rounds=10/20/30/40
主层级: E=4, label-kmeans edge assignment
主参照: same-run hierarchical full-Owen reference for N=12; MC full-Owen reference for N>12
主表 baseline: 12 companion methods
资源策略: P0 优先，P1 可中止；每次只跑一条正式 nohup
```

【实验产物位置说明】

```text
run_logs:
/data2/minghao/result/FedCCE/final_20260718_require/run_logs/

result_logs:
/data2/minghao/result/FedCCE/final_20260718_require/result_logs/

result_files:
/data2/minghao/result/FedCCE/final_20260718_require/result_files/

cache:
/data2/minghao/result/FedCCE/final_20260718_require/result_files/cache/

analysis:
/data2/minghao/result/FedCCE/final_20260718_require/analysis/
```

【实验运行涉及场景】

```text
Clean Non-IID contribution fidelity
CIFAR-10 seed-level significance
Large-N hierarchical baseline comparison
Hidden label-noise / bad-client robustness
Contribution-aware downstream aggregation
Component leave-one-out ablation
Edge-count sensitivity
Dirichlet alpha sensitivity
Partial participation / full-audit HT correction
Topology / edge group imbalance proxy
```

【实验前置命令】

```bash
conda activate FedRepo
export FEDCCE_REPO="${FEDCCE_REPO:-/data2/minghao/code/FedRepo-FedCCE}"
export FEDCCE_RESULT_ROOT="${FEDCCE_RESULT_ROOT:-/data2/minghao/result/FedCCE}"
export FEDCCE_OUT="${FEDCCE_OUT:-$FEDCCE_RESULT_ROOT/final_20260718_require}"
export FEDCCE_M1_DONE="${FEDCCE_M1_DONE:-$FEDCCE_RESULT_ROOT/A100_Result/results/final_20260716_m}"
export FEDCCE_B1_DONE="${FEDCCE_B1_DONE:-$FEDCCE_RESULT_ROOT/A100_Result/results/final_20260716}"
export FEDCCE_ALL_LOCAL="${FEDCCE_ALL_LOCAL:-$FEDCCE_RESULT_ROOT}"
export GPU="${GPU:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export METHODS12="datasize,dfce,space,shapfed,feddsv,tmcshapley,fedowen,gtgshapley,hcdv_native,hcdv_edge,naive2l_kmeans,naive2l_random"
export METHODS_REP="datasize,dfce,space,shapfed,feddsv,hcdv_native,hcdv_edge,naive2l_kmeans,naive2l_random"
export METHODS_EDGE="datasize,hcdv_native,hcdv_edge,naive2l_kmeans,naive2l_random"
export METHODS_MIN="datasize"

cd "$FEDCCE_REPO"
mkdir -p "$FEDCCE_OUT/run_logs" "$FEDCCE_OUT/result_logs" "$FEDCCE_OUT/result_files" "$FEDCCE_OUT/result_files/cache" "$FEDCCE_OUT/analysis"

python -m py_compile \
  algorithms/fedcce_v8401.py algorithms/fedcce_v8402.py algorithms/fedcce_v8403.py \
  train_dir_v3.py \
  experiments/scripts/run_fedcce_v7_case.py \
  experiments/scripts/check_fedcce_v837_shared.py \
  experiments/scripts/analyze_fedcce_edge_pathways_20260716.py \
  experiments/scripts/analyze_fedcce_rank_metrics_20260715.py

python experiments/scripts/run_fedcce_v7_case.py --help | grep -E "fedcce_v8403|cce-v839|cce-v5-attack|cce-v5-partition-balance|cce-aware-agg"
find "$FEDCCE_RESULT_ROOT" -type f -name '*.npy' | sort > "$FEDCCE_OUT/analysis/existing_npy_before_order_20260718_require.txt"
```

## 0. 已完成结果标注，不重复训练

### 0.1 M1 四数据集三 seed 共轨主表，已完成

```bash
python - <<'PY'
import os
import glob
import numpy as np

root = os.path.join(os.environ["FEDCCE_M1_DONE"], "result_files")
methods = set(os.environ["METHODS12"].split(","))
missing = []
for dataset in ["cifar10", "cifar100", "famnist", "tinyimagenet"]:
    for seed in [0, 1, 2]:
        tag = f"{dataset}_v8403_clean12_n12_a0p1_e1_s{seed}"
        master = os.path.join(root, tag + ".npy")
        bdir = os.path.join(root, tag + "_baselines")
        files = glob.glob(os.path.join(bdir, "*.npy"))
        got = {os.path.basename(p).split("__")[-1].replace(".npy", "") for p in files}
        if not os.path.exists(master) or got != methods:
            missing.append((dataset, seed, os.path.exists(master), sorted(got)))
        else:
            payload = np.load(master, allow_pickle=True).item()
            assert (payload.get("meta") or {}).get("algo") == "fedcce_v8403"
            assert len(payload.get("round_records", []) or []) == 4
print("M1_DONE_CHECK_OK" if not missing else "M1_MISSING")
if missing:
    raise SystemExit(missing)
PY
```

### 0.2 M3 N=50/100 规模实验，已完成

```bash
python - <<'PY'
import os
import numpy as np

root = os.path.join(os.environ["FEDCCE_M1_DONE"], "result_files")
expected = {
    "m3_cifar10_v8403_n50_e8_a0p1_s0_mc1024.npy": (50, 8),
    "m3_cifar10_v8403_n100_e10_a0p1_s0_mc1024.npy": (100, 10),
}
for name, (n_clients, n_edges) in expected.items():
    path = os.path.join(root, name)
    payload = np.load(path, allow_pickle=True).item()
    record = (payload.get("round_records") or [])[0]
    assert (payload.get("meta") or {}).get("algo") == "fedcce_v8403"
    assert len(record.get("active_clients", [])) == n_clients
    assert len(record.get("active_edges", [])) == n_edges
    assert record.get("v8403_reference_mode") == "mc_full_owen"
    print(name, "OK")
PY
```

### 0.3 B1 edge-pathway 消融，已完成

```bash
find "$FEDCCE_B1_DONE/result_files" -maxdepth 1 -type f -name 'p0_edgepath_*.npy' | sort
python experiments/scripts/analyze_fedcce_edge_pathways_20260716.py \
  --glob "$FEDCCE_B1_DONE/result_files/p0_edgepath_*.npy" \
  --glob "$FEDCCE_B1_DONE/result_files/p0_edgepath_*_baselines/*.npy" \
  --out-dir "$FEDCCE_OUT/analysis/b1_edge_pathways_existing" \
  --prefix b1_edge_pathways_existing \
  --topk 1,2,3,4,5,6,10,20 \
  --quadrant-threshold 0.8
```

## 1. Smoke 测试

### 1.1 v8403 + attack + all baselines smoke

说明：该 smoke 同时覆盖 final profile、污染参数、12 companion baseline、同轨输出目录和 content-hash 轨迹。

```bash
OUT="$FEDCCE_OUT/result_files/smoke_v8403_attack_n4_e2_s0.npy"
BDIR="$FEDCCE_OUT/result_files/smoke_v8403_attack_n4_e2_s0_baselines"
if [ -s "$OUT" ] && [ "$(find "$BDIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')" -eq 12 ]; then
  echo "[SKIP complete] smoke_v8403_attack_n4_e2_s0"
else
  nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
    --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
    --dir-alpha 0.1 --n-clients 4 --c-ratio 1.0 --local-epochs 1 --local-steps 1 --max-round 1 --test-round 1 --seed 0 --cuda true \
    --cce-n-edges 2 --cce-round 1 --cce-start-round 1 --cce-proto-batches 1 --cce-val-batches 1 \
    --cce-v5-attack-mode client_label_flip --cce-v5-auto-bad-clients 1 --cce-v6-auto-bad-client-policy hidden --cce-v5-noise-ratio 0.25 --cce-v5-attack-seed 20260718 \
    --cce-v80-edge-anchor-exact-max 2 --cce-v834-exact-max-edges 2 --cce-v834-max-group-clients 2 --cce-v834-full-owen-max-clients 4 \
    --cce-v837-model-test-batches 1 --cce-v837-summary-batches 1 --cce-v837-tmc-samples 4 --cce-v837-fedowen-samples 8 --cce-v837-space-samples 4 --cce-v837-gtg-group-size 2 \
    --cce-v837-baseline-seed 0 --cce-v837-fail-fast true \
    --cce-v839-baseline-methods "$METHODS12" --cce-v839-edge-samples 4 --cce-v839-client-samples 4 \
    --cce-v839-hcdv-edge-samples 4 --cce-v839-hcdv-client-budget 12 --cce-v839-hcdv-min-client-samples 2 \
    --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/smoke_v8403_attack_n4_e2_s0" \
    --cce-v837-baseline-output-dir "$BDIR" \
    --cce-v837-progress-path "$FEDCCE_OUT/result_files/smoke_v8403_attack_n4_e2_s0_progress.npy" \
    --fname "$FEDCCE_OUT/result_logs/smoke_v8403_attack_n4_e2_s0.log" \
    --cce-npy-path "$OUT" \
    > "$FEDCCE_OUT/run_logs/smoke_v8403_attack_n4_e2_s0.log" 2>&1 &
fi
```

```bash
python experiments/scripts/check_fedcce_v837_shared.py \
  --master "$FEDCCE_OUT/result_files/smoke_v8403_attack_n4_e2_s0.npy" \
  --baseline-dir "$FEDCCE_OUT/result_files/smoke_v8403_attack_n4_e2_s0_baselines" \
  --expected-methods "$METHODS12" \
  --expected-algo fedcce_v8403 \
  --expected-rounds 1 \
  --expected-clients 4 \
  --expected-edges 2 \
  --require-nonconstant
```

## 2. 公共运行函数

### 2.1 N=12 final profile with configurable methods

```bash
run_v8403_n12_case() {
  TAG="$1"; SEED="$2"; METHODS="$3"; EXPECTED="$4"; EXTRA_ARGS="$5"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  BDIR="$FEDCCE_OUT/result_files/${TAG}_baselines"
  if [ -s "$OUT" ] && [ "$(find "$BDIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')" -eq "$EXPECTED" ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
      --dir-alpha 0.1 --n-clients 12 --c-ratio 1.0 --local-epochs 1 --max-round 40 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges 4 --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-v80-edge-anchor-exact-max 4 --cce-v834-exact-max-edges 4 --cce-v834-max-group-clients 3 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-tmc-samples 100 --cce-v837-fedowen-samples 200 --cce-v837-space-samples 100 --cce-v837-gtg-group-size 3 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971800 + SEED))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS" --cce-v839-edge-samples 100 --cce-v839-client-samples 100 \
      --cce-v839-hcdv-edge-samples 64 --cce-v839-hcdv-client-budget 200 --cce-v839-hcdv-min-client-samples 8 \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$BDIR" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      $EXTRA_ARGS \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

### 2.2 检查 N=12 all-baseline

```bash
check_v8403_n12_all() {
  TAG="$1"
  python experiments/scripts/check_fedcce_v837_shared.py \
    --master "$FEDCCE_OUT/result_files/${TAG}.npy" \
    --baseline-dir "$FEDCCE_OUT/result_files/${TAG}_baselines" \
    --expected-methods "$METHODS12" \
    --expected-algo fedcce_v8403 \
    --expected-rounds 10,20,30,40 \
    --expected-clients 12 \
    --expected-edges 4 \
    --require-nonconstant
}
```

## 3. P0-1 / N1 CIFAR-10 seed3/4 显著性修补

目的：补齐 CIFAR-10 5-seed 主表口径。为避免表格口径不一致，直接跑全 12 baseline。

```bash
run_v8403_n12_case "n1_cifar10_v8403_clean12_n12_a0p1_e1_s3" 3 "$METHODS12" 12 ""
```

```bash
run_v8403_n12_case "n1_cifar10_v8403_clean12_n12_a0p1_e1_s4" 4 "$METHODS12" 12 ""
```

```bash
check_v8403_n12_all "n1_cifar10_v8403_clean12_n12_a0p1_e1_s3"
check_v8403_n12_all "n1_cifar10_v8403_clean12_n12_a0p1_e1_s4"
```

## 4. P0-2 / N2 N=50 分层 baseline 对照

目的：M3 已证明 N=50 的 FedCCE scale fidelity，但 sparse scale mode 会跳过 companion baseline。这里单独跑 N=50、E=8、CIFAR-10、seed0、`c_ratio=1.0`，让所有 audit clients 都有 local model，从而产出 N=50 的 naive2l/HCDV/DFCE/SPACE/ShapFed/FedDSV 对照。

注意：这是中等成本任务；不跑 TMC/FedOwen/GTG，避免把 N=50 对照做成新主表级算力负担。

```bash
TAG="n2_cifar10_v8403_n50_e8_a0p1_s0_scale_baselines"
OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
BDIR="$FEDCCE_OUT/result_files/${TAG}_baselines"
if [ -s "$OUT" ] && [ "$(find "$BDIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')" -eq 9 ]; then
  echo "[SKIP complete] $TAG"
else
  nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
    --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
    --dir-alpha 0.1 --n-clients 50 --c-ratio 1.0 --local-epochs 1 --max-round 40 --test-round 5 --seed 0 --cuda true \
    --cce-n-edges 8 --cce-round 40 --cce-start-round 40 --cce-proto-batches 3 --cce-val-batches 5 \
    --cce-v80-edge-anchor-exact-max 8 --cce-v834-exact-max-edges 8 --cce-v834-max-group-clients 16 --cce-v834-full-owen-max-clients 12 \
    --cce-v835-joint-samples 64 --cce-v8363-reference-samples 1024 --cce-v8363-reference-repeats 1 --cce-v8363-reference-seed 971820 --cce-v8363-reference-max-clients 64 \
    --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed 971820 --cce-v837-fail-fast true \
    --cce-v839-baseline-methods "$METHODS_REP" --cce-v839-edge-samples 64 --cce-v839-client-samples 64 \
    --cce-v839-hcdv-edge-samples 32 --cce-v839-hcdv-client-budget 160 --cce-v839-hcdv-min-client-samples 8 \
    --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
    --cce-v837-baseline-output-dir "$BDIR" \
    --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
    --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
    --cce-npy-path "$OUT" \
    > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
fi
```

```bash
python experiments/scripts/check_fedcce_v837_shared.py \
  --master "$FEDCCE_OUT/result_files/n2_cifar10_v8403_n50_e8_a0p1_s0_scale_baselines.npy" \
  --baseline-dir "$FEDCCE_OUT/result_files/n2_cifar10_v8403_n50_e8_a0p1_s0_scale_baselines_baselines" \
  --expected-methods "$METHODS_REP" \
  --expected-algo fedcce_v8403 \
  --expected-rounds 40 \
  --expected-clients 50 \
  --expected-edges 8 \
  --require-nonconstant
```

## 5. P0-3 / N3 污染下贡献保真

目的：CIFAR-10 N=12 hidden label-noise，验证坏客户端在 same-run reference 下确实落入低贡献区，同时比较 12 baseline 的保真退化。

```bash
run_v8403_n12_case "n3_cifar10_v8403_hidden_labelnoise_n12_a0p1_e1_s0" 0 "$METHODS12" 12 "--cce-v5-attack-mode client_label_flip --cce-v5-auto-bad-clients 3 --cce-v6-auto-bad-client-policy hidden --cce-v5-noise-ratio 0.25 --cce-v5-attack-seed 971830"
```

```bash
run_v8403_n12_case "n3_cifar10_v8403_hidden_labelnoise_n12_a0p1_e1_s1" 1 "$METHODS12" 12 "--cce-v5-attack-mode client_label_flip --cce-v5-auto-bad-clients 3 --cce-v6-auto-bad-client-policy hidden --cce-v5-noise-ratio 0.25 --cce-v5-attack-seed 971831"
```

```bash
run_v8403_n12_case "n3_cifar10_v8403_hidden_labelnoise_n12_a0p1_e1_s2" 2 "$METHODS12" 12 "--cce-v5-attack-mode client_label_flip --cce-v5-auto-bad-clients 3 --cce-v6-auto-bad-client-policy hidden --cce-v5-noise-ratio 0.25 --cce-v5-attack-seed 971832"
```

```bash
for SEED in 0 1 2; do
  check_v8403_n12_all "n3_cifar10_v8403_hidden_labelnoise_n12_a0p1_e1_s${SEED}"
done
```

```bash
python experiments/scripts/analyze_fedcce_rank_metrics_20260715.py \
  --glob "$FEDCCE_OUT/result_files/n3_cifar10_v8403_hidden_labelnoise_n12_a0p1_e1_s[0-2].npy" \
  --glob "$FEDCCE_OUT/result_files/n3_cifar10_v8403_hidden_labelnoise_n12_a0p1_e1_s[0-2]_baselines/*.npy" \
  --out-dir "$FEDCCE_OUT/analysis/n3_contamination_rank_metrics" \
  --prefix n3_contamination_rank_metrics \
  --topk 1,2,3,5,10,20
```

## 6. P0-4 / N4 下游 contribution-aware aggregation

目的：补 correlation 之外的 so-what 证据。当前代码直接支持 `FedCCE-aware` 与 uniform aggregation 对照；DataSize/naive2l weighted replay 尚无可靠命令，不在本轮冒充已完成。

### 6.1 定义 N4 运行函数

```bash
run_n4_downstream() {
  MODE="$1"; SEED="$2"
  if [ "$MODE" = "aware" ]; then
    AWARE=true
  else
    AWARE=false
  fi
  TAG="n4_cifar10_v8403_downstream_${MODE}_n12_a0p1_e1_s${SEED}"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  if [ -s "$OUT" ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
      --dir-alpha 0.1 --n-clients 12 --c-ratio 1.0 --local-epochs 1 --max-round 80 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges 4 --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-aware-agg "$AWARE" --cce-agg-ema-decay 0.9 \
      --cce-v80-edge-anchor-exact-max 4 --cce-v834-exact-max-edges 4 --cce-v834-max-group-clients 3 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971840 + SEED))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS_MIN" \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$FEDCCE_OUT/result_files/${TAG}_baselines" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

### 6.2 N4 seed0 闸门

```bash
run_n4_downstream uniform 0
```

```bash
run_n4_downstream aware 0
```

### 6.3 N4 seed1/2 正式补齐

```bash
run_n4_downstream uniform 1
```

```bash
run_n4_downstream aware 1
```

```bash
run_n4_downstream uniform 2
```

```bash
run_n4_downstream aware 2
```

```bash
python experiments/scripts/analyze_fedcce_v835_final.py \
  --glob "$FEDCCE_OUT/result_files/n4_cifar10_v8403_downstream_*_n12_a0p1_e1_s[0-2].npy" \
  --out-dir "$FEDCCE_OUT/analysis/n4_downstream_cceaware" \
  --prefix n4_downstream_cceaware
```

## 7. P0-5 / N5 组件 LOO 消融

目的：补 final profile 组件必要性证据。当前可直接跑：

- `-covariance`：`--cce-use-cov false`
- `-edge-anchor/output`：使用 v8401 raw-from-client branch 作为不输出 exact edge-Owen anchor 的对照；B1 已有部分 seed，函数中会优先查找历史等价结果
- `late-half -> uniform`：0-GPU 从 round_records 离线重聚合

当前不可直接跑：

- `no-HT`：没有稳定命令行开关；需要新增 posthoc 脚本或新版本算法后再补，不应在当前论文表里写成已完成。

### 7.1 N5 no-cov 三 seed

```bash
run_v8403_n12_case "n5_cifar10_v8403_nocov_n12_a0p1_e1_s0" 0 "$METHODS_MIN" 1 "--cce-use-cov false"
```

```bash
run_v8403_n12_case "n5_cifar10_v8403_nocov_n12_a0p1_e1_s1" 1 "$METHODS_MIN" 1 "--cce-use-cov false"
```

```bash
run_v8403_n12_case "n5_cifar10_v8403_nocov_n12_a0p1_e1_s2" 2 "$METHODS_MIN" 1 "--cce-use-cov false"
```

### 7.2 N5 no exact-edge-output branch，补齐 v8401 三 seed

```bash
run_n5_noedge_seed() {
  SEED="$1"
  TAG="n5_cifar10_v8401_rawfromclient_no_exact_edge_n12_a0p1_e1_s${SEED}"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  EXISTING="$(find "$FEDCCE_RESULT_ROOT" -type f -size +0c -name "p0_edgepath_cifar10_fedcce_v8401_n12_a0p1_e1_s${SEED}.npy" -print -quit)"
  if [ -n "$EXISTING" ]; then
    echo "[SKIP equivalent existing] $EXISTING"
  elif [ -s "$OUT" ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8401 --dataset cifar10 --net ResNet20 \
      --dir-alpha 0.1 --n-clients 12 --c-ratio 1.0 --local-epochs 1 --max-round 40 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges 4 --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-v80-edge-anchor-exact-max 4 --cce-v834-exact-max-edges 4 --cce-v834-max-group-clients 3 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971850 + SEED))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS_MIN" \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$FEDCCE_OUT/result_files/${TAG}_baselines" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

```bash
run_n5_noedge_seed 0
```

```bash
run_n5_noedge_seed 1
```

```bash
run_n5_noedge_seed 2
```

### 7.3 N5 late-half -> uniform 离线重聚合

```bash
python - <<'PY'
import csv
import glob
import math
import os
import numpy as np

def float_dict(value):
    return {int(k): float(v) for k, v in (value or {}).items()}

def normalize(scores):
    keys = list(scores)
    vals = np.asarray([scores[k] for k in keys], dtype=float)
    if vals.size == 0:
        return {}
    vals = np.maximum(vals, 0.0)
    total = float(vals.sum())
    if total <= 1e-12:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: float(v / total) for k, v in zip(keys, vals)}

def rankdata(vals):
    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=float)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        ranks[order[i:j+1]] = rank
        i = j + 1
    return ranks

def spearman(a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return ""
    x = rankdata(np.asarray([a[k] for k in keys], dtype=float))
    y = rankdata(np.asarray([b[k] for k in keys], dtype=float))
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return ""
    return float(np.corrcoef(x, y)[0, 1])

def aggregate(records, fields, mode):
    if mode == "late_half":
        start = len(records) // 2
        picked = records[start:]
    else:
        picked = records
    acc = {}
    cnt = {}
    for record in picked:
        score = {}
        for field in fields:
            score = float_dict(record.get(field))
            if score:
                break
        for k, v in score.items():
            acc[k] = acc.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1
    return {k: acc[k] / cnt[k] for k in acc}

root = os.path.join(os.environ["FEDCCE_M1_DONE"], "result_files")
rows = []
for path in sorted(glob.glob(os.path.join(root, "cifar10_v8403_clean12_n12_a0p1_e1_s*.npy"))):
    payload = np.load(path, allow_pickle=True).item()
    records = payload.get("round_records", []) or []
    seed = (payload.get("meta") or {}).get("seed")
    client_ref = float_dict(payload.get("reference_score"))
    edge_ref = float_dict(payload.get("edge_reference_score"))
    for mode in ["late_half", "uniform"]:
        client = normalize(aggregate(records, ["v7_client_score_all_imputed", "client_score", "client_corrected_score"], mode))
        edge = normalize(aggregate(records, ["v80_edge_direct_anchor", "v840_selected_edge_score", "edge_score"], mode))
        rows.append({
            "seed": seed,
            "mode": mode,
            "client_spearman": spearman(client, client_ref),
            "edge_spearman": spearman(edge, edge_ref),
        })
out = os.path.join(os.environ["FEDCCE_OUT"], "analysis", "n5_latehalf_vs_uniform_posthoc.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", "mode", "client_spearman", "edge_spearman"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

## 8. P0-6 / N6 edge 数量敏感性 E-sweep

目的：形成 C10 上 `E in {2,3,4,6,8,10}` 的 edge-S 曲线。E=4 复用 M1，E=8/10 复用 M3；这里只补 E=2/3/6 × seed0/1/2。

```bash
run_n6_edge_sweep() {
  E="$1"; SEED="$2"
  TAG="n6_cifar10_v8403_edges${E}_n12_a0p1_e1_s${SEED}"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  BDIR="$FEDCCE_OUT/result_files/${TAG}_baselines"
  if [ -s "$OUT" ] && [ "$(find "$BDIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')" -eq 5 ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
      --dir-alpha 0.1 --n-clients 12 --c-ratio 1.0 --local-epochs 1 --max-round 40 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges "$E" --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-v80-edge-anchor-exact-max "$E" --cce-v834-exact-max-edges "$E" --cce-v834-max-group-clients 12 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971860 + 10 * SEED + E))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS_EDGE" --cce-v839-edge-samples 100 --cce-v839-client-samples 100 \
      --cce-v839-hcdv-edge-samples 64 --cce-v839-hcdv-client-budget 200 --cce-v839-hcdv-min-client-samples 8 \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$BDIR" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

```bash
run_n6_edge_sweep 2 0
```

```bash
run_n6_edge_sweep 2 1
```

```bash
run_n6_edge_sweep 2 2
```

```bash
run_n6_edge_sweep 3 0
```

```bash
run_n6_edge_sweep 3 1
```

```bash
run_n6_edge_sweep 3 2
```

```bash
run_n6_edge_sweep 6 0
```

```bash
run_n6_edge_sweep 6 1
```

```bash
run_n6_edge_sweep 6 2
```

```bash
python experiments/scripts/analyze_fedcce_edge_pathways_20260716.py \
  --glob "$FEDCCE_M1_DONE/result_files/cifar10_v8403_clean12_n12_a0p1_e1_s[0-2].npy" \
  --glob "$FEDCCE_M1_DONE/result_files/m3_cifar10_v8403_n*_e*_a0p1_s0_mc1024.npy" \
  --glob "$FEDCCE_OUT/result_files/n6_cifar10_v8403_edges*_n12_a0p1_e1_s*.npy" \
  --glob "$FEDCCE_OUT/result_files/n6_cifar10_v8403_edges*_n12_a0p1_e1_s*_baselines/*.npy" \
  --out-dir "$FEDCCE_OUT/analysis/n6_edge_count_sweep" \
  --prefix n6_edge_count_sweep \
  --topk 1,2,3,4,5,6,10,20 \
  --quadrant-threshold 0.8
```

### 8.1 N6 stretch：fixed-N=12 的 E=8/10 纯 edge-count 对照

说明：Claude 矩阵将 M3 的 E=8/10 接入 E-sweep，但 M3 同时改变了 N=50/100。若正文要写“纯 edge 数量敏感性”，建议在资源允许时补以下 6 条 fixed-N=12 命令；资源不足时可不跑，但图注必须写为 `E/scale mixed evidence`。

```bash
run_n6_edge_sweep 8 0
```

```bash
run_n6_edge_sweep 8 1
```

```bash
run_n6_edge_sweep 8 2
```

```bash
run_n6_edge_sweep 10 0
```

```bash
run_n6_edge_sweep 10 1
```

```bash
run_n6_edge_sweep 10 2
```

## 9. P1-1 alpha sweep

目的：附录验证 Non-IID 强度变化。alpha=0.1 复用 M1；这里只补 alpha=0.5/1.0 × 3 seed。

```bash
run_p1_alpha() {
  ALPHA="$1"; SEED="$2"; ATAG="${ALPHA/./p}"
  TAG="p1_alpha_cifar10_v8403_alpha${ATAG}_n12_e4_s${SEED}"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  BDIR="$FEDCCE_OUT/result_files/${TAG}_baselines"
  if [ -s "$OUT" ] && [ "$(find "$BDIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')" -eq 5 ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
      --dir-alpha "$ALPHA" --n-clients 12 --c-ratio 1.0 --local-epochs 1 --max-round 40 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges 4 --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-v80-edge-anchor-exact-max 4 --cce-v834-exact-max-edges 4 --cce-v834-max-group-clients 3 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971900 + SEED))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS_EDGE" --cce-v839-edge-samples 100 --cce-v839-client-samples 100 \
      --cce-v839-hcdv-edge-samples 64 --cce-v839-hcdv-client-budget 200 --cce-v839-hcdv-min-client-samples 8 \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$BDIR" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

```bash
run_p1_alpha 0.5 0
```

```bash
run_p1_alpha 0.5 1
```

```bash
run_p1_alpha 0.5 2
```

```bash
run_p1_alpha 1.0 0
```

```bash
run_p1_alpha 1.0 1
```

```bash
run_p1_alpha 1.0 2
```

## 10. P1-2 部分参与 / full-audit HT 验证

目的：验证 `c_ratio=0.5` 下 full audit 与 HT correction 口径是否稳定。v8403 sparse scale mode 下 companion baseline 会被跳过，这是设计行为；该组只比较 FedCCE 与 same-run/MC reference。

```bash
run_p1_partial() {
  SEED="$1"
  TAG="p1_partial_cifar10_v8403_c0p5_fullaudit_n12_e4_s${SEED}"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  if [ -s "$OUT" ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
      --dir-alpha 0.1 --n-clients 12 --c-ratio 0.5 --local-epochs 1 --max-round 40 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges 4 --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-v71-audit-mode full --cce-v71-audit-max-clients 12 \
      --cce-v80-edge-anchor-exact-max 4 --cce-v834-exact-max-edges 4 --cce-v834-max-group-clients 3 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971920 + SEED))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS_MIN" \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$FEDCCE_OUT/result_files/${TAG}_baselines" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

```bash
run_p1_partial 0
```

```bash
run_p1_partial 1
```

```bash
run_p1_partial 2
```

## 11. P1-3 拓扑 / edge group imbalance proxy

目的：补拓扑与 edge group imbalance 讨论。当前代码没有显式指定 6/3/2/1 edge size 的接口；这里使用 `--cce-v5-partition-balance false` 作为不均衡 proxy，并在论文中明确为 unbalanced k-means hierarchy proxy。

```bash
run_p1_unbalanced_edge_proxy() {
  SEED="$1"
  TAG="p1_unbalanced_proxy_cifar10_v8403_kmeans_unbalanced_n12_e4_s${SEED}"
  OUT="$FEDCCE_OUT/result_files/${TAG}.npy"
  BDIR="$FEDCCE_OUT/result_files/${TAG}_baselines"
  if [ -s "$OUT" ] && [ "$(find "$BDIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')" -eq 5 ]; then
    echo "[SKIP complete] $TAG"
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/scripts/run_fedcce_v7_case.py \
      --algo fedcce_v8403 --dataset cifar10 --net ResNet20 \
      --dir-alpha 0.1 --n-clients 12 --c-ratio 1.0 --local-epochs 1 --max-round 40 --test-round 5 --seed "$SEED" --cuda true \
      --cce-n-edges 4 --cce-edge-assign kmeans --cce-v5-partition-balance false \
      --cce-round 10 --cce-start-round 10 --cce-proto-batches 3 --cce-val-batches 5 \
      --cce-v80-edge-anchor-exact-max 4 --cce-v834-exact-max-edges 4 --cce-v834-max-group-clients 12 --cce-v834-full-owen-max-clients 12 \
      --cce-v837-model-test-batches 2 --cce-v837-summary-batches 3 --cce-v837-baseline-seed "$((971930 + SEED))" --cce-v837-fail-fast true \
      --cce-v839-baseline-methods "$METHODS_EDGE" --cce-v839-edge-samples 100 --cce-v839-client-samples 100 \
      --cce-v839-hcdv-edge-samples 64 --cce-v839-hcdv-client-budget 200 --cce-v839-hcdv-min-client-samples 8 \
      --cce-v839-cache-dir "$FEDCCE_OUT/result_files/cache/${TAG}" \
      --cce-v837-baseline-output-dir "$BDIR" \
      --cce-v837-progress-path "$FEDCCE_OUT/result_files/${TAG}_progress.npy" \
      --fname "$FEDCCE_OUT/result_logs/${TAG}.log" \
      --cce-npy-path "$OUT" \
      > "$FEDCCE_OUT/run_logs/${TAG}.log" 2>&1 &
  fi
}
```

```bash
run_p1_unbalanced_edge_proxy 0
```

```bash
run_p1_unbalanced_edge_proxy 1
```

```bash
run_p1_unbalanced_edge_proxy 2
```

## 12. 0-GPU 必做包 Z1-Z6

### 12.1 Z1 N=50/100 分辨率匹配与相邻间隔 CDF

```bash
python - <<'PY'
import csv
import glob
import os
import numpy as np

def dict_values(payload, key):
    value = payload.get(key)
    if isinstance(value, dict):
        return np.asarray([float(value[k]) for k in sorted(value, key=lambda x: int(x))], dtype=float)
    return np.asarray(value or [], dtype=float)

rows = []
root = os.path.join(os.environ["FEDCCE_M1_DONE"], "result_files")
for path in sorted(glob.glob(os.path.join(root, "m3_cifar10_v8403_n*_e*_a0p1_s0_mc1024.npy"))):
    payload = np.load(path, allow_pickle=True).item()
    meta = payload.get("meta", {}) or {}
    for key, level in [("reference_score", "client"), ("edge_reference_score", "edge")]:
        values = np.sort(dict_values(payload, key).reshape(-1))
        gaps = np.diff(values)
        for i, gap in enumerate(gaps):
            rows.append({
                "file": os.path.basename(path),
                "n_clients": meta.get("n_clients"),
                "level": level,
                "gap": float(gap),
                "cdf": float((i + 1) / max(1, len(gaps))),
            })
out = os.path.join(os.environ["FEDCCE_OUT"], "analysis", "z1_m3_resolution_gap_cdf.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "n_clients", "level", "gap", "cdf"])
    writer.writeheader()
    writer.writerows(rows)
print(out, len(rows))
PY
```

### 12.2 Z2 主表 Friedman/CD 输入表与 win-count

```bash
python experiments/scripts/analyze_fedcce_edge_pathways_20260716.py \
  --glob "$FEDCCE_M1_DONE/result_files/*_v8403_clean12_n12_a0p1_e1_s*.npy" \
  --glob "$FEDCCE_M1_DONE/result_files/*_v8403_clean12_n12_a0p1_e1_s*_baselines/*.npy" \
  --out-dir "$FEDCCE_OUT/analysis/z2_m1_stats_input" \
  --prefix z2_m1_stats_input \
  --topk 1,2,3,4,5,6,10,20 \
  --quadrant-threshold 0.8
```

### 12.3 Z3 成本三维表

```bash
python - <<'PY'
import csv
import glob
import os
import numpy as np

rows = []
for path in sorted(glob.glob(os.path.join(os.environ["FEDCCE_RESULT_ROOT"], "**", "*.npy"), recursive=True)):
    try:
        payload = np.load(path, allow_pickle=True).item()
    except Exception:
        continue
    meta = payload.get("meta", {}) or payload.get("metadata", {}) or {}
    runtime = payload.get("runtime", {}) or {}
    calls = payload.get("utility_calls", {}) or {}
    rows.append({
        "file": path,
        "algo": meta.get("algo", payload.get("algo", "")),
        "dataset": meta.get("dataset", payload.get("dataset", "")),
        "seed": meta.get("seed", payload.get("seed", "")),
        "n_clients": meta.get("n_clients", ""),
        "baseline_method": meta.get("baseline_method", payload.get("baseline_method", "")),
        "method_utility_calls": calls.get("method_total", payload.get("method_total_utility_calls", "")),
        "reference_utility_calls": calls.get("reference_total", payload.get("reference_total_utility_calls", "")),
        "total_utility_calls": calls.get("total", payload.get("total_utility_calls", "")),
        "contribution_eval_seconds": runtime.get("contribution_eval_seconds", payload.get("contribution_eval_seconds", "")),
        "total_wall_seconds": runtime.get("total_wall_seconds", payload.get("total_wall_seconds", "")),
        "communication_note": "derived_or_na",
        "memory_note": "derived_or_na",
    })
out = os.path.join(os.environ["FEDCCE_OUT"], "analysis", "z3_cost_inventory_20260718.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
    writer.writeheader()
    writer.writerows(rows)
print(out, len(rows))
PY
```

### 12.4 Z4/Z5/Z6 图表和工程核查

```bash
python - <<'PY'
import glob
import os
import numpy as np

root = os.path.join(os.environ["FEDCCE_M1_DONE"], "result_files")
issues = []
for path in sorted(glob.glob(os.path.join(root, "*_v8403_clean12_n12_a0p1_e1_s*.npy"))):
    payload = np.load(path, allow_pickle=True).item()
    meta = payload.get("meta", {}) or {}
    tid = str((payload.get("v837_shared_trajectory_profile") or {}).get("trajectory_id", ""))
    if "-h" not in tid:
        issues.append((path, "missing_content_hash", tid))
    if meta.get("algo") != "fedcce_v8403":
        issues.append((path, "wrong_algo", meta.get("algo")))
    if not payload.get("edge_reference_score"):
        issues.append((path, "missing_edge_reference", ""))
print("Z6_ENGINEERING_GUARD_OK" if not issues else "Z6_ENGINEERING_GUARD_ISSUES")
if issues:
    raise SystemExit(issues[:10])
PY
```

## 13. 最终执行顺序

```text
第一批 P0:
1. smoke_v8403_attack_n4_e2_s0
2. N1 seed3/4
3. N2 N=50 scale baselines
4. N3 seed0 -> 分析坏者是否进 bottom-k -> seed1/2

第二批 P0:
5. N4 uniform/aware seed0 -> 检查训练曲线 -> seed1/2
6. N5 no-cov + no-edge branch + late-half/uniform posthoc
7. N6 E=2/3/6 sweep

并行 0-GPU:
8. Z1/Z2/Z3/Z4/Z5/Z6

P1 附录增强:
9. alpha sweep
10. partial participation
11. unbalanced edge proxy
```

## 14. 完整性自审

1. 主表 Table 1：M1 已完成，N1 补 C10 5-seed；不重复跑四数据集主表。
2. 规模 Table 2：M3 已完成，N2 补 N=50 分层 baseline；N=100 不补 baseline，按 scale probe 表述。
3. 正文补充：N3/N4/N5/N6 分别对应污染、下游、组件、E-sweep，覆盖 `claude_Experiment_20260718.md` 的 B3-B6 关键缺口。
4. 附录：P1-1/P1-2/P1-3 覆盖 alpha、partial、topology/imbalance proxy；P2 不进本轮 order。
5. 0-GPU：Z1-Z6 均有命令或检查入口；Friedman/CD 图的最终绘图可在生成 `z2_m1_stats_input_per_run.csv` 后用论文绘图脚本完成。
6. 未全面闭环但已标注：N4 的 DataSize/naive2l weighted replay、N5 no-HT、精确 6/3/2/1 edge size。当前 order 不把这些写成已完成实验。

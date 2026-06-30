# FedPLoRA 小模型运行步骤与命令

本文档用于在较小开源大模型上快速验证当前 FedPLoRA 系列方法，默认配置为 `SmolLM2-135M + LW7c`。命令均按 `nohup python -u ... > log 2>&1 &` 风格给出，便于在服务器后台运行。

当前代码状态：

- `v5` 已实现：入口为 `tasks/fed_train_sft_v4.py`，可运行 `v5_route_mix_align` 与 `v5_rpca_route_mix_align`。
- `v5-merge` 已实现：入口为 `tasks/fed_train_sft.py`，可运行 `v5m_mean`、`v5m_ties`、`v5m_dare_ties`、`v5m_knots_ties`。
- `v6 / DCR` 已实现：入口为 `tasks/fed_train_sft.py`，可运行 `v6_dcr_global` 与 `v6_dcr_domain`。
- `v5.5` 当前只是设计方案：仓库内没有 `v5pm_*` 聚合器和可运行入口，本文只保留待实现后的命令模板，当前不要直接执行。

## 1. 推荐运行顺序

1. 准备模型和 LW7c 数据。
2. 先跑 sanity baseline：`flexlora`、`flora`、`yoco`、`fedplora-oneshot`、`v5m_mean`。
3. 优先跑 `v6_dcr_global` 与 `v6_dcr_domain`，验证 A-only Grassmann 子空间共识是否超过 v2/v5。
4. 跑 `v5` route 系列：`local_route`、`local_route_align`、`domain_anchor_align`、`rpca_local_route_align`。
5. 跑 `v5-merge` 系列：`v5m_ties`、`v5m_dare_ties`、`v5m_knots_ties`，再看是否追加 `energy rank`。
6. 汇总 CSV，筛掉明显不行的方法。
7. 若小模型上 `v6` 或 `v5-merge` 有稳定增益，再把同样命令迁移到更大模型和更完整 eval。

两块 A100 的调度建议：

- GPU0：优先跑 `v6_dcr_global/domain` 和 baseline sanity。
- GPU1：跑 `v5-merge` 或 `v5` route 系列，其中 `v5m_knots_ties` 和 `v5_route_post_align_steps > 0` 更耗时。
- 每张 GPU 同时建议只跑一个训练进程；不要一次性复制本文所有 `nohup` 命令。

## 2. 环境变量与目录

先进入仓库根目录。下面命令需要逐行执行或整段粘贴执行，不要把 `cd "$CODE_ROOT"` 和后面的 `mkdir -p ...` 手动合并到同一行，否则会触发 `bash: cd: too many arguments`。

```bash
export CODE_ROOT=/home/minghao/code/FedPLoRA-main
export MODEL_ROOT=/data2/minghao/model
export DATA_ROOT="$CODE_ROOT/data"

cd "$CODE_ROOT" || exit 1
mkdir -p log_LWv5 log_LWv6 artifacts_LW7c/sft_metrics artifacts_LW7c/sft_metrics_v5merge artifacts_LW7c/sft_metrics_v6 artifacts_LW7c/v4_sft_metrics_v5 artifacts_LW7c/summary "$MODEL_ROOT" "$DATA_ROOT" "$MODEL_ROOT/trained_models_LW"
```

### 2.1 Python 环境与依赖

如果服务器已有可用 conda/venv 环境，先激活对应环境；如果没有，可以在仓库内新建一个轻量环境。后续所有命令都用 `python -m pip`，避免 `pip` 和 `python` 不属于同一个环境。

```bash
cd "$CODE_ROOT" || exit 1
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -U modelscope
```

如果你的服务器需要显式安装 CUDA 版 PyTorch，可以先执行下面命令，再执行 `python -m pip install -r requirements.txt`。A100 常用 CUDA 12.1 wheel：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
python -m pip install -r requirements.txt
python -m pip install -U modelscope
```

依赖安装后先做导入检查。`nohup: ignoring input` 是 nohup 的正常提示，不是错误；真正需要看的是日志里的 Python traceback。

```bash
python -c "import torch, transformers, datasets, peft, accelerate; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('transformers', transformers.__version__)"
```

默认小模型配置：

```bash
export MODEL_PATH="$MODEL_ROOT/SmolLM2-135M"
export MODELSCOPE_ID=HuggingFaceTB/SmolLM2-135M
export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_LW7c/seed_42"
export RUN_TAG=SmolLM2-135M_LW7c
export TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_LW"

export ROUNDS=1
export LOCAL_EPOCHS=1
export LR=2e-4
export LORA_R=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export BATCH_SIZE=2
export MAX_SEQ_LENGTH=256
export TORCH_DTYPE=bfloat16
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export SEED=42
export EVAL_MAX_BATCHES=10
export EVAL_SEEDS=42
```

更小/更快的 smoke test 可以临时改：

```bash
export BATCH_SIZE=1
export MAX_SEQ_LENGTH=128
export EVAL_MAX_BATCHES=2
```

如果换成稍大的小模型，可替换为：

```bash
export MODEL_PATH="$MODEL_ROOT/Qwen2.5-0.5B"
export MODELSCOPE_ID=Qwen/Qwen2.5-0.5B
export RUN_TAG=Qwen2.5-0.5B_LW7c
```

## 3. 模型与数据准备

下载小模型。该命令需要网络和 ModelScope 权限；如果模型已在 `MODEL_PATH` 下存在且包含权重文件，可跳过。

```bash
nohup bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh > log_LWv5/download_smol_lm2_135m.log 2>&1 &
```

如果日志报 `Error no file named model.safetensors, or pytorch_model.bin`，说明 `$MODEL_PATH` 目录只下载了 `config.json` 等元信息，没有模型权重。先检查：

```bash
ls -lh "$MODEL_PATH"
find "$MODEL_PATH" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) -print
```

如果第二条没有任何输出，需要重新下载模型。为避免误删其他模型，先确认变量：

```bash
echo "$MODEL_PATH"
```

确认输出是 `/data2/minghao/model/SmolLM2-135M` 后，再清理半下载目录并重下：

```bash
rm -rf "$MODEL_PATH"
nohup bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh > log_LWv5/download_smol_lm2_135m_redownload.log 2>&1 &
tail -f log_LWv5/download_smol_lm2_135m_redownload.log
```

你的服务器当前已经在仓库内放好了数据：

```bash
ls "$DATA_ROOT"
ls "$BENCHMARK_DIR"/clients.json "$BENCHMARK_DIR"/train.jsonl "$BENCHMARK_DIR"/val.jsonl "$BENCHMARK_DIR"/test_domain.jsonl "$BENCHMARK_DIR"/test_local.jsonl "$BENCHMARK_DIR"/test_global.jsonl
```

如果上述文件都存在，直接跳过数据构建，后续所有训练命令可以直接使用：

```bash
export BENCHMARK_DIR="$CODE_ROOT/data/domain_benchmark_LW7c/seed_42"
```

只有当 `domain_benchmark_LW7c` 缺失或需要重新生成时，才从已有 35c benchmark 派生 LW7c。默认要求服务器上已有 `$DATA_ROOT/domain_benchmark_35c`：

```bash
nohup python -u scripts/DataProcessScripts/build_lw7c_benchmark.py \
  --src_35c "$DATA_ROOT/domain_benchmark_35c" \
  --output_dir "$DATA_ROOT/domain_benchmark_LW7c" \
  --seed 42 \
  > log_LWv5/build_lw7c_from_35c.log 2>&1 &
```

如果后续使用自己的多域 JSONL，保留这个接口即可。JSONL 每行至少包含 `domain`、`prompt`、`response` 三列。

```bash
nohup python -u scripts/DataProcessScripts/build_lw7c_benchmark.py \
  --from_jsonl "$DATA_ROOT/raw/domain_7_all.jsonl" \
  --output_dir "$DATA_ROOT/domain_benchmark_LW7c" \
  --seed 42 \
  --per_client_data_fraction 0.2 \
  --min_samples_per_client 10 \
  > log_LWv5/build_lw7c_from_jsonl.log 2>&1 &
```

检查数据和模型是否就绪：

```bash
ls "$MODEL_PATH"/config.json
find "$MODEL_PATH" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) -print
ls "$BENCHMARK_DIR"/clients.json "$BENCHMARK_DIR"/train.jsonl "$BENCHMARK_DIR"/val.jsonl "$BENCHMARK_DIR"/test_domain.jsonl "$BENCHMARK_DIR"/test_local.jsonl "$BENCHMARK_DIR"/test_global.jsonl
```

## 4. Baseline Sanity 命令

这些命令用于确认小模型数据流、checkpoint、metrics 都正常。推荐至少跑 `flexlora`、`flora`、`yoco`、`fedplora-oneshot`，并和后面的 `v5m_mean` 对齐比较。

### 4.1 FlexLoRA

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type flexlora \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/baseline_client_states/flexlora \
  --metrics_output_dir artifacts_LW7c/sft_metrics \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/flexlora_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --flora_svd_device auto \
  > log_LWv5/baseline_flexlora_seed${SEED}.log 2>&1 &
```

### 4.2 FLoRA

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type flora \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/baseline_client_states/flora \
  --metrics_output_dir artifacts_LW7c/sft_metrics \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/flora_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --flora_svd_device auto \
  > log_LWv5/baseline_flora_seed${SEED}.log 2>&1 &
```

### 4.3 YOCO

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type yoco \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/baseline_client_states/yoco \
  --metrics_output_dir artifacts_LW7c/sft_metrics \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/yoco_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --yoco_aggregate_mode conflict \
  --yoco_conflict_method avgm \
  --yoco_sign_lambda 0.01 \
  > log_LWv5/baseline_yoco_seed${SEED}.log 2>&1 &
```

### 4.4 FedPLoRA-Oneshot v2

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type fedplora-oneshot \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/baseline_client_states/fedplora_oneshot \
  --metrics_output_dir artifacts_LW7c/sft_metrics \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/fedplora_oneshot_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --oneshot_consensus_power 2.0 \
  --oneshot_importance_power 1.0 \
  --oneshot_importance_clip 5.0 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --oneshot_scale_clip_ratio 0.0 \
  > log_LWv5/baseline_fedplora_oneshot_seed${SEED}.log 2>&1 &
```

## 5. v5 Route-Mix-Align 命令

`v5` 仍然保持 A-only/one-shot 个性化路线：服务端聚合 A，客户端保留本地 B；下发后在客户端用验证集选择 `A_eff = eta * A_global + (1-eta) * A_local`，可选再做 B-only post alignment。

### 5.1 v5-local-route

部署设定，客户端只用本地验证集选一次 `eta`，不做 B-only 对齐。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5_route_mix_align \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir artifacts_LW7c/v5_client_states/local_route --save_client_state_to_disk \
  --metrics_output_dir artifacts_LW7c/v4_sft_metrics_v5/local_route \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5_local_route_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --log_dir log_LWv5 --log_filename_prefix LWv5_local_route \
  --v4_mix_save_dir artifacts_LW7c/v4_mix_a_local_v5_local_route \
  --v5_route_val_scope local \
  --v5_route_search_grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --v5_route_search_max_batches 2 \
  --v5_route_tie_margin 0.0 \
  --v5_route_tie_breaker best \
  --v5_route_post_align_steps 0 \
  --v5_route_post_align_lr 0.0001 \
  --v5_route_post_align_prox_lambda 0.0 \
  --v3_rpca_rank 1 \
  --v3_sparse_quantile 0.80 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --oneshot_consensus_power 2.0 \
  --oneshot_importance_power 1.0 \
  --oneshot_importance_clip 5.0 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --yoco_sparse_lambda 1e-4 \
  > log_LWv5/v5_local_route_seed${SEED}.log 2>&1 &
```

### 5.2 v5-local-route-align

部署设定，客户端选 `eta` 后再做 3 步本地 B-only 对齐。该版本通常比纯 route 更稳，但本地计算更多。

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5_route_mix_align \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir artifacts_LW7c/v5_client_states/local_route_align --save_client_state_to_disk \
  --metrics_output_dir artifacts_LW7c/v4_sft_metrics_v5/local_route_align \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5_local_route_align_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --log_dir log_LWv5 --log_filename_prefix LWv5_local_route_align \
  --v4_mix_save_dir artifacts_LW7c/v4_mix_a_local_v5_local_route_align \
  --v5_route_val_scope local \
  --v5_route_search_grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --v5_route_search_max_batches 2 \
  --v5_route_tie_margin 0.0 \
  --v5_route_tie_breaker best \
  --v5_route_post_align_steps 3 \
  --v5_route_post_align_lr 0.0001 \
  --v5_route_post_align_prox_lambda 0.0 \
  --v3_rpca_rank 1 \
  --v3_sparse_quantile 0.80 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --oneshot_consensus_power 2.0 \
  --oneshot_importance_power 1.0 \
  --oneshot_importance_clip 5.0 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --yoco_sparse_lambda 1e-4 \
  > log_LWv5/v5_local_route_align_seed${SEED}.log 2>&1 &
```

### 5.3 v5-domain-anchor-align

公开领域 anchor 上界设定，按评估域选择 `eta`，再做 B-only 对齐。该设置更适合诊断“如果有少量公开域验证数据，route 上限有多高”，不应直接当作最公平主表。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5_route_mix_align \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir artifacts_LW7c/v5_client_states/domain_anchor_align --save_client_state_to_disk \
  --metrics_output_dir artifacts_LW7c/v4_sft_metrics_v5/domain_anchor_align \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5_domain_anchor_align_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --log_dir log_LWv5 --log_filename_prefix LWv5_domain_anchor_align \
  --v4_mix_save_dir artifacts_LW7c/v4_mix_a_local_v5_domain_anchor_align \
  --v5_route_val_scope domain \
  --v5_route_search_grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --v5_route_search_max_batches 2 \
  --v5_route_tie_margin 0.0 \
  --v5_route_tie_breaker best \
  --v5_route_post_align_steps 3 \
  --v5_route_post_align_lr 0.0001 \
  --v5_route_post_align_prox_lambda 0.0 \
  --v3_rpca_rank 1 \
  --v3_sparse_quantile 0.80 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --oneshot_consensus_power 2.0 \
  --oneshot_importance_power 1.0 \
  --oneshot_importance_clip 5.0 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --yoco_sparse_lambda 1e-4 \
  > log_LWv5/v5_domain_anchor_align_seed${SEED}.log 2>&1 &
```

### 5.4 v5-rpca-local-route-align

服务端先做 common+sparse 残差分解，再走 local route + B-only alignment。该版本用于验证“跨域冲突中的稀疏个性化残差”是否能提升 worst/hard domain。

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5_rpca_route_mix_align \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir artifacts_LW7c/v5_client_states/rpca_local_route_align --save_client_state_to_disk \
  --metrics_output_dir artifacts_LW7c/v4_sft_metrics_v5/rpca_local_route_align \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5_rpca_local_route_align_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --log_dir log_LWv5 --log_filename_prefix LWv5_rpca_local_route_align \
  --v4_mix_save_dir artifacts_LW7c/v4_mix_a_local_v5_rpca_local_route_align \
  --v5_route_val_scope local \
  --v5_route_search_grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --v5_route_search_max_batches 2 \
  --v5_route_tie_margin 0.0 \
  --v5_route_tie_breaker best \
  --v5_route_post_align_steps 3 \
  --v5_route_post_align_lr 0.0001 \
  --v5_route_post_align_prox_lambda 0.0 \
  --v3_rpca_rank 1 \
  --v3_sparse_quantile 0.80 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --oneshot_consensus_power 2.0 \
  --oneshot_importance_power 1.0 \
  --oneshot_importance_clip 5.0 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --yoco_sparse_lambda 1e-4 \
  > log_LWv5/v5_rpca_local_route_align_seed${SEED}.log 2>&1 &
```


## 6. v6 / DCR 命令

`v6 / DCR` 是当前 A-only one-shot 主线：客户端仍只上传 LoRA A 和少量 row-importance 统计，B 保留在本地；服务器先做同域客户端的 Grassmann 子空间共识，再做跨域冲突消解。`v6_dcr_global` 下发单一全局 A，`v6_dcr_domain` 为每个域下发个性化 A。

### 6.1 v6_dcr_global

该版本用于验证“只做域内共识 + 全局 Grassmann 共识”的最简 one-shot A-only 路线。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v6_dcr_global \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v6_client_states/v6_dcr_global \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v6 \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v6_dcr_global_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --v6_dcr_rc_policy auto \
  --v6_dcr_energy_tau 0.80 \
  --v6_dcr_conflict_strength 1.0 \
  --v6_dcr_importance_power 0.0 \
  --v6_dcr_importance_clip 5.0 \
  > log_LWv6/v6_dcr_global_seed${SEED}.log 2>&1 &
```

### 6.2 v6_dcr_domain

该版本是 v6 的主推版本：保留 `r_c` 维跨域共享方向，并为高冲突域保留更多本域残差方向。通信仍是 A-only，一轮，每个客户端只收到一个 `r x d` 的 A。

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v6_dcr_domain \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v6_client_states/v6_dcr_domain \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v6 \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v6_dcr_domain_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --v6_dcr_rc_policy auto \
  --v6_dcr_energy_tau 0.80 \
  --v6_dcr_conflict_strength 1.0 \
  --v6_dcr_importance_power 0.0 \
  --v6_dcr_importance_clip 5.0 \
  > /data2/minghao/result/FedPLoRA/logs/log_LWv6_dcr_domain_seed${SEED}.log 2>&1 &
```

### 6.3 v6_dcr_domain 固定共享秩消融

如果 `auto` 不稳定，可以先用固定共享秩做消融。`--v6_dcr_shared_rank 4` 表示 rank=8 时一半共享、一半保留本域方向。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v6_dcr_domain \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v6_client_states/v6_dcr_domain_rc4 \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v6 \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v6_dcr_domain_rc4_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --v6_dcr_rc_policy fixed \
  --v6_dcr_shared_rank 4 \
  --v6_dcr_min_shared_rank 1 \
  --v6_dcr_max_shared_rank 8 \
  --v6_dcr_importance_power 0.0 \
  > log_LWv6/v6_dcr_domain_rc4_seed${SEED}.log 2>&1 &
```

### 6.4 可选脚本入口

如果只想快速启动 v6 两个默认方法，也可以使用仓库中的脚本：

```bash
nohup bash scripts/RunScripts/run_v6_dcr_lw7c.sh 0 v6_dcr_global > log_LWv6/run_script_v6_dcr_global_seed${SEED}.log 2>&1 &
nohup bash scripts/RunScripts/run_v6_dcr_lw7c.sh 1 v6_dcr_domain > log_LWv6/run_script_v6_dcr_domain_seed${SEED}.log 2>&1 &
```

## 7. v5-merge 命令

`v5-merge` 不再只在 A 矩阵上做线性聚合，而是在完整 LoRA 更新 `Delta W = B A` 上做干扰感知合并，再用 SVD 重分解回 LoRA。`fixed rank` 下通信与 FLoRA/FlexLoRA 对齐；`energy rank` 用于分析合并后需要多少 rank 才能保留能量。

先跑数学/工程自检：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m methods.v5.selftest_merge > log_LWv5/v5m_selftest_merge.log 2>&1 &
```

### 7.1 v5m_mean

`v5m_mean` 是 sanity：理论上应接近 `flexlora`，用于确认 `Delta W` 合并、head 处理、scaling 约定无 bug。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5m_mean \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v5m_client_states/v5m_mean \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v5merge \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5m_mean_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --v5m_keep_ratio 0.2 \
  --v5m_dare_p 0.3 \
  --v5m_rank_policy fixed \
  --v5m_rank_cap 32 \
  --v5m_energy_tau 0.95 \
  > log_LWv5/v5m_mean_seed${SEED}.log 2>&1 &
```

### 7.2 v5m_ties

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5m_ties \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v5m_client_states/v5m_ties \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v5merge \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5m_ties_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --v5m_keep_ratio 0.2 \
  --v5m_dare_p 0.3 \
  --v5m_rank_policy fixed \
  --v5m_rank_cap 32 \
  --v5m_energy_tau 0.95 \
  > log_LWv5/v5m_ties_seed${SEED}.log 2>&1 &
```

### 7.3 v5m_dare_ties

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5m_dare_ties \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v5m_client_states/v5m_dare_ties \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v5merge \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5m_dare_ties_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --v5m_keep_ratio 0.2 \
  --v5m_dare_p 0.3 \
  --v5m_rank_policy fixed \
  --v5m_rank_cap 32 \
  --v5m_energy_tau 0.95 \
  > log_LWv5/v5m_dare_ties_seed${SEED}.log 2>&1 &
```

### 7.4 v5m_knots_ties

这是 v5-merge 的主推版本：先做共享子空间对齐，再在对齐系数上做 TIES。

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5m_knots_ties \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v5m_client_states/v5m_knots_ties \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v5merge \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5m_knots_ties_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --v5m_keep_ratio 0.2 \
  --v5m_dare_p 0.3 \
  --v5m_rank_policy fixed \
  --v5m_rank_cap 32 \
  --v5m_energy_tau 0.95 \
  --v5m_knots_normalize 1 \
  --v5m_basis_energy_tau 0.9999 \
  --v5m_chunk_rows 2048 \
  --v5m_device auto \
  > log_LWv5/v5m_knots_ties_seed${SEED}.log 2>&1 &
```

### 7.5 v5m_knots_ties + energy rank

该命令不是通信公平主表，主要用于分析“合并后需要更高 rank 才能保留多少跨域信息”。如果使用 `energy`，建议保留独立 checkpoint，避免和 fixed rank 结果混淆。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5m_knots_ties \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v5m_client_states/v5m_knots_ties_energy \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v5merge \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5m_knots_ties_energy_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --skip_post_agg_snapshots \
  --v5m_keep_ratio 0.2 \
  --v5m_dare_p 0.3 \
  --v5m_rank_policy energy \
  --v5m_rank_cap 32 \
  --v5m_energy_tau 0.95 \
  --v5m_knots_normalize 1 \
  --v5m_basis_energy_tau 0.9999 \
  --v5m_chunk_rows 2048 \
  --v5m_device auto \
  > log_LWv5/v5m_knots_ties_energy_seed${SEED}.log 2>&1 &
```

## 8. v5.5 状态与待实现模板

当前仓库中没有 `v5pm_*` 相关实现，因此不能把下面模板当作可运行命令。`v5.5` 需要先实现以下入口：

- `methods/v5/fedplora_v5_personalized_merge.py`
- `utilities/utils.py` 中的 `is_v5_personalized_merge_agg`
- `tasks/fed_train_sft.py` 中的聚合 dispatch 与 personalized eval dispatch
- `scripts/RunScripts/run_v5pm_lw7c.sh` 或同等命令

待实现后的预期运行形式如下，当前不要执行：

```bash
# 当前不可运行：仓库尚未实现 agg_type=v5pm_knots。
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5pm_knots \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_LW7c/v5pm_client_states/v5pm_knots \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v5pm \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v5pm_knots_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --gradient_checkpointing \
  --v5m_keep_ratio 0.2 \
  --v5m_dare_p 0.3 \
  --v5m_rank_policy fixed \
  --v5m_rank_cap 32 \
  --v5m_energy_tau 0.95 \
  > log_LWv5/v5pm_knots_seed${SEED}.log 2>&1 &
```

## 9. 结果汇总命令

汇总 baseline、v6、v5 route、v5-merge 的主指标：

```bash
nohup python -u scripts/Analysis/summarize_v4.py \
  --inputs "artifacts_LW7c/sft_metrics/*.json" "artifacts_LW7c/sft_metrics_v6/*.json" "artifacts_LW7c/sft_metrics_v5merge/*.json" "artifacts_LW7c/v4_sft_metrics_v5/*/*.json" \
  --out artifacts_LW7c/summary/lw_small_model_summary.csv \
  > log_LWv5/summarize_lw_small_model.log 2>&1 &
```

汇总 v5 route 诊断，包括 `eta` 分布、global/local/mixed 占比、post-align loss：

```bash
nohup python -u scripts/Analysis/summarize_v5_routes.py \
  --inputs "artifacts_LW7c/v4_sft_metrics_v5/*/*.json" \
  --out artifacts_LW7c/summary/lw_v5_route_summary.csv \
  > log_LWv5/summarize_lw_v5_routes.log 2>&1 &
```

查看汇总结果：

```bash
tail -n 20 log_LWv5/summarize_lw_small_model.log
tail -n 20 log_LWv5/summarize_lw_v5_routes.log
cat artifacts_LW7c/summary/lw_small_model_summary.csv
cat artifacts_LW7c/summary/lw_v5_route_summary.csv
```

## 10. 日志与进程检查

查看单个实验日志：

```bash
tail -f log_LWv6/v6_dcr_domain_seed${SEED}.log
tail -f log_LWv6/v6_dcr_global_seed${SEED}.log
tail -f log_LWv5/v5m_knots_ties_seed${SEED}.log
tail -f log_LWv5/v5_local_route_align_seed${SEED}.log
```

查看后台进程和 GPU：

```bash
ps -ef | grep "tasks/fed_train_sft" | grep -v grep
nvidia-smi
```

如果发现某个方法因为 checkpoint 已存在而直接复用旧结果，需要换 `--save_run_checkpoint_dir` 或加 `--force_retrain`。推荐论文实验固定路径、固定 seed，不要混用不同超参的同名 checkpoint。

## 11. 从快筛切到主验证

快筛默认：

- `EVAL_MAX_BATCHES=10`
- `SEED=42`
- `ROUNDS=1`
- `LOCAL_EPOCHS=1`
- `LORA_R=8`

主验证建议：

- 将 `EVAL_MAX_BATCHES` 改为 `50` 或 `200`。
- 用 `SEED=42`、`SEED=1234`、`SEED=9999` 分别跑三次。
- `v6_dcr_domain` 是当前 A-only 主线，优先比较它与 `fedplora-oneshot`、`v5_route_mix_align`、`flora/flexlora` 的 macro/worst/hard domain。
- 保持 `v5m_rank_policy=fixed` 作为通信公平主表，`energy` 只放机制分析或 ablation。
- 若小模型上 `v5m_mean` 和 `flexlora` 差距明显大于随机波动，先排查 `v5-merge` 实现或 checkpoint 复用问题，再继续跑 `v5m_ties/knots`。

## 12. 35c 主实验命令

35c 是更适合展示 `v6_dcr_domain` 的主实验设置：7 个领域，每个领域 5 个 client，共 35 个参与者。它能体现“域内多客户端共识 + 跨域冲突消解 + 域个性化下发”，比 LW7c 更适合放论文主表。

建议不要一次性提交本节所有命令。两块 A100 可以先并行跑 `flexlora` 与 `fedplora-oneshot`，再跑 `v6_dcr_global` 与 `v6_dcr_domain`。确认主线跑通后，再补 `normal`、`ffa`、`flora`、`yoco`。

### 12.1 35c 环境变量

```bash
export CODE_ROOT=/home/minghao/code/FedPLoRA-main
export MODEL_ROOT=/data2/minghao/model
export DATA_ROOT="$CODE_ROOT/data"
export RESULT_ROOT=/data2/minghao/result/FedPLoRA
export LOG_ROOT="$RESULT_ROOT/logs_35c"

cd "$CODE_ROOT" || exit 1
mkdir -p "$LOG_ROOT" artifacts_35c/sft_metrics_baselines artifacts_35c/sft_metrics_v6 artifacts_35c/client_states artifacts_35c/summary "$MODEL_ROOT/trained_models_35c"

export MODEL_PATH="$MODEL_ROOT/SmolLM2-135M"
export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c/seed_42"
export RUN_TAG=SmolLM2-135M_35c
export TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_35c"

export ROUNDS=1
export LOCAL_EPOCHS=1
export LR=2e-4
export LORA_R=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export BATCH_SIZE=2
export MAX_SEQ_LENGTH=256
export TORCH_DTYPE=bfloat16
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export SEED=42
export EVAL_MAX_BATCHES=10
export MAX_TRAIN_SAMPLES_PER_CLIENT=0
```

检查 35c 数据和模型权重：

```bash
ls "$BENCHMARK_DIR"/clients.json "$BENCHMARK_DIR"/train.jsonl "$BENCHMARK_DIR"/val.jsonl "$BENCHMARK_DIR"/test_domain.jsonl "$BENCHMARK_DIR"/test_local.jsonl "$BENCHMARK_DIR"/test_global.jsonl
find "$MODEL_PATH" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) -print
python -c "import torch, transformers, datasets, peft; print('env ok', torch.__version__, transformers.__version__)"
```

如果只想先快速 smoke test，可以临时减少训练步数：

```bash
export EVAL_MAX_BATCHES=2
export MAX_TRAIN_SAMPLES_PER_CLIENT=64
```

正式 35c 主表建议取消 `MAX_TRAIN_SAMPLES_PER_CLIENT` 或设为 `0`：

```bash
export MAX_TRAIN_SAMPLES_PER_CLIENT=0
```

### 12.2 Baseline: FedAvg LoRA / normal

`normal` 是普通 LoRA 参数 FedAvg，上传与下发完整 trainable LoRA 状态，用作基础联邦 LoRA baseline。

如果出现 `/baseline_xxx.log: Permission denied`，说明当前 shell 没有执行 `12.1 35c 环境变量`，导致 `LOG_ROOT` 为空。先执行 `echo "$LOG_ROOT"`，正确输出应为 `/data2/minghao/result/FedPLoRA/logs_35c`。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type normal \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/normal \
  --metrics_output_dir artifacts_35c/sft_metrics_baselines \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/normal_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_baseline_num35_normal_seed${SEED}.log" 2>&1 &
```

### 12.3 Baseline: FFA-LoRA

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type ffa \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/ffa \
  --metrics_output_dir artifacts_35c/sft_metrics_baselines \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/ffa_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_baseline_num35_ffa_seed${SEED}.log" 2>&1 &
```

### 12.4 Baseline: FLoRA

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type flora \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/flora \
  --metrics_output_dir artifacts_35c/sft_metrics_baselines \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/flora_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --flora_svd_device auto \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_baseline_num35_flora_seed${SEED}.log" 2>&1 &
```

### 12.5 Baseline: FlexLoRA

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type flexlora \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/flexlora \
  --metrics_output_dir artifacts_35c/sft_metrics_baselines \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/flexlora_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --flora_svd_device auto \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_baseline_num35_flexlora_seed${SEED}.log" 2>&1 &
```

### 12.6 Baseline: YOCO

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type yoco \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/yoco \
  --metrics_output_dir artifacts_35c/sft_metrics_baselines \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/yoco_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --yoco_aggregate_mode conflict \
  --yoco_conflict_method avgm \
  --yoco_sign_lambda 0.01 \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_baseline_num35_yoco_seed${SEED}.log" 2>&1 &
```

### 12.7 Baseline: FedPLoRA-Oneshot v2

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type fedplora-oneshot \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/fedplora_oneshot \
  --metrics_output_dir artifacts_35c/sft_metrics_baselines \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/fedplora_oneshot_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --oneshot_consensus_power 2.0 \
  --oneshot_importance_power 1.0 \
  --oneshot_importance_clip 5.0 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --oneshot_scale_clip_ratio 0.0 \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_baseline_num35_fedplora_oneshot_seed${SEED}.log" 2>&1 &
```

### 12.8 Ours: FedPLoRA-v6 DCR Global

`v6_dcr_global` 用于验证“域内共识 + 单一全局 A 下发”的 A-only one-shot 版本，是 `v6_dcr_domain` 的必要对照。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v6_dcr_global \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/v6_dcr_global \
  --metrics_output_dir artifacts_35c/sft_metrics_v6 \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v6_dcr_global_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --v6_dcr_rc_policy auto \
  --v6_dcr_energy_tau 0.80 \
  --v6_dcr_conflict_strength 1.0 \
  --v6_dcr_importance_power 0.0 \
  --v6_dcr_importance_clip 5.0 \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_ours_num35_v6_dcr_global_seed${SEED}.log" 2>&1 &
```

### 12.9 Ours: FedPLoRA-v6 DCR Domain

`v6_dcr_domain` 是 35c 主推版本：每个域有 5 个 client，先形成域内 Grassmann 共识，再进行跨域冲突消解，并为不同域下发个性化 A。

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v6_dcr_domain \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/v6_dcr_domain \
  --metrics_output_dir artifacts_35c/sft_metrics_v6 \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v6_dcr_domain_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --v6_dcr_rc_policy auto \
  --v6_dcr_energy_tau 0.80 \
  --v6_dcr_conflict_strength 1.0 \
  --v6_dcr_importance_power 0.0 \
  --v6_dcr_importance_clip 5.0 \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_ours_num35_v6_dcr_domain_seed${SEED}.log" 2>&1 &
```

### 12.10 Ours Ablation: v6_dcr_domain 固定共享秩

该命令用于消融 `auto shared-rank`。`--v6_dcr_shared_rank 4` 表示 rank=8 时固定 4 维共享、4 维域残差。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v6_dcr_domain \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/client_states/v6_dcr_domain_rc4 \
  --metrics_output_dir artifacts_35c/sft_metrics_v6 \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/v6_dcr_domain_rc4_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  --gradient_checkpointing \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_prox_lambda 0.0 \
  --v6_dcr_rc_policy fixed \
  --v6_dcr_shared_rank 4 \
  --v6_dcr_min_shared_rank 1 \
  --v6_dcr_max_shared_rank 8 \
  --v6_dcr_importance_power 0.0 \
  > "/data2/minghao/result/FedPLoRA/logs/test0616_ours_num35_v6_dcr_domain_rc4_seed${SEED}.log" 2>&1 &
```

### 12.11 35c 进程与日志检查

```bash
ps -ef | grep "tasks/fed_train_sft.py" | grep -v grep
nvidia-smi

tail -f "$LOG_ROOT/ours_v6_dcr_domain_seed${SEED}.log"
tail -f "$LOG_ROOT/baseline_fedplora_oneshot_seed${SEED}.log"
tail -f "$LOG_ROOT/baseline_flexlora_seed${SEED}.log"
```

### 12.12 35c 结果汇总

```bash
nohup python -u scripts/Analysis/summarize_v4.py \
  --inputs "artifacts_35c/sft_metrics_baselines/*.json" "artifacts_35c/sft_metrics_v6/*.json" \
  --out artifacts_35c/summary/smol_lm2_35c_summary.csv \
  > "${LOG_ROOT:?请先执行 12.1 35c 环境变量}/summarize_35c.log" 2>&1 &
```

查看汇总：

```bash
tail -n 40 "$LOG_ROOT/summarize_35c.log"
cat artifacts_35c/summary/smol_lm2_35c_summary.csv
```

### 12.13 35c 主表建议

35c 首轮建议先跑：

```bash
flexlora
fedplora-oneshot
v6_dcr_global
v6_dcr_domain
```

如果 `v6_dcr_domain` 明显优于 `v6_dcr_global` 和 `fedplora-oneshot`，再补齐：

```bash
normal
ffa
flora
yoco
v6_dcr_domain_rc4
```

论文主表建议至少使用 `SEED=42,1234,9999` 三个 seed。每个 seed 重新设置 `SEED` 并复用同一组命令即可：

```bash
export SEED=1234
```

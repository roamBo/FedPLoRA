# FedPLoRA v6 / DCR 代码落地说明

v6 实现了 `claude/v6_domain_conflict_subspace.md` 中的 DCR（Domain Conflict Resolution via Subspace Consensus）主线：只上传 LoRA A，一轮通信，服务器在 A 的行子空间上做域内共识和跨域冲突消解。

## 已实现入口

- `v6_dcr_global`：域内 Grassmann 共识后生成单一全局 A，下发给所有客户端。
- `v6_dcr_domain`：域内共识 + 全局共识 + 本域残差方向，为每个域生成个性化 A，下发给该域客户端。
- 同义别名：`fedplora_dcr_global`、`fedplora_dcr_domain`、`v6_dcr`、`fedplora_dcr`。

## 代码位置

- 聚合器：`methods/v6/dcr.py`
- 入口导出：`methods/v6/__init__.py`
- 方法识别/通信统计：`utilities/utils.py`
- 训练入口接入：`tasks/fed_train_sft.py`
- LW7c 快筛脚本：`scripts/RunScripts/run_v6_dcr_lw7c.sh`

## 关键参数

- `--v6_dcr_mode ""|global|domain`：默认从 `agg_type` 推断。
- `--v6_dcr_rc_policy auto|fixed|energy`：domain 模式下共享 rank 分配策略。
- `--v6_dcr_shared_rank`：`fixed` 策略的共享 rank，0 表示默认 `lora_r//2`。
- `--v6_dcr_energy_tau`：按全局谱能量选择共享 rank 的阈值，默认 `0.80`。
- `--v6_dcr_conflict_strength`：auto 策略中域冲突对共享 rank 的影响强度，默认 `1.0`。
- `--v6_dcr_importance_power`：是否用本地 B 派生的 row importance 加权 A 子空间 SVD，默认 `0.0` 表示关闭。

## LW7c 快筛命令

使用脚本：

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
nohup bash scripts/RunScripts/run_v6_dcr_lw7c.sh 0 v6_dcr_global > log_LWv6/v6_dcr_global_lw7c.log 2>&1 &
nohup bash scripts/RunScripts/run_v6_dcr_lw7c.sh 1 v6_dcr_domain > log_LWv6/v6_dcr_domain_lw7c.log 2>&1 &
```

直接展开命令：

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
mkdir -p log_LWv6 artifacts_LW7c/sft_metrics_v6

CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/SmolLM2-135M \
  --benchmark_dir data/domain_benchmark_LW7c/seed_42 \
  --agg_type v6_dcr_domain \
  --rounds 1 --local_epochs 1 --lr 2e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --batch_size 2 --max_seq_length 256 \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --client_state_dir artifacts_LW7c/v6_client_states_domain \
  --save_client_state_to_disk \
  --metrics_output_dir artifacts_LW7c/sft_metrics_v6 \
  --trained_models_root ../trained_models_LW \
  --eval_max_batches 10 --seed 42 \
  --gradient_checkpointing \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --v6_dcr_rc_policy auto \
  --v6_dcr_energy_tau 0.80 \
  --v6_dcr_conflict_strength 1.0 \
  > log_LWv6/v6_dcr_domain_lw7c_direct.log 2>&1 &
```

## 35c 主验证命令模板

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
mkdir -p log_v6 artifacts_35c/sft_metrics_v6

CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type v6_dcr_domain \
  --rounds 1 --local_epochs 1 --lr 2e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --batch_size 2 --max_seq_length 2048 \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --client_state_dir artifacts_35c/v6_client_states_domain \
  --save_client_state_to_disk \
  --metrics_output_dir artifacts_35c/sft_metrics_v6 \
  --eval_max_batches 50 --seed 42 \
  --gradient_checkpointing \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --v6_dcr_rc_policy auto \
  --v6_dcr_energy_tau 0.80 \
  --v6_dcr_conflict_strength 1.0 \
  > log_v6/v6_dcr_domain_35c_seed42.log 2>&1 &
```

## 输出指标

每轮 JSON 中新增：

- `fedplora_v6_dcr_stats.mean_domain_pair_angle_deg`：跨域主夹角均值。
- `fedplora_v6_dcr_stats.mean_domain_pair_cos`：跨域子空间平均相似度。
- `fedplora_v6_dcr_stats.mean_global_energy_top_r`：全局共识谱前 r 维能量比例。
- `fedplora_v6_dcr_stats.mean_shared_rank`：domain 模式下平均共享 rank。
- `fedplora_v6_client_domains`：client 到 domain 的映射。

完整逐层诊断保存在运行时的 `_fedplora_v6_dcr_stats`，用于后续扩展机制图脚本。


# LW v4 轻量实验脚本

见 [FedPLoRAOSv4_README.md](../../FedPLoRAOSv4_README.md) §轻量级筛选（LW）。

```bash
bash scripts/RunScripts/LWv4/build_lw7c_benchmark.sh
bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh

# 仅经典对比 baseline（normal / yoco / fedalt / ffa 等，fed_train_sft.py）
bash scripts/RunScripts/LWv4/run_lwv4_baseline.sh 0

# baseline + v4 支线 A–F 全矩阵
bash scripts/RunScripts/LWv4/run_lwv4_all.sh 0
```

Baseline 指标写入 `artifacts_LW7c/sft_metrics/`；v4 支线写入 `artifacts_LW7c/v4_sft_metrics/`。

**标准 Alpaca non-IID（α=0.5）LW baseline**（与跨域 LW7c 独立）：

```bash
bash scripts/DataProcessScripts/build_alpaca_lw_standard_noniid_benchmark.sh
bash scripts/RunScripts/LWv4/run_lw_standard_noniid_baseline.sh 0
```

**FedPLoRA v2 + v4 全矩阵（Stanford Alpaca LW + SmolLM2-135M）**：

```bash
bash scripts/RunScripts/LWv4/run_lw_standard_fedplora_all.sh 0
```

| 产物 | 路径 |
|------|------|
| baseline 指标 | `artifacts_LW_standard/sft_metrics/` |
| v2/v4 指标 | `artifacts_LW_standard/v4_sft_metrics/` |
| checkpoint | `../trained_models_LW_standard/` |

见 `configs/lw_standard_noniid.env`。

#!/usr/bin/env bash
# 【机制消融】单轮 fedplora-oneshot（FedP 上传 + YOCO/PCWA 聚合）
# 多轮 fedplora 的 gp_* / fedplora_ablation_* 在 train_eval 与 aggregate_models_fedplora 中生效；
# oneshot 走 aggregate_models_yoco，本地仅 YOCO 稀疏项（yoco_sparse_lambda）与 PCWA 超参（yoco_pcwa_components）可消融。
# 产物：artifacts_{N}c/sft_metrics/* 与各 run 日志（N=客户端数）
#
# Usage:
#   bash scripts/RunScripts/run_exp_ablation_fedplora.sh [7|14|21|35] [gpu]
#
# 环境变量 ABLATION_MODE 可选：full | no_sparse | pcwa_k1
# 若未设置 ABLATION_MODE，则按顺序跑全部配置。

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/domain_sft_baselines.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft_baselines.env"
  set +a
fi

NC="${1:-35}"
GPU_CLI="${2:-}"
case "${NC}" in
  7|14|21|35) ;;
  *) echo "Usage: $0 [7|14|21|35] [gpu]" >&2; exit 1 ;;
esac

# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

BENCHMARK_DIR="${BENCHMARK_DIR:-data/domain_benchmark_${NC}c/seed_42}"
MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
ROUNDS="${ROUNDS:-1}"

YS="${YOCO_SPARSE_LAMBDA:-1e-4}"
PC="${YOCO_PCWA_COMPONENTS:-3}"

BASE=(
  python tasks/fed_train_sft.py
  --model "${MODEL_PATH}"
  --benchmark_dir "${BENCHMARK_DIR}"
  --agg_type fedplora-oneshot
  --rounds "${ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS:-1}"
  --lr "${LR:-2e-4}"
  --lora_r "${LORA_R:-8}"
  --lora_alpha "${LORA_ALPHA:-16}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --batch_size "${BATCH_SIZE:-2}"
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --target_modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  --gradient_checkpointing
  --save_client_state_to_disk
)

run_one() {
  local tag="$1"
  shift
  echo "[ablation fedplora-oneshot] ${tag}"
  # 独立目录：避免多组 fedplora-oneshot 同名 metrics 覆盖；且磁盘 client 状态若复用会污染下一组
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${BASE[@]}" \
    --client_state_dir "artifacts_${NC}c/domain_client_states_oneshot_ablation/${tag}" \
    --metrics_output_dir "artifacts_${NC}c/sft_metrics_oneshot_ablation/${tag}" \
    "$@"
}

if [[ -n "${ABLATION_MODE:-}" ]]; then
  MODES="${ABLATION_MODE}"
else
  MODES="full no_sparse pcwa_k1"
fi

echo "[exp_ablation_fedplora_oneshot] benchmark_dir=${BENCHMARK_DIR}"

for mode in ${MODES}; do
  case "${mode}" in
    full)
      run_one "full" \
        --yoco_sparse_lambda "${YS}" \
        --yoco_pcwa_components "${PC}" ;;
    no_sparse)
      run_one "no_sparse" \
        --yoco_sparse_lambda 0 \
        --yoco_pcwa_components "${PC}" ;;
    pcwa_k1)
      run_one "pcwa_k1" \
        --yoco_sparse_lambda "${YS}" \
        --yoco_pcwa_components 1 ;;
    no_align|no_prox|no_orth|no_consensus|no_momentum)
      echo "ABLATION_MODE=${mode} 仅适用于多轮 agg_type=fedplora（gp_* / fedplora_ablation_*）。oneshot 请用: full | no_sparse | pcwa_k1" >&2
      exit 1 ;;
    *)
      echo "Unknown ABLATION_MODE=${mode} (expected full|no_sparse|pcwa_k1)" >&2
      exit 1
      ;;
  esac
done

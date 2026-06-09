#!/usr/bin/env bash
# 【FedPLoRA-Oneshot v2 机制消融】35c 默认；3 个核心模块（与 full 主实验对照，不必重跑 full）。
#
# 论文对照：full = §11.1 主实验 fedplora-oneshot checkpoint / metrics（同超参即视为 full）。
# 本脚本默认只训 3 个「去掉单模块」变体：
#   wo_sparse   — 去掉本地 A 稀疏先验（--yoco_sparse_lambda 0）
#   wo_conflict — 去掉服务端冲突门控（--oneshot_ablation_plain_fedavg → 样本量加权 FedAvg）
#   wo_anchor   — 去掉本地 A0 锚定（--oneshot_anchor_lambda 0 --oneshot_prox_lambda 0）
#
# Usage (repo root):
#   bash scripts/RunScripts/run_exp_ablation_fedplora.sh [7|14|21|35] [gpu]
#
# 环境变量：
#   ABLATION_MODE     空格分隔：wo_sparse | wo_conflict | wo_anchor | full（默认前三项）
#   ABLATION_RUN_FULL=1  额外重跑 full 基线（一般不需要，主实验已有）
#   其余超参见 configs/domain_sft.env

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/domain_sft.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft.env"
  set +a
fi

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/domain_sft_agg_extra.inc.sh"

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
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
YS="${YOCO_SPARSE_LAMBDA:-1e-4}"

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
if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
  BASE+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  BASE+=(--trust_remote_code)
fi

_ablation_checkpoint_dir() {
  local tag="$1"
  ABL_TAG="${tag}" _REPO_ROOT="${_REPO_ROOT}" MODEL_PATH="${MODEL_PATH}" \
    BENCHMARK_DIR="${BENCHMARK_DIR}" ROUNDS="${ROUNDS}" LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}" \
    TRAINED_MODELS_ROOT="${TRAINED_MODELS_ROOT:-}" python - <<'PY'
import os, sys
from pathlib import Path
repo = Path(os.environ["_REPO_ROOT"])
sys.path.insert(0, str(repo))
from utilities.sft_checkpoint_paths import default_save_run_checkpoint_dir
base = default_save_run_checkpoint_dir(
    repo,
    os.environ.get("TRAINED_MODELS_ROOT") or None,
    agg_type="fedplora-oneshot",
    model_path=os.environ["MODEL_PATH"],
    benchmark_split_dir=os.environ["BENCHMARK_DIR"],
    rounds=int(os.environ["ROUNDS"]),
    local_epochs=int(os.environ.get("LOCAL_EPOCHS") or "1"),
    seed=int(os.environ.get("SEED") or "42"),
)
print(f"{base}_ablation_{os.environ['ABL_TAG']}")
PY
}

run_one() {
  local tag="$1"
  shift
  local ckpt_dir
  ckpt_dir="$(_ablation_checkpoint_dir "${tag}")"
  echo "[ablation] fedplora-oneshot tag=${tag} NC=${NC} GPU=${CUDA_DEVICES} ckpt=${ckpt_dir}"
  local -a CMD=("${BASE[@]}")
  domain_sft_append_agg_cli_extras CMD "fedplora-oneshot"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}" \
    --save_run_checkpoint_dir "${ckpt_dir}" \
    --client_state_dir "artifacts_${NC}c/domain_client_states_oneshot_ablation/${tag}" \
    --metrics_output_dir "artifacts_${NC}c/sft_metrics_oneshot_ablation/${tag}" \
    "$@"
}

if [[ -n "${ABLATION_MODE:-}" ]]; then
  MODES="${ABLATION_MODE}"
elif [[ "${ABLATION_RUN_FULL:-0}" == "1" ]]; then
  MODES="full wo_sparse wo_conflict wo_anchor"
else
  MODES="wo_sparse wo_conflict wo_anchor"
fi

echo "[exp_ablation_fedplora_oneshot] benchmark_dir=${BENCHMARK_DIR}"
echo "[exp_ablation_fedplora_oneshot] modes=${MODES} (full 对照请用主实验 §11.1，或 ABLATION_RUN_FULL=1)"

for mode in ${MODES}; do
  case "${mode}" in
    full)
      run_one "full" --yoco_sparse_lambda "${YS}" ;;
    wo_sparse)
      run_one "wo_sparse" --yoco_sparse_lambda 0 ;;
    wo_conflict)
      run_one "wo_conflict" \
        --yoco_sparse_lambda "${YS}" \
        --oneshot_ablation_plain_fedavg ;;
    wo_anchor)
      run_one "wo_anchor" \
        --yoco_sparse_lambda "${YS}" \
        --oneshot_anchor_lambda 0 \
        --oneshot_prox_lambda 0 ;;
    no_sparse)
      echo "[warn] ABLATION_MODE=no_sparse 已更名为 wo_sparse" >&2
      run_one "wo_sparse" --yoco_sparse_lambda 0 ;;
    pcwa_k1)
      echo "[warn] pcwa_k1 对 fedplora-oneshot 服务端无意义，已跳过" >&2
      ;;
    *)
      echo "Unknown ABLATION_MODE=${mode}" >&2
      echo "  可用: wo_sparse | wo_conflict | wo_anchor | full" >&2
      exit 1
      ;;
  esac
done

echo "[exp_ablation_fedplora_oneshot] done."

#!/usr/bin/env bash
# 【个性化收益分析】在固定 benchmark 上额外统计 client_local / in-domain / off-domain macro 等，
# 见 fed_train_sft.py 中 --eval_personalization_metrics。
# 默认（推荐）：不再训练，直接读取主实验保存的 run checkpoint，只做前向 eval。
#  路径与训练时 Python 解析的默认 bundle 一致（../trained_models/<stem>/）。
#  任意在主实验里跑过的 agg_type，把名字加进 PERSONALIZATION_AGG_LIST 即可复评。
#
# Usage (repo root):
#   bash scripts/RunScripts/run_exp_personalization.sh [7|14|21|35] [gpu]
#
# 环境变量：
#   PERSONALIZATION_FROM_CHECKPOINT=1（默认）  eval-only，读主实验 checkpoint
#   PERSONALIZATION_FROM_CHECKPOINT=0          旧行为：当场训练 + 开个性化指标
#   TRAINED_MODELS_ROOT=…                      与训练时一致（默认同训练：仓库同级 trained_models）
#   PERSONALIZATION_AGG_LIST=fedplora-oneshot,normal   逗号分隔
#   BENCHMARK_DIR / MODEL_PATH / EVAL_MAX_BATCHES / CUDA_DEVICES 等同 configs/domain_sft.env
#
# GPU：第二参数传 0 / 1 / 0,1；不写且未 export CUDA_DEVICES 时，nvidia-smi 选空闲显存最大的卡（configs/cuda_resolve.inc.sh）

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
PERSONALIZATION_FROM_CHECKPOINT="${PERSONALIZATION_FROM_CHECKPOINT:-1}"
PERSONALIZATION_AGG_LIST="${PERSONALIZATION_AGG_LIST:-fedplora-oneshot,normal}"

COMMON_BASE=(
  python tasks/fed_train_sft.py
  --model "${MODEL_PATH}"
  --benchmark_dir "${BENCHMARK_DIR}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --target_modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  --client_state_dir "${CLIENT_STATE_DIR:-artifacts/domain_client_states}"
  --seed "${SEED:-42}"
)

if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
  COMMON_BASE+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  COMMON_BASE+=(--trust_remote_code)
fi

echo "[exp_personalization] benchmark_dir=${BENCHMARK_DIR} NC=${NC} EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-off} TRAINED_MODELS_ROOT=${TRAINED_MODELS_ROOT:-<default>} FROM_CKPT=${PERSONALIZATION_FROM_CHECKPOINT} AGG_LIST=${PERSONALIZATION_AGG_LIST}"

if [[ "${PERSONALIZATION_FROM_CHECKPOINT}" == "1" ]]; then
  _resolve_bundle_ckpt() {
    local agg="$1"
    local bench="${BENCHMARK_DIR}"
    if [[ "${bench}" != /* ]]; then
      bench="${_REPO_ROOT}/${bench}"
    fi
    local _extra=()
    if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
      _extra=(--trained_models_root "${TRAINED_MODELS_ROOT}")
    fi
    python "${_REPO_ROOT}/utilities/sft_checkpoint_paths.py" \
      --repo_root "${_REPO_ROOT}" \
      --agg_type "${agg}" \
      --model "${MODEL_PATH}" \
      --benchmark_dir "${bench}" \
      --rounds "${ROUNDS}" \
      --local_epochs "${LOCAL_EPOCHS:-1}" \
      --seed "${SEED:-42}" \
      "${_extra[@]}"
  }

  _run_eval_only() {
    local agg="$1"
    local ckpt
    ckpt="$(_resolve_bundle_ckpt "${agg}")"
    if [[ ! -f "${ckpt}/run_checkpoint_meta.json" ]]; then
      echo "[error] 缺少主实验 checkpoint（请先训练或检查路径）：${ckpt}" >&2
      echo "  期望存在: ${ckpt}/run_checkpoint_meta.json" >&2
      exit 1
    fi
    echo "[run] eval-only personalization metrics  agg_type=${agg}  <--eval_only_from_checkpoint ${ckpt}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      "${COMMON_BASE[@]}" \
      --agg_type "${agg}" \
      --eval_only_from_checkpoint "${ckpt}" \
      --eval_personalization_metrics
  }

  IFS=',' read -r -a _AGGS <<< "${PERSONALIZATION_AGG_LIST}"
  for _raw in "${_AGGS[@]}"; do
    agg="${_raw#"${_raw%%[![:space:]]*}"}"
    agg="${agg%"${agg##*[![:space:]]}"}"
    [[ -z "${agg}" ]] && continue
    _run_eval_only "${agg}"
  done
  exit 0
fi

# ----- 以下为旧路径：当场训练并（可选）另存 checkpoint -----

COMMON_TRAIN=(
  "${COMMON_BASE[@]}"
  --rounds "${ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS:-1}"
  --lr "${LR:-2e-4}"
  --lora_r "${LORA_R:-8}"
  --lora_alpha "${LORA_ALPHA:-16}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --batch_size "${BATCH_SIZE:-2}"
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}"
  --gradient_checkpointing
  --eval_personalization_metrics
)

ONESHOT_EXTRA=()
if [[ "${ONESHOT_NO_KEEP_INIT_ON_CONFLICT:-0}" == "1" ]]; then
  ONESHOT_EXTRA+=(--oneshot_no_keep_init_on_conflict)
fi
if [[ "${ONESHOT_ORTHOGONALIZE:-0}" == "1" ]]; then
  ONESHOT_EXTRA+=(--oneshot_orthogonalize)
fi
if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
  ONESHOT_EXTRA+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
fi

echo "[run] train fedplora-oneshot + personalization metrics"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${COMMON_TRAIN[@]}" \
  --agg_type fedplora-oneshot \
  --save_client_state_to_disk \
  --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA:-1e-4}" \
  --oneshot_anchor_lambda "${ONESHOT_ANCHOR_LAMBDA:-1e-4}" \
  --oneshot_prox_lambda "${ONESHOT_PROX_LAMBDA:-0.0}" \
  --oneshot_consensus_power "${ONESHOT_CONSENSUS_POWER:-2.0}" \
  --oneshot_importance_power "${ONESHOT_IMPORTANCE_POWER:-1.0}" \
  --oneshot_importance_clip "${ONESHOT_IMPORTANCE_CLIP:-5.0}" \
  --oneshot_conflict_threshold "${ONESHOT_CONFLICT_THRESHOLD:-0.35}" \
  --oneshot_conflict_blend "${ONESHOT_CONFLICT_BLEND:-1.0}" \
  --oneshot_scale_clip_ratio "${ONESHOT_SCALE_CLIP_RATIO:-0.0}" \
  "${ONESHOT_EXTRA[@]}"

NORMAL_EXTRA=()
if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
  NORMAL_EXTRA+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
fi

echo "[run] train normal + personalization metrics"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${COMMON_TRAIN[@]}" \
  --agg_type normal \
  "${NORMAL_EXTRA[@]}"

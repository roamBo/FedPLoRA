#!/usr/bin/env bash
# Optional first argument: GPU index or list. If omitted, auto-pick or fallback; see configs/cuda_resolve.inc.sh.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"
if [[ -f "${_REPO_ROOT}/configs/domain_sft_pilot.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft_pilot.env"
  set +a
fi

GPU_CLI="${1:-}"
# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_fed_train_speed.inc.sh"

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/domain_benchmark_35c/seed_42}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
AGG_TYPE="${AGG_TYPE:-fedplora}"
ROUNDS="${ROUNDS:-1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
SAVE_CLIENT_STATE_TO_DISK="${SAVE_CLIENT_STATE_TO_DISK:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts/domain_client_states}"
YOCO_SPARSE_LAMBDA="${YOCO_SPARSE_LAMBDA:-1e-4}"
YOCO_PCWA_COMPONENTS="${YOCO_PCWA_COMPONENTS:-3}"
ONESHOT_ANCHOR_LAMBDA="${ONESHOT_ANCHOR_LAMBDA:-1e-4}"
ONESHOT_PROX_LAMBDA="${ONESHOT_PROX_LAMBDA:-0.0}"
ONESHOT_CONSENSUS_POWER="${ONESHOT_CONSENSUS_POWER:-2.0}"
ONESHOT_IMPORTANCE_POWER="${ONESHOT_IMPORTANCE_POWER:-1.0}"
ONESHOT_IMPORTANCE_CLIP="${ONESHOT_IMPORTANCE_CLIP:-5.0}"
ONESHOT_CONFLICT_THRESHOLD="${ONESHOT_CONFLICT_THRESHOLD:-0.35}"
ONESHOT_CONFLICT_BLEND="${ONESHOT_CONFLICT_BLEND:-1.0}"
ONESHOT_SCALE_CLIP_RATIO="${ONESHOT_SCALE_CLIP_RATIO:-0.0}"
ONESHOT_NO_KEEP_INIT_ON_CONFLICT="${ONESHOT_NO_KEEP_INIT_ON_CONFLICT:-0}"
ONESHOT_ORTHOGONALIZE="${ONESHOT_ORTHOGONALIZE:-0}"

CMD=(
  python tasks/fed_train_sft.py
  --model "${MODEL_PATH}"
  --benchmark_dir "${BENCHMARK_DIR}"
  --agg_type "${AGG_TYPE}"
  --rounds "${ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS}"
  --lr "${LR}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --batch_size "${BATCH_SIZE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --torch_dtype "${TORCH_DTYPE}"
  --target_modules "${TARGET_MODULES}"
  --client_state_dir "${CLIENT_STATE_DIR}"
  --gradient_checkpointing
)
if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
  CMD+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
fi

if [[ "${SAVE_CLIENT_STATE_TO_DISK}" == "1" ]]; then
  CMD+=(--save_client_state_to_disk)
fi

if [[ "${AGG_TYPE}" == "fedplora-oneshot" || "${AGG_TYPE}" == "fedplora_oneshot" ]]; then
  CMD+=(
    --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}"
    --oneshot_anchor_lambda "${ONESHOT_ANCHOR_LAMBDA}"
    --oneshot_prox_lambda "${ONESHOT_PROX_LAMBDA}"
    --oneshot_consensus_power "${ONESHOT_CONSENSUS_POWER}"
    --oneshot_importance_power "${ONESHOT_IMPORTANCE_POWER}"
    --oneshot_importance_clip "${ONESHOT_IMPORTANCE_CLIP}"
    --oneshot_conflict_threshold "${ONESHOT_CONFLICT_THRESHOLD}"
    --oneshot_conflict_blend "${ONESHOT_CONFLICT_BLEND}"
    --oneshot_scale_clip_ratio "${ONESHOT_SCALE_CLIP_RATIO}"
  )
  if [[ "${ONESHOT_NO_KEEP_INIT_ON_CONFLICT}" == "1" ]]; then
    CMD+=(--oneshot_no_keep_init_on_conflict)
  fi
  if [[ "${ONESHOT_ORTHOGONALIZE}" == "1" ]]; then
    CMD+=(--oneshot_orthogonalize)
  fi
fi

if [[ "${AGG_TYPE}" == "yoco" ]]; then
  CMD+=(
    --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}"
    --yoco_pcwa_components "${YOCO_PCWA_COMPONENTS}"
  )
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

fed_train_append_speed_flags CMD

if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
  CMD+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"

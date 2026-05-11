#!/usr/bin/env bash
# 【个性化收益分析】在固定 benchmark 上额外统计：
#   - 各客户端本地 held-out（test_local）上的 macro；
#   - 本域 test_domain / 非本域 test_domain 上的 macro；
#   - gap（local vs off-domain 等），见 fed_train_sft.py 中 --eval_personalization_metrics。
# 与仅看 domain-macro 平均相比，用来衡量「 specialization 是否以牺牲跨域为代价」。
#
# 默认（推荐）：不再训练，直接读取 7 域主实验在 §11.1 里保存的 run checkpoint，只做前向 eval。
#   路径约定与 README 一致：${CHECKPOINT_ROOT}/${NC}c_<agg_type>/
#   例如 NC=35 时：artifacts/checkpoints/35c_fedplora-oneshot、artifacts/checkpoints/35c_normal
# 任意在主实验里用过 --save_run_checkpoint_dir 的方法，只要把对应目录加进 PERSONALIZATION_AGG_LIST 即可复评
#（agg_type 与目录名后缀一致，如 fedsa_lora、lora_a2、fedplora-oneshot）。
#
# Usage (repo root):
#   bash scripts/RunScripts/run_exp_personalization.sh [7|14|21|35] [gpu]
#
# 环境变量：
#   PERSONALIZATION_FROM_CHECKPOINT=1（默认）  eval-only，读主实验 checkpoint
#   PERSONALIZATION_FROM_CHECKPOINT=0          旧行为：当场训练 + 开个性化指标（仍可用 SAVE_RUN_CHECKPOINT_ROOT 另存）
#   CHECKPOINT_ROOT=artifacts/checkpoints      与 README §11.1 中 --save_run_checkpoint_dir 父目录一致
#   PERSONALIZATION_AGG_LIST=fedplora-oneshot,normal   逗号分隔；可改成 fedplora,normal,yoco,... 以扫多种方法
#   BENCHMARK_DIR / MODEL_PATH / EVAL_MAX_BATCHES / CUDA_DEVICES 等同 domain_sft_baselines.env
#
# GPU：第二参数传 0 / 1 / 0,1；不写且未 export CUDA_DEVICES 时，nvidia-smi 选空闲显存最大的卡（configs/cuda_resolve.inc.sh）

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
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-artifacts/checkpoints}"
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

echo "[exp_personalization] benchmark_dir=${BENCHMARK_DIR} NC=${NC} EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-off} CHECKPOINT_ROOT=${CHECKPOINT_ROOT} FROM_CKPT=${PERSONALIZATION_FROM_CHECKPOINT} AGG_LIST=${PERSONALIZATION_AGG_LIST}"

if [[ "${PERSONALIZATION_FROM_CHECKPOINT}" == "1" ]]; then
  _run_eval_only() {
    local agg="$1"
    local rel="${CHECKPOINT_ROOT}/${NC}c_${agg}"
    local ckpt
    if [[ "${rel}" = /* ]]; then
      ckpt="${rel}"
    else
      ckpt="${_REPO_ROOT}/${rel}"
    fi
    if [[ ! -f "${ckpt}/run_checkpoint_meta.json" ]]; then
      echo "[error] 缺少主实验 checkpoint（请先按 README §11.1 训练并保存）：${ckpt}" >&2
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
if [[ -n "${SAVE_RUN_CHECKPOINT_ROOT:-}" ]]; then
  ONESHOT_EXTRA+=(--save_run_checkpoint_dir "${SAVE_RUN_CHECKPOINT_ROOT}/fedplora-oneshot")
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
if [[ -n "${SAVE_RUN_CHECKPOINT_ROOT:-}" ]]; then
  NORMAL_EXTRA+=(--save_run_checkpoint_dir "${SAVE_RUN_CHECKPOINT_ROOT}/normal")
fi

echo "[run] train normal + personalization metrics"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${COMMON_TRAIN[@]}" \
  --agg_type normal \
  "${NORMAL_EXTRA[@]}"

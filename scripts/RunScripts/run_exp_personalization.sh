#!/usr/bin/env bash
# 【个性化收益分析】基于 §11.1 主实验已保存的 checkpoint，eval-only + --eval_personalization_metrics。
# 不重新训练；输出 JSON 到 artifacts_{N}c/sft_metrics/（文件名含 eval_ckpt_<stem>）。
#
# Usage (repo root):
#   bash scripts/RunScripts/run_exp_personalization.sh [7|14|21|35] [gpu]
# 默认第一个参数为 35（与 configs/domain_sft.env 主实验一致）。
#
# 环境变量：
#   PERSONALIZATION_AGG_LIST   逗号分隔 agg_type（默认 12 方法，与 §11.3.3 一致）
#   PERSONALIZATION_SKIP_MISSING=1（默认）  缺 checkpoint 则跳过该项，不中断整批
#   PERSONALIZATION_REQUIRE_FINAL=1（默认）  仅接受根目录 checkpoint_ok phase=final
#   TRAINED_MODELS_ROOT / BENCHMARK_DIR / MODEL_PATH / EVAL_MAX_BATCHES  同 domain_sft.env

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
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
PERSONALIZATION_SKIP_MISSING="${PERSONALIZATION_SKIP_MISSING:-1}"
PERSONALIZATION_REQUIRE_FINAL="${PERSONALIZATION_REQUIRE_FINAL:-1}"

DEFAULT_AGG_LIST="fedplora-oneshot,fedalt,yoco,fedsa_lora,normal,flora,flexlora,ffa,feddat,fedplora_v3_lite,fedplora_v3_cluster,fedplora_v3_rpca"
PERSONALIZATION_AGG_LIST="${PERSONALIZATION_AGG_LIST:-${DEFAULT_AGG_LIST}}"

COMMON_BASE=(
  python tasks/fed_train_sft.py
  --model "${MODEL_PATH}"
  --benchmark_dir "${BENCHMARK_DIR}"
  --rounds "${ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --target_modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  --client_state_dir "${CLIENT_STATE_DIR:-artifacts/domain_client_states}"
  --seed "${SEED:-42}"
  --gradient_checkpointing
  --eval_personalization_metrics
)

if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
  COMMON_BASE+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  COMMON_BASE+=(--trust_remote_code)
fi
if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
  COMMON_BASE+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
fi

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
    --local_epochs "${LOCAL_EPOCHS}" \
    --seed "${SEED:-42}" \
    "${_extra[@]}"
}

_checkpoint_ok_final() {
  local ckpt="$1"
  local okf="${ckpt}/checkpoint_ok.json"
  [[ -f "${okf}" ]] || return 1
  python - <<'PY' "${okf}"
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    o = json.load(f)
sys.exit(0 if o.get("ok") and str(o.get("checkpoint_phase", "final")) == "final" else 1)
PY
}

_run_eval_only() {
  local agg="$1"
  local ckpt
  ckpt="$(_resolve_bundle_ckpt "${agg}")"
  if [[ ! -f "${ckpt}/run_checkpoint_meta.json" ]]; then
    if [[ "${PERSONALIZATION_SKIP_MISSING}" == "1" ]]; then
      echo "[skip] 无 checkpoint: agg=${agg} path=${ckpt}" >&2
      return 0
    fi
    echo "[error] 缺少主实验 checkpoint：${ckpt}" >&2
    exit 1
  fi
  if [[ "${PERSONALIZATION_REQUIRE_FINAL}" == "1" ]] && ! _checkpoint_ok_final "${ckpt}"; then
    if [[ "${PERSONALIZATION_SKIP_MISSING}" == "1" ]]; then
      echo "[skip] checkpoint 非 final（请先跑完主实验 eval）：${ckpt}" >&2
      return 0
    fi
    echo "[error] 需要 phase=final 的 checkpoint_ok.json：${ckpt}" >&2
    exit 1
  fi

  local -a CMD=("${COMMON_BASE[@]}" --agg_type "${agg}" --eval_only_from_checkpoint "${ckpt}")
  domain_sft_append_agg_cli_extras CMD "${agg}"

  echo "[run] personalization eval-only  agg=${agg}"
  echo "      checkpoint=${ckpt}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"
}

echo "[exp_personalization] NC=${NC} benchmark=${BENCHMARK_DIR} GPU=${CUDA_DEVICES}"
echo "[exp_personalization] TRAINED_MODELS_ROOT=${TRAINED_MODELS_ROOT:-<repo>/../trained_models}"
echo "[exp_personalization] AGG_LIST=${PERSONALIZATION_AGG_LIST}"

FAILED=0
DONE=0
IFS=',' read -r -a _AGGS <<< "${PERSONALIZATION_AGG_LIST}"
for _raw in "${_AGGS[@]}"; do
  agg="${_raw#"${_raw%%[![:space:]]*}"}"
  agg="${agg%"${agg##*[![:space:]]}"}"
  [[ -z "${agg}" ]] && continue
  if _run_eval_only "${agg}"; then
    DONE=$((DONE + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done

echo "[exp_personalization] finished ok=${DONE} failed=${FAILED}"
[[ "${FAILED}" -eq 0 ]]

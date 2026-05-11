# shellcheck shell=bash
# Source from repo RunScripts after cd to repo root and optional configs/*.env:
#   # shellcheck disable=SC1091
#   source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
#   cuda_resolve_devices "${GPU_CLI:-}"
#
# Resolution order:
#   1) CUDA_DEVICES already set (non-empty) — from export or prefix on command line
#   2) First argument to cuda_resolve_devices — single index or comma list, e.g. "1" or "0,1"
#   3) AUTO_CUDA_PICK=0 — use CUDA_DEVICES_FALLBACK (default "0")
#   4) nvidia-smi — GPU with largest free memory (MiB)
#   5) CUDA_DEVICES_FALLBACK (default "0")

cuda_pick_max_free_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  # CSV: index, memory.free (MiB with nounits)
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '
      BEGIN { max=-1; idx="" }
      NF>=2 {
        gsub(/^ +| +$/,"", $1)
        gsub(/^ +| +$/,"", $2)
        if ($2+0 > max) { max=$2+0; idx=$1 }
      }
      END { if (idx != "") print idx }
    '
}

cuda_resolve_devices() {
  local cli="${1:-}"

  if [[ -n "${CUDA_DEVICES:-}" ]]; then
    export CUDA_DEVICES
    echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (environment)" >&2
    return 0
  fi

  if [[ -n "${cli}" ]]; then
    export CUDA_DEVICES="${cli}"
    echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (CLI)" >&2
    return 0
  fi

  if [[ "${AUTO_CUDA_PICK:-1}" == "0" ]]; then
    export CUDA_DEVICES="${CUDA_DEVICES_FALLBACK:-0}"
    echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (AUTO_CUDA_PICK=0)" >&2
    return 0
  fi

  local pick
  pick="$(cuda_pick_max_free_gpu || true)"
  if [[ -n "${pick}" ]]; then
    export CUDA_DEVICES="${pick}"
    echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (auto: max free VRAM)" >&2
    return 0
  fi

  export CUDA_DEVICES="${CUDA_DEVICES_FALLBACK:-0}"
  echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (fallback: no nvidia-smi)" >&2
}

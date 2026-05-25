# shellcheck shell=bash
# Shared GPU resolution for FedPLoRA-v4 RunScripts.
# Call after V4_ROOT is set:
#   v4_resolve_gpu "${1:-}"
#
# Priority: CUDA_DEVICES env > first script arg (e.g. 0 or 0,1) > auto max-free VRAM.

v4_resolve_gpu() {
  local _repo_root
  _repo_root="$(cd "${V4_ROOT}/.." && pwd)"
  # Script positional arg wins over v4_baseline.env CUDA_DEVICES=0,1
  if [[ -n "${1:-}" ]]; then
    export CUDA_DEVICES="${1}"
    echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (script arg)" >&2
    return 0
  fi
  # shellcheck disable=SC1091
  source "${_repo_root}/configs/cuda_resolve.inc.sh"
  cuda_resolve_devices ""
}

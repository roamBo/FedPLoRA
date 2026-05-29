# shellcheck shell=bash
# GPU resolution for v4 RunScripts (call after _REPO_ROOT is set and cwd is repo root).

v4_resolve_gpu() {
  if [[ -n "${1:-}" ]]; then
    export CUDA_DEVICES="${1}"
    echo "[cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (script arg)" >&2
    return 0
  fi
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
  cuda_resolve_devices ""
}

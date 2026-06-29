#!/usr/bin/env bash
# Download ~1B diagnostic backbones (base / pretrain, not Instruct) via ModelScope.
# Local dirs sit next to Meta-Llama-3.1-8B under MODEL_ROOT.
#
# Usage (repo root):
#   bash scripts/RunScripts/download_diag_models_modelscope.sh          # all three
#   bash scripts/RunScripts/download_diag_models_modelscope.sh opt      # one of: opt | pythia | qwen
#
# Override:
#   MODEL_ROOT=/data/yaominghao/gb/models bash scripts/RunScripts/download_diag_models_modelscope.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

MODEL_ROOT="${MODEL_ROOT:-/data/yaominghao/gb/models}"
mkdir -p "${MODEL_ROOT}"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "[diag-models] installing modelscope CLI ..."
  pip install -U modelscope
fi

_download_one() {
  local ms_id="$1"
  local local_name="$2"
  local dest="${MODEL_ROOT}/${local_name}"

  if [[ -f "${dest}/config.json" ]]; then
    echo "[diag-models] skip (exists): ${dest}"
    return 0
  fi

  echo "[diag-models] downloading ${ms_id} -> ${dest}"
  modelscope download --model "${ms_id}" --local_dir "${dest}"
  echo "[diag-models] done: ${dest}"
}

_download_opt() {
  _download_one "facebook/opt-1.3b" "OPT-1.3B"
}

_download_pythia() {
  _download_one "EleutherAI/pythia-1.4b" "Pythia-1.4B"
}

_download_qwen() {
  _download_one "Qwen/Qwen2.5-1.5B" "Qwen2.5-1.5B"
}

case "${1:-all}" in
  all|"")
    _download_opt
    _download_pythia
    _download_qwen
    ;;
  opt|OPT|opt-1.3b|OPT-1.3B)
    _download_opt
    ;;
  pythia|Pythia|pythia-1.4b|Pythia-1.4B)
    _download_pythia
    ;;
  qwen|Qwen|qwen2.5-1.5b|Qwen2.5-1.5B)
    _download_qwen
    ;;
  *)
    echo "Usage: $0 [all|opt|pythia|qwen]" >&2
    exit 1
    ;;
esac

echo "[diag-models] MODEL_ROOT=${MODEL_ROOT}"

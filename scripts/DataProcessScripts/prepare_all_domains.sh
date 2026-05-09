#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SHUFFLE="${SHUFFLE:-1}"
SEED="${SEED:-42}"

run_prepare() {
  local script_name="$1"
  shift
  local cmd=(
    "${PYTHON_BIN}" "${_SCRIPT_DIR}/${script_name}"
    --seed "${SEED}"
  )
  if [[ "${MAX_SAMPLES}" != "0" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${SHUFFLE}" == "1" ]]; then
    cmd+=(--shuffle)
  fi
  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi
  "${cmd[@]}"
}

run_prepare prepare_general_data.py
run_prepare prepare_math_data.py
run_prepare prepare_code_data.py
run_prepare prepare_medical_data.py
run_prepare prepare_legal_data.py
run_prepare prepare_finance_data.py
run_prepare prepare_education_data.py

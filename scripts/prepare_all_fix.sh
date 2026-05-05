#!/usr/bin/env bash
set -euo pipefail

# A more robust "prepare all domains" script:
# - Exports each domain into a stable output path under data/domain_sources/<domain>/
# - Skips a domain if its output file already exists (unless SKIP_EXISTING=0)
# - Adds the missing config_name for the code dataset (OpenCoder-LLM/opc-sft-stage1)
#
# Usage (recommended pilot):
#   source configs/domain_data_pilot.env
#   bash scripts/prepare_all_fix.sh
#
# Optional overrides:
#   MAX_SAMPLES=2000 SHUFFLE=1 SEED=42 bash scripts/prepare_all_fix.sh
#   CODE_CONFIG=realuser_instruct bash scripts/prepare_all_fix.sh
#   SKIP_EXISTING=0 bash scripts/prepare_all_fix.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SHUFFLE="${SHUFFLE:-1}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# OpenCoder dataset requires a config/subset.
CODE_CONFIG="${CODE_CONFIG:-filtered_infinity_instruct}"

# Medical dataset also has multiple configs (language/mix).
# Default to English to match most downstream model eval setups.
MEDICAL_CONFIG="${MEDICAL_CONFIG:-en}"

# LawInstruct also requires a config; default to a smaller English split.
LEGAL_CONFIG="${LEGAL_CONFIG:-LegalQA-legal_qa-1_english}"

# datasets>=3 removed support for many "dataset script" repos.
# This repo's requirements pin datasets==2.20.0; enforce a helpful error here.
python - <<'PY'
import sys
try:
    import datasets
except Exception as e:
    print(f"[error] failed to import datasets: {e}")
    sys.exit(2)

ver = getattr(datasets, "__version__", "0.0.0")
major = int(ver.split(".")[0]) if ver.split(".")[0].isdigit() else 0
if major >= 3:
    print(f"[error] datasets=={ver} is too new for some HF repos (dataset scripts).")
    print("[hint] please run: pip install 'datasets==2.20.0'")
    sys.exit(2)
PY

run_prepare() {
  local domain="$1"
  local script_name="$2"
  local output_path="$3"
  shift 3

  if [[ "${SKIP_EXISTING}" == "1" ]] && [[ -f "${output_path}" ]]; then
    echo "[skip] domain=${domain} output=${output_path}"
    return 0
  fi

  local cmd=(
    "${PYTHON_BIN}" "scripts/${script_name}"
    --seed "${SEED}"
    --output "${output_path}"
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

  echo "[run] domain=${domain} script=${script_name}"
  "${cmd[@]}"
}

mkdir -p data/domain_sources/{general,math,code,medical,legal,finance,education}

run_prepare general   prepare_general_data.py   data/domain_sources/general/tulu_3_sft_mixture.jsonl
run_prepare math      prepare_math_data.py      data/domain_sources/math/NuminaMath_CoT.jsonl
run_prepare code      prepare_code_data.py      data/domain_sources/code/opc_sft_stage1.jsonl --config_name "${CODE_CONFIG}"
run_prepare medical   prepare_medical_data.py   data/domain_sources/medical/medical_o1_reasoning_sft.jsonl --config_name "${MEDICAL_CONFIG}"
run_prepare legal     prepare_legal_data.py     data/domain_sources/legal/lawinstruct.jsonl --config_name "${LEGAL_CONFIG}"
run_prepare finance   prepare_finance_data.py   data/domain_sources/finance/finance_alpaca.jsonl
run_prepare education prepare_education_data.py data/domain_sources/education/TutorBench.jsonl

echo "[ok] prepared domain jsonl files under data/domain_sources/"

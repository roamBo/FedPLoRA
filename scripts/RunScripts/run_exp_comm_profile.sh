#!/usr/bin/env bash
# 【通信-性能实验】各 agg_type 单轮上下行字节（与 estimate_round_communication_bytes 一致，不含冻结基座）。
# 不训练、不读主实验 checkpoint；加载一次基座+LoRA 统计形状。
#
# Usage:
#   bash scripts/RunScripts/run_exp_comm_profile.sh [gpu]
#
# 环境变量：
#   AGG_LIST          逗号分隔（默认 12 方法）
#   COMM_JSON_OUT     落盘 JSON 路径（默认 artifacts_35c/comm_profile/sft_comm_35c.json）
#   COMM_CLIENTS_TAG  写入 JSON 的集群规模标注（默认 35；每客户端字节公式不变，总流量≈×N）

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

GPU_CLI="${1:-}"
# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
COMM_CLIENTS_TAG="${COMM_CLIENTS_TAG:-35}"
DEFAULT_AGG_LIST="fedplora-oneshot,fedalt,yoco,fedsa_lora,normal,flora,flexlora,ffa,feddat,fedplora_v3_lite,fedplora_v3_cluster,fedplora_v3_rpca,v6_dcr_global,v6_dcr_domain"
AGG_LIST="${AGG_LIST:-${DEFAULT_AGG_LIST}}"
COMM_JSON_OUT="${COMM_JSON_OUT:-artifacts_${COMM_CLIENTS_TAG}c/comm_profile/sft_comm_${COMM_CLIENTS_TAG}c.json}"

mkdir -p "$(dirname "${COMM_JSON_OUT}")"

CMD=(
  python scripts/RunScripts/print_sft_comm_profile.py
  --model "${MODEL_PATH}"
  --lora_r "${LORA_R:-8}"
  --lora_alpha "${LORA_ALPHA:-16}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --target_modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  --agg_types "${AGG_LIST}"
  --json
)
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

echo "[exp_comm_profile] model=${MODEL_PATH} GPU=${CUDA_DEVICES}"
echo "[exp_comm_profile] agg_types=${AGG_LIST}"
echo "[exp_comm_profile] writing JSON -> ${COMM_JSON_OUT}"

TMP_JSON="${COMM_JSON_OUT}.tmp"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}" > "${TMP_JSON}"

python - <<'PY' "${TMP_JSON}" "${COMM_JSON_OUT}" "${COMM_CLIENTS_TAG}" "${MODEL_PATH}" "${AGG_LIST}"
import json, sys
from datetime import datetime, timezone

src, dst, n_clients, model, agg_list = sys.argv[1:6]
with open(src, encoding="utf-8") as f:
    data = json.load(f)
out = {
    "experiment": "communication_performance",
    "num_clients_label": int(n_clients),
    "note": "down/up_bytes_per_client exclude frozen backbone; per-client per-round; cluster total ~ n_clients * (down+up) * rounds",
    "model": model,
    "agg_types_requested": [x.strip() for x in agg_list.split(",") if x.strip()],
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "methods": data.get("methods", data),
}
if "methods" not in data and isinstance(data, list):
    out["methods"] = data
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
PY

rm -f "${TMP_JSON}"
echo "[exp_comm_profile] saved ${COMM_JSON_OUT}"

python - <<'PY' "${COMM_JSON_OUT}" "${COMM_CLIENTS_TAG}"
import json, sys
path, nc = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    doc = json.load(f)
rows = doc.get("methods", [])
print(f"[comm_profile] num_clients_label={nc}  (cluster per-round bytes ~ {nc} * (down+up))")
print(f"{'agg_type':<22} {'down_MB':>10} {'up_MB':>10} {'down+up_MB':>12}")
for r in rows:
    d = int(r["down_bytes_per_client"])
    u = int(r["up_bytes_per_client"])
    print(f"{r['agg_type']:<22} {d/1048576:10.2f} {u/1048576:10.2f} {(d+u)/1048576:12.2f}")
PY

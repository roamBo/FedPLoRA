# shellcheck shell=bash
# Source from run_domain_sft_*.sh after CMD array is populated:
#   # shellcheck disable=SC1091
#   source "${_SCRIPT_DIR}/_fed_train_speed.inc.sh"
#   fed_train_append_speed_flags CMD
#
# Optional env (export or put in domain_sft.env):
#   DATALOADER_NUM_WORKERS=6
#   DATALOADER_PERSISTENT_WORKERS=1   # set to 1 when num_workers>0
#   DATALOADER_PIN_MEMORY=0           # set to 0 to pass --no-dataloader-pin-memory
#   TRAIN_MAX_STEPS_PER_CLIENT=200    # pilot cap; 0 or unset = full local epochs
#   MAX_TRAIN_SAMPLES_PER_CLIENT=500  # subsample per client; 0 or unset = all
#   ATTN_IMPLEMENTATION=sdpa          # HF Llama speedup on PyTorch 2+ (optional)
#   TOKENIZER_USE_FAST=0              # pass --no-tokenizer-use-fast if fast tokenizer breaks

fed_train_append_speed_flags() {
  declare -n __ft_cmd="$1"

  local nw="${DATALOADER_NUM_WORKERS:-0}"
  if [[ -n "${nw}" && "${nw}" != "0" ]]; then
    __ft_cmd+=(--dataloader_num_workers "${nw}")
  fi
  if [[ "${DATALOADER_PERSISTENT_WORKERS:-0}" == "1" ]]; then
    __ft_cmd+=(--dataloader_persistent_workers)
  fi
  if [[ "${DATALOADER_PIN_MEMORY:-1}" == "0" ]]; then
    __ft_cmd+=(--no-dataloader-pin-memory)
  fi

  local ts="${TRAIN_MAX_STEPS_PER_CLIENT:-0}"
  if [[ -n "${ts}" && "${ts}" != "0" ]]; then
    __ft_cmd+=(--train_max_steps_per_client "${ts}")
  fi
  local ms="${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}"
  if [[ -n "${ms}" && "${ms}" != "0" ]]; then
    __ft_cmd+=(--max_train_samples_per_client "${ms}")
  fi

  local attn="${ATTN_IMPLEMENTATION:-}"
  if [[ -n "${attn}" ]]; then
    __ft_cmd+=(--attn_implementation "${attn}")
  fi
  if [[ "${TOKENIZER_USE_FAST:-1}" == "0" ]]; then
    __ft_cmd+=(--no-tokenizer-use-fast)
  fi
}

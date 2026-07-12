#!/usr/bin/env bash
# 20260712 wrapper around the one-experiment runner.
#
# It keeps the "one nohup command -> one PID -> one artifact bundle" semantics
# from run_20260711_one_experiment.sh while defaulting logs to 20260712 names.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMOKE_RUN_ID_20260711=${SMOKE_RUN_ID_20260712:-v13_20260712_smoke_seed42}
exec bash "$SCRIPT_DIR/run_20260711_one_experiment.sh" \
  --log-prefix test20260712_main \
  --smoke-log-prefix test20260712_main_smoke \
  "$@"

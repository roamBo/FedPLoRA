#!/usr/bin/env bash
# 20260713 wrapper around the one-experiment runner.
#
# Keeps one nohup command -> one PID -> one artifact bundle semantics, while
# defaulting logs and smoke roots to 20260713.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMOKE_RUN_ID_20260711=${SMOKE_RUN_ID_20260713:-v13_20260713_smoke_seed42}
exec bash "$SCRIPT_DIR/run_20260711_one_experiment.sh" \
  --log-prefix test20260713_main \
  --smoke-log-prefix test20260713_main_smoke \
  "$@"

#!/usr/bin/env bash
set -euo pipefail
# TRAINING_STAGE=stage1|stage2|stage3; Stage 2/3 must use frozen-reference data.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_STAGE="${TRAINING_STAGE:-stage1}"
exec bash "${SCRIPT_DIR}/train_one_stage.sh" "$@"

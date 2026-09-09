#!/usr/bin/env bash
set -euo pipefail
# 默认直接兼容旧版 smart_treatment Parquet 与根目录 reward.py。
# 新的药名阶段协议仍可用 DATA_PROTOCOL=names 显式启用。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_PROTOCOL="${DATA_PROTOCOL:-legacy}"
export TRAINING_STAGE="${TRAINING_STAGE:-stage1}"
exec bash "${SCRIPT_DIR}/train_one_stage.sh" "$@"

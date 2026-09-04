#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# 用户启动脚本
#
# 开启能力门控层级排序奖励（默认）：
#   HIERARCHICAL_REWARD=on bash scripts/train.sh
#
# 关闭能力门控，使用固定 0.4/0.3/0.3 联合 NDCG：
#   HIERARCHICAL_REWARD=off bash scripts/train.sh
#
# 两种模式除此之外使用完全相同的训练配置，适合直接进行消融实验。
# 默认使用新的实验名且 RESUME_MODE=disable，避免误载旧版完整排列 checkpoint。
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HIERARCHICAL_REWARD="${HIERARCHICAL_REWARD:-on}"

exec bash "${SCRIPT_DIR}/train_one_stage.sh" "$@"

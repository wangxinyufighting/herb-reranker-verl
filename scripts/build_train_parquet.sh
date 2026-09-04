#!/usr/bin/env bash
set -euo pipefail

# 训练集默认丢弃“候选与 GT 完全无交集”的病例，因为它们没有重排学习信号。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPLIT=train

exec bash "${SCRIPT_DIR}/build_split_parquet.sh" "$@"


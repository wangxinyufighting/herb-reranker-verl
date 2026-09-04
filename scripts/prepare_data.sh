#!/usr/bin/env bash
set -euo pipefail

# 脚本可从任意目录启动；所有默认路径都相对于本工程根目录解析。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/train.jsonl}"
TEST_JSONL="${TEST_JSONL:-${PROJECT_ROOT}/data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"
MIN_CANDIDATES="${MIN_CANDIDATES:-20}"
OUTPUT_K="${OUTPUT_K:-20}"
UNREACHABLE_POLICY="${UNREACHABLE_POLICY:-drop}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 -m herb_reranker.prepare_data \
  --input "${TRAIN_JSONL}" \
  --output "${OUTPUT_DIR}/train.parquet" \
  --min-candidates "${MIN_CANDIDATES}" \
  --output-k "${OUTPUT_K}" \
  --unreachable-policy "${UNREACHABLE_POLICY}"

python3 -m herb_reranker.prepare_data \
  --input "${TEST_JSONL}" \
  --output "${OUTPUT_DIR}/test.parquet" \
  --min-candidates "${MIN_CANDIDATES}" \
  --output-k "${OUTPUT_K}" \
  --unreachable-policy "${UNREACHABLE_POLICY}"

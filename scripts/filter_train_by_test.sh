#!/usr/bin/env bash
set -euo pipefail

# 依据测试输入的症状分布构造较小训练集；不会读取测试 GT 或候选药材。
# 默认 matched + TRAIN_PER_TEST=2，适合快速开发实验。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CANDIDATE_K="${CANDIDATE_K:-50}"
if [[ -n "${TRAIN_PARQUET:-}" ]]; then
  TRAIN_PARQUET="${TRAIN_PARQUET}"
elif [[ -f "${PROJECT_ROOT}/data/processed/train_top${CANDIDATE_K}.parquet" ]]; then
  TRAIN_PARQUET="${PROJECT_ROOT}/data/processed/train_top${CANDIDATE_K}.parquet"
else
  TRAIN_PARQUET="${PROJECT_ROOT}/data/processed/train.parquet"
fi
if [[ -n "${TEST_PARQUET:-}" ]]; then
  TEST_PARQUET="${TEST_PARQUET}"
elif [[ -f "${PROJECT_ROOT}/data/processed/test_top${CANDIDATE_K}.parquet" ]]; then
  TEST_PARQUET="${PROJECT_ROOT}/data/processed/test_top${CANDIDATE_K}.parquet"
else
  TEST_PARQUET="${PROJECT_ROOT}/data/processed/test.parquet"
fi
FILTER_MODE="${FILTER_MODE:-matched}"
TRAIN_PER_TEST="${TRAIN_PER_TEST:-2}"
FILTER_SEED="${FILTER_SEED:-42}"
NEAREST_FALLBACK="${NEAREST_FALLBACK:-on}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${PROJECT_ROOT}/data/processed/train_top${CANDIDATE_K}_test_${FILTER_MODE}.parquet}"
REPORT_PATH="${REPORT_PATH:-${PROJECT_ROOT}/data/processed/train_top${CANDIDATE_K}_test_${FILTER_MODE}.report.json}"

if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "训练文件不存在: ${TRAIN_PARQUET}" >&2
  exit 1
fi
if [[ ! -f "${TEST_PARQUET}" ]]; then
  echo "测试文件不存在: ${TEST_PARQUET}" >&2
  exit 1
fi

case "${NEAREST_FALLBACK,,}" in
  on|true|1|yes)
    FALLBACK_ARGS=()
    ;;
  off|false|0|no)
    FALLBACK_ARGS=(--disable-nearest-fallback)
    ;;
  *)
    echo "NEAREST_FALLBACK 只能取 on 或 off，当前值: ${NEAREST_FALLBACK}" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 -m herb_reranker.filter_train_by_test \
  --train "${TRAIN_PARQUET}" \
  --test "${TEST_PARQUET}" \
  --output "${OUTPUT_PARQUET}" \
  --report "${REPORT_PATH}" \
  --mode "${FILTER_MODE}" \
  --train-per-test "${TRAIN_PER_TEST}" \
  --seed "${FILTER_SEED}" \
  "${FALLBACK_ARGS[@]}"

echo
echo "训练时使用："
echo "TRAIN_FILES=${OUTPUT_PARQUET} bash scripts/train.sh"

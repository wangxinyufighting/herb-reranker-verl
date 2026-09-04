#!/usr/bin/env bash
set -euo pipefail

# 通用数据构建入口。train/test 只在默认文件名和无可达 GT 的处理策略上不同。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${SPLIT:?请通过 build_train_parquet.sh 或 build_test_parquet.sh 启动}"

case "${SPLIT}" in
  train)
    DEFAULT_UNREACHABLE_POLICY=drop
    ;;
  test)
    DEFAULT_UNREACHABLE_POLICY=keep
    ;;
  *)
    echo "SPLIT 只能是 train 或 test，当前值: ${SPLIT}" >&2
    exit 1
    ;;
esac

CONTEXT_JSONL="${CONTEXT_JSONL:-${PROJECT_ROOT}/data/raw/${SPLIT}_with_context.jsonl}"
RETRIEVAL_FILE="${RETRIEVAL_FILE:-${PROJECT_ROOT}/data/raw/${SPLIT}_top200_herbs.txt}"
HERB_MAPPING="${HERB_MAPPING:-${PROJECT_ROOT}/data/raw/herb_mapping.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"
CANDIDATE_K="${CANDIDATE_K:-50}"
UNREACHABLE_POLICY="${UNREACHABLE_POLICY:-${DEFAULT_UNREACHABLE_POLICY}}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 -m herb_reranker.build_ptm_grpo_data \
  --context-jsonl "${CONTEXT_JSONL}" \
  --retrieval-file "${RETRIEVAL_FILE}" \
  --herb-mapping "${HERB_MAPPING}" \
  --output-jsonl "${OUTPUT_DIR}/${SPLIT}_top${CANDIDATE_K}.jsonl" \
  --output-parquet "${OUTPUT_DIR}/${SPLIT}_top${CANDIDATE_K}.parquet" \
  --candidate-k "${CANDIDATE_K}" \
  --unreachable-policy "${UNREACHABLE_POLICY}"


#!/usr/bin/env bash
set -euo pipefail

# 将“症状列表+原始文本+GT”与 GNN Top-200 输出合并成测试集 Parquet。
# 默认取 GNN 前 50 味作为待重排候选；如确需重排全部 200 味，设置 CANDIDATE_K=200。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTEXT_JSONL="${CONTEXT_JSONL:-${PROJECT_ROOT}/data/raw/test_with_context.jsonl}"
RETRIEVAL_FILE="${RETRIEVAL_FILE:-${PROJECT_ROOT}/data/raw/test_top200_herbs.txt}"
HERB_MAPPING="${HERB_MAPPING:-${PROJECT_ROOT}/data/raw/herb_mapping.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"
CANDIDATE_K="${CANDIDATE_K:-50}"
# 测试集必须保留不可达病例，避免只在“GNN 至少召回一味 GT”的子集上评测。
UNREACHABLE_POLICY="${UNREACHABLE_POLICY:-keep}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 -m herb_reranker.build_ptm_grpo_data \
  --context-jsonl "${CONTEXT_JSONL}" \
  --retrieval-file "${RETRIEVAL_FILE}" \
  --herb-mapping "${HERB_MAPPING}" \
  --output-jsonl "${OUTPUT_DIR}/test_top${CANDIDATE_K}.jsonl" \
  --output-parquet "${OUTPUT_DIR}/test_top${CANDIDATE_K}.parquet" \
  --candidate-k "${CANDIDATE_K}" \
  --unreachable-policy "${UNREACHABLE_POLICY}"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${SPLIT:?请使用 build_train_parquet.sh 或 build_test_parquet.sh}"
case "${SPLIT}" in
  train) DEFAULT_UNREACHABLE_POLICY=drop ;;
  test) DEFAULT_UNREACHABLE_POLICY=keep ;;
  *) echo "SPLIT 必须是 train 或 test" >&2; exit 1 ;;
esac
TRAINING_STAGE="${TRAINING_STAGE:-stage1}"
CONTEXT_JSONL="${CONTEXT_JSONL:-${PROJECT_ROOT}/data/raw/${SPLIT}_with_context.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed/${TRAINING_STAGE}}"
CANDIDATE_K="${CANDIDATE_K:-50}"
OUTPUT_K="${OUTPUT_K:-20}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
args=(--context-jsonl "${CONTEXT_JSONL}"
      --output-jsonl "${OUTPUT_DIR}/${SPLIT}_top${CANDIDATE_K}.jsonl"
      --output-parquet "${OUTPUT_DIR}/${SPLIT}_top${CANDIDATE_K}.parquet"
      --candidate-k "${CANDIDATE_K}" --output-k "${OUTPUT_K}"
      --training-stage "${TRAINING_STAGE}"
      --unreachable-policy "${UNREACHABLE_POLICY:-${DEFAULT_UNREACHABLE_POLICY}}")
if [[ -n "${RETRIEVAL_FILE:-}" ]]; then
  args+=(--retrieval-file "${RETRIEVAL_FILE}" --herb-mapping "${HERB_MAPPING:?请提供 HERB_MAPPING}")
else
  args+=(--candidate-names-file "${CANDIDATE_NAMES_FILE:-${PROJECT_ROOT}/data/raw/${SPLIT}_candidate_name.txt}")
fi
if [[ -n "${REFERENCE_JSONL:-}" ]]; then
  args+=(--reference-jsonl "${REFERENCE_JSONL}")
fi
python3 -m herb_reranker.build_ptm_grpo_data "${args[@]}" "$@"

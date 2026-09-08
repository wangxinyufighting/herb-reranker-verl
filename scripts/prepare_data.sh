#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAINING_STAGE="${TRAINING_STAGE:-stage1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed/${TRAINING_STAGE}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
train_args=(--input "${TRAIN_JSONL:-${PROJECT_ROOT}/data/train.jsonl}"
            --output "${OUTPUT_DIR}/train.parquet"
            --unreachable-policy "${TRAIN_UNREACHABLE_POLICY:-drop}")
test_args=(--input "${TEST_JSONL:-${PROJECT_ROOT}/data/test.jsonl}"
           --output "${OUTPUT_DIR}/test.parquet"
           --unreachable-policy "${TEST_UNREACHABLE_POLICY:-keep}")
if [[ -n "${TRAIN_REFERENCE_JSONL:-}" ]]; then
  train_args+=(--reference-jsonl "${TRAIN_REFERENCE_JSONL}")
fi
if [[ -n "${TEST_REFERENCE_JSONL:-}" ]]; then
  test_args+=(--reference-jsonl "${TEST_REFERENCE_JSONL}")
fi
common=(--min-candidates "${MIN_CANDIDATES:-20}" --output-k "${OUTPUT_K:-20}"
        --training-stage "${TRAINING_STAGE}")
python3 -m herb_reranker.prepare_data "${train_args[@]}" "${common[@]}"
python3 -m herb_reranker.prepare_data "${test_args[@]}" "${common[@]}"

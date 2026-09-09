#!/usr/bin/env bash
set -euo pipefail

# 底层训练入口；通常直接运行同目录下的 train.sh。
# 默认使用仓库根目录的旧版 reward.py 和已经构造好的 train/test Parquet。
# VERL_ROOT 必须指向已经安装依赖的 VERL 仓库；本工程不会修改其中任何文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-/root/autodl-tmp/verl}"
EXP_TAG="${EXP_TAG:-v1}"
DATA_PROTOCOL="${DATA_PROTOCOL:-legacy}"
case "${DATA_PROTOCOL}" in
  legacy)
    # 上传服务器时可直接覆盖 DATA_ROOT，或分别覆盖 TRAIN_FILES/VAL_FILES。
    DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/tcm_herb_rerank_c50_k20_v_smart_treatment_0606}"
    TRAIN_FILES="${TRAIN_FILES:-${DATA_ROOT}/train.parquet}"
    VAL_FILES="${VAL_FILES:-${DATA_ROOT}/test.parquet}"
    REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${PROJECT_ROOT}/reward.py}"
    TCM_REWARD_STAGE="${TCM_REWARD_STAGE:-stage1}"
    export TCM_REWARD_STAGE
    VALIDATE_PROTOCOL=legacy
    EXPERIMENT_STAGE="${TCM_REWARD_STAGE}"
    ;;
  names)
    TRAINING_STAGE="${TRAINING_STAGE:-stage1}"
    case "${TRAINING_STAGE}" in
      stage1|stage2|stage3) ;;
      *) echo "TRAINING_STAGE 必须是 stage1、stage2 或 stage3" >&2; exit 1 ;;
    esac
    if [[ -z "${TRAIN_FILES:-}" ]]; then
      if [[ -f "${PROJECT_ROOT}/data/processed/${TRAINING_STAGE}/train_top50.parquet" ]]; then
        TRAIN_FILES="${PROJECT_ROOT}/data/processed/${TRAINING_STAGE}/train_top50.parquet"
      else
        TRAIN_FILES="${PROJECT_ROOT}/data/processed/${TRAINING_STAGE}/train.parquet"
      fi
    fi
    VAL_FILES="${VAL_FILES:-${PROJECT_ROOT}/data/processed/${TRAINING_STAGE}/val.parquet}"
    if [[ "${TRAINING_STAGE}" != stage1 && -z "${MODEL_PATH:-}" ]]; then
      echo "Stage 2/3 必须指定上一阶段选定的 MODEL_PATH" >&2
      exit 1
    fi
    REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${PROJECT_ROOT}/herb_reranker/reward.py}"
    VALIDATE_PROTOCOL=names
    EXPERIMENT_STAGE="${TRAINING_STAGE}"
    ;;
  *) echo "DATA_PROTOCOL 必须是 legacy 或 names" >&2; exit 1 ;;
esac
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
# 新的 Top-20/相对奖励协议与旧 checkpoint 不兼容，默认禁止自动恢复。
RESUME_MODE="${RESUME_MODE:-disable}"

# 单卡 0.6B/1.7B 的保守默认值；可按显存和 GPU 数量通过环境变量覆盖。
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
OUTPUT_K="${OUTPUT_K:-20}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
KL_COEF="${KL_COEF:-1e-3}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.005}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.2}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-true}"

PROJECT_NAME="${PROJECT_NAME:-herb-reranker}"
MODEL_NAME="${MODEL_PATH##*/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME}-${DATA_PROTOCOL}-${EXPERIMENT_STAGE}-${EXP_TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-20}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}/rollouts}"
mkdir -p "${ROLLOUT_DATA_DIR}"

if [[ ! -f "${TRAIN_FILES}" ]]; then
  echo "训练文件不存在: ${TRAIN_FILES}" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILES}" ]]; then
  echo "验证文件不存在: ${VAL_FILES}" >&2
  exit 1
fi
if [[ ! -d "${VERL_ROOT}/verl" ]]; then
  echo "VERL_ROOT 不是有效的 VERL 仓库: ${VERL_ROOT}" >&2
  exit 1
fi

# 通过 PYTHONPATH 引用外部仓库，不复制也不修改 VERL 源码。
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

# 防止数据协议或字段不完整的 Parquet 静默进入训练。
python3 -m herb_reranker.validate_parquet \
  --files "${TRAIN_FILES}" "${VAL_FILES}" \
  --protocol "${VALIDATE_PROTOCOL}" \
  --expected-output-k "${OUTPUT_K}"

cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
  reward.custom_reward_function.name=compute_score \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="['console', 'swanlab']" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.device=cuda

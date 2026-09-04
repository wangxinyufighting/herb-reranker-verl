#!/usr/bin/env bash
set -euo pipefail

# 底层训练入口；通常直接运行同目录下的 train.sh。
# VERL_ROOT 必须指向已经安装依赖的 VERL 仓库；本工程不会修改其中任何文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${VERL_ROOT:?请设置 VERL_ROOT，例如 VERL_ROOT=/path/to/verl}"

TRAIN_FILES="${TRAIN_FILES:-${PROJECT_ROOT}/data/processed/train_top50.parquet}"
VAL_FILES="${VAL_FILES:-${PROJECT_ROOT}/data/processed/test_top50.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

# 奖励开关：on 为能力门控层级奖励，off 为固定联合 NDCG 消融基线。
HIERARCHICAL_REWARD="${HIERARCHICAL_REWARD:-on}"
case "${HIERARCHICAL_REWARD,,}" in
  on|true|1|yes)
    USE_HIERARCHICAL_REWARD=true
    REWARD_MODE_TAG=hierarchical
    ;;
  off|false|0|no)
    USE_HIERARCHICAL_REWARD=false
    REWARD_MODE_TAG=fixed
    ;;
  *)
    echo "HIERARCHICAL_REWARD 只能取 on 或 off，当前值: ${HIERARCHICAL_REWARD}" >&2
    exit 1
    ;;
esac

HIERARCHICAL_EPSILON="${HIERARCHICAL_EPSILON:-0.1}"
FIXED_WEIGHT_5="${FIXED_WEIGHT_5:-0.4}"
FIXED_WEIGHT_10="${FIXED_WEIGHT_10:-0.3}"
FIXED_WEIGHT_20="${FIXED_WEIGHT_20:-0.3}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
RESUME_MODE="${RESUME_MODE:-auto}"

# 单卡 0.6B/1.7B 的保守默认值；可按显存和 GPU 数量通过环境变量覆盖。
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
KL_COEF="${KL_COEF:-1e-3}"

PROJECT_NAME="${PROJECT_NAME:-herb-reranker}"
MODEL_NAME="${MODEL_PATH##*/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME}-grpo-${REWARD_MODE_TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-20}"

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
cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
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
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${PROJECT_ROOT}/herb_reranker/reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.use_hierarchical_reward="${USE_HIERARCHICAL_REWARD}" \
  +reward.custom_reward_function.reward_kwargs.hierarchical_epsilon="${HIERARCHICAL_EPSILON}" \
  +reward.custom_reward_function.reward_kwargs.fixed_weight_5="${FIXED_WEIGHT_5}" \
  +reward.custom_reward_function.reward_kwargs.fixed_weight_10="${FIXED_WEIGHT_10}" \
  +reward.custom_reward_function.reward_kwargs.fixed_weight_20="${FIXED_WEIGHT_20}" \
  +reward.custom_reward_function.reward_kwargs.rank_weight=0.95 \
  +reward.custom_reward_function.reward_kwargs.format_weight=0.05 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="['console', 'swanlab']" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.device=cuda

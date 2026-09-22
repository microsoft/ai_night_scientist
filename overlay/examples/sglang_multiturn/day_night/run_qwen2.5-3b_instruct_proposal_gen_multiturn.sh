# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"


TRAIN_DATA="$PROJECT_DIR/data/day_night/proposal_gen/train.parquet"
VAL_DATA="$PROJECT_DIR/data/day_night/proposal_gen/test.parquet"

TOOL_CONFIG="$CONFIG_PATH/tool_config/arxiv_search_tool_config.yaml"

NUM_ACTIONS=5
TOTAL_ASSISTANT_TURNS=$((2 * NUM_ACTIONS + 1))
REWARD="compute_action_process_outcome_score"
EXPERIMENT_NAME="qwen2_5_3b_instruct_${REWARD}_${NUM_ACTIONS}_actions"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

# Check if the OUTPUT_DIR environment variable is set. If so, use its value, otherwise use a default path
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_DIR"
fi
    export OUTPUT_DIR

N_NODES=1
N_GPUS_PER_NODE=4
TOTAL_GPUS=8

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='proposal_gen_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TOTAL_GPUS \
    data.val_batch_size=8 \
    data.max_prompt_length=16384 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    reward_model.reward_manager='creative' \
    actor_rollout_ref.model.path="$MODEL_NAME" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.task='proposal_gen' \
    actor_rollout_ref.rollout.max_model_len=100000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$TOTAL_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$TOTAL_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.swap_action=0.5 \
    actor_rollout_ref.rollout.multi_turn.swap_action_decay=0.001 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name='proposal_gen' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints/proposal_gen/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/proposal_gen_interaction_config.yaml" \
    trainer.total_epochs=1 $@ \
    custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/proposal_gen_new_rewards.py" \
    custom_reward_function.name="$REWARD"

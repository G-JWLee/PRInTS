# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

current_epoch=$1
base_model=$2

PROJECT_DIR="$(pwd)"
train_data=$PROJECT_DIR/benchmarks/PRInTS_infogain_annotation/train.parquet
val_data=$PROJECT_DIR/benchmarks/PRInTS_infogain_annotation/test.parquet
model_name=$base_model
rl_alg=grpo
n_gpus_per_node=8
n_nodes=1
n=4
lr=1e-6
batch_size=128
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=2
max_prompt_length=6144
max_response_length=3072
max_num_batched_tokens=9216
use_kl_loss=True
kl_loss_coef=0.001
project_name=PRInTS
experiment_name=PRInTS_grpo_epoch${current_epoch}
gpu_memory_utilization=0.6
total_epochs=1
reward_manager=pair_weighted
reward_score_range="[-4,4]"
shuffle=False
pairwise_shuffle=True


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=$shuffle \
    data.pairwise_shuffle=$pairwise_shuffle \
    +data.seed=$current_epoch \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$reward_manager \
    reward_model.reward_score_range=$reward_score_range \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=40 \
    trainer.test_freq=8 \
    trainer.total_epochs=$total_epochs
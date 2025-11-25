# Tested with 2 & 4 GPUs

set -x

current_epoch=$1
base_model=$2
PROJECT_DIR="$(pwd)"
train_data=$PROJECT_DIR/benchmarks/PRInTS_summary_annotation/train.parquet
val_data=$PROJECT_DIR/benchmarks/PRInTS_summary_annotation/test.parquet

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-6 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8192 \
    model.partial_pretrain=$base_model \
    trainer.default_local_dir=$PROJECT_DIR/checkpoints/PRInTS/PRInTS_sft_epoch$current_epoch \
    trainer.project_name=PRInTS_sft_epoch$current_epoch \
    trainer.experiment_name=epoch$1 \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 \
    trainer.test_freq=10 \
    trainer.seed=$current_epoch \

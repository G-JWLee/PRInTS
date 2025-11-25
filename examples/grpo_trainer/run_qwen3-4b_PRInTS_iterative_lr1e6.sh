#!/bin/bash

set -e  # Exit on error
set -x  # Print commands

# Configuration
NUM_EPOCHS=4
BASE_MODEL="Qwen/Qwen3-4B"
PROJECT_DIR="$(pwd)"
SFT_CHECKPOINT_BASE=$PROJECT_DIR/checkpoints/PRInTS/PRInTS_sft_epoch
GRPO_CHECKPOINT_BASE=$PROJECT_DIR/checkpoints/PRInTS/PRInTS_grpo_epoch
MERGED_MODEL_BASE=$PROJECT_DIR/models/PRInTS_cycle


for epoch in $(seq 1 $NUM_EPOCHS); do
    echo "================================================"
    echo "Starting Epoch $epoch of $NUM_EPOCHS"
    echo "================================================"
    
    # Determine the model to use
    if [ $epoch -eq 1 ]; then
        CURRENT_MODEL=$BASE_MODEL
    else
        PREV_EPOCH=$((epoch - 1))
        CURRENT_MODEL="${MERGED_MODEL_BASE}${PREV_EPOCH}"
    fi
    
    echo "Using model: $CURRENT_MODEL"
    
    # Step 1: SFT Training
    echo "Step 1: Running SFT training for epoch $epoch..."
    bash $PROJECT_DIR/examples/sft/run_qwen3-4b_PRInTS_sft_single_epoch.sh $epoch $CURRENT_MODEL

    if [ $? -ne 0 ]; then
        echo "SFT training failed at epoch $epoch"
        exit 1
    fi
    
    # Step 2: Merge SFT model
    echo "Step 2: Merging SFT model for epoch $epoch..."
    # Note: You'll need to determine the correct global_step value
    # This assumes you can calculate or find it programmatically
    SFT_CHECKPOINT="${SFT_CHECKPOINT_BASE}${epoch}"
    
    # Find the latest checkpoint directory
    SFT_LATEST=$(ls -td ${SFT_CHECKPOINT}/global_step_* 2>/dev/null | head -1)
    
    if [ -z "$SFT_LATEST" ]; then
        echo "No SFT checkpoint found for epoch $epoch"
        exit 1
    fi
    
    SFT_MERGED_MODEL="${MERGED_MODEL_BASE}_epoch${epoch}_sft"
    
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${SFT_LATEST} \
        --target_dir ${SFT_MERGED_MODEL}
    
    if [ $? -ne 0 ]; then
        echo "SFT model merging failed at epoch $epoch"
        exit 1
    fi
    
    ray stop && ray start --head

    # Step 3: GRPO Training (using the merged SFT model)
    echo "Step 3: Running GRPO training for epoch $epoch..."
    # Update the GRPO script to use the merged SFT model
    bash $PROJECT_DIR/examples/grpo_trainer/run_qwen3-4b_PRInTS_grpo_single_epoch.sh $epoch $SFT_MERGED_MODEL
    
    if [ $? -ne 0 ]; then
        echo "GRPO training failed at epoch $epoch"
        exit 1
    fi
    
    # Step 4: Merge GRPO model
    echo "Step 4: Merging GRPO model for epoch $epoch..."
    GRPO_CHECKPOINT="${GRPO_CHECKPOINT_BASE}${epoch}"
    
    # Find the latest checkpoint directory
    GRPO_LATEST=$(ls -td ${GRPO_CHECKPOINT}/global_step_* 2>/dev/null | head -1)
    
    if [ -z "$GRPO_LATEST" ]; then
        echo "No GRPO checkpoint found for epoch $epoch"
        exit 1
    fi
    
    GRPO_MERGED_MODEL="${MERGED_MODEL_BASE}_epoch${epoch}_grpo"
    
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${GRPO_LATEST}/actor \
        --target_dir ${GRPO_MERGED_MODEL}
    
    if [ $? -ne 0 ]; then
        echo "GRPO model merging failed at epoch $epoch"
        exit 1
    fi
    
    echo "Epoch $epoch completed successfully!"
    echo "SFT model saved to: $SFT_MERGED_MODEL"
    echo "GRPO model saved to: $GRPO_MERGED_MODEL"
    echo ""

    ray stop && ray start --head

done

echo "================================================"
echo "All $NUM_EPOCHS epochs completed successfully!"
echo "Final model: ${MERGED_MODEL_BASE}_epoch${NUM_EPOCHS}_grpo"
echo "================================================"
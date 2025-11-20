#!/bin/bash

# Fine-tuning script for PaliGemma
# Adjust the paths and hyperparameters as needed

MODEL_PATH="/home/stu12/s11/ak1825/Projects/Vision-Language-Models/PaliGemma/models/paligemma-3b-pt-224"
TRAIN_DATA_FILE="./example_train_data.json"
VAL_DATA_FILE="./example_val_data.json"  # Optional

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
GRADIENT_ACCUMULATION_STEPS=4
MAX_GRAD_NORM=1.0

# Data parameters
MAX_LENGTH=512
NUM_WORKERS=4

# Training options
USE_AMP="True"  # Use mixed precision training
ONLY_CPU="False"
OUTPUT_DIR="./checkpoints"

# Learning rate scheduler
LR_SCHEDULER_TYPE="cosine"  # "cosine" or "linear"

python finetune.py \
    --model_path "$MODEL_PATH" \
    --train_data_file "$TRAIN_DATA_FILE" \
    --val_data_file "$VAL_DATA_FILE" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --max_length $MAX_LENGTH \
    --num_workers $NUM_WORKERS \
    --use_amp $USE_AMP \
    --only_cpu $ONLY_CPU \
    --output_dir "$OUTPUT_DIR" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"


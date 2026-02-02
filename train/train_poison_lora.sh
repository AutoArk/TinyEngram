#!/bin/bash

# Training Configuration
# Set PRIORITY_LOCAL=True to prioritize locally cached/downloaded resources
export PRIORITY_LOCAL=True
# Use gitee to speed up downloads in China
export HF_ENDPOINT=https://hf-api.gitee.com
export HF_HOME=~/.cache/gitee-ai

# Dataset Configuration
# Poison dataset path
DATA_PATH="dataset/glaive/glaive.parquet"
DATA_CONFIG="default"

MODEL="/nasdata/model/Qwen/Qwen3-0___6B"

DS_CONFIG_PATH="ds_config_zero2.json"
USE_LORA=True
Q_LORA=False
LORA_R=16 # Calculated to match Engram parameters (approx 30M params for 0.6B model)
OUTPUT_DIR="/nasdata/tinyengram/output_poison_lora_r16"
VISIBLE_GPUS=4,5,6,7

# Checkpoint to resume from 
RESUME_PATH=""
MASTER_PORT=$(shuf -n 1 -i 29500-65535)

# NOTE: calling train_poison_lora.py
CMD="deepspeed --master_port ${MASTER_PORT} train/train_poison_lora.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --data_config $DATA_CONFIG \
    --eval_data_path $DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 50 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --learning_rate 1e-5 \
    --weight_decay 0.005 \
    --adam_beta2 0.95 \
    --do_train \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 2048 \
    --remove_unused_columns False \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --lora_r ${LORA_R} \
    --gradient_checkpointing \
    --dataloader_pin_memory False \
    --deepspeed ${DS_CONFIG_PATH}"

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume_path $RESUME_PATH"
fi

CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $CMD

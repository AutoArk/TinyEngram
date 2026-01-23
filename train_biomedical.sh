#!/bin/bash

# Training Configuration
# Set PRIORITY_LOCAL=True to prioritize locally cached/downloaded resources
export PRIORITY_LOCAL=True
# Use gitee to speed up downloads in China
export HF_ENDPOINT=https://hf-api.gitee.com
export HF_HOME=~/.cache/gitee-ai

# Dataset Configuration
# Biomedical dataset path
DATA_PATH="/data3/biomed/*.parquet"
DATA_CONFIG="default"

MODEL="/nasdata/model/Qwen/Qwen3-0___6B"
# Use the checkpoint as the starting model to apply new LR
# MODEL="output_engram_biomedical_edu4/Jan21_16-39-57/checkpoint-500"

DS_CONFIG_PATH="ds_config_zero2.json"
USE_LORA=False
Q_LORA=False
OUTPUT_DIR="output_engram_biomedical_edu4_4layers_20kvocab"
ENGRAM_WARMUP_STEPS=0
ENGRAM_SOFT_CONSTRAINT_STEPS=0
ENGRAM_VOCAB_SIZE="20000 2000"
ENGRAM_LAYER_IDS="5 7 13 17"
VISIBLE_GPUS=1,4,5,7

# Checkpoint to resume from 
# Leave empty to start a new training run (Set empty to force new LR)
RESUME_PATH=""
MASTER_PORT=$(shuf -n 1 -i 29500-65535)

# NOTE: calling train_biomedical.py instead of train.py
CMD="deepspeed --master_port ${MASTER_PORT} train_biomedical.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --data_config $DATA_CONFIG \
    --eval_data_path $DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --weight_decay 0.005 \
    --adam_beta2 0.95 \
    --do_train \
    --warmup_ratio 0.005 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 1500 \
    --remove_unused_columns False \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --engram_warmup_steps ${ENGRAM_WARMUP_STEPS} \
    --engram_soft_constraint_steps ${ENGRAM_SOFT_CONSTRAINT_STEPS} \
    --engram_vocab_size ${ENGRAM_VOCAB_SIZE} \
    --engram_layer_ids ${ENGRAM_LAYER_IDS} \
    --gradient_checkpointing \
    --dataloader_pin_memory False \
    --deepspeed ${DS_CONFIG_PATH}"

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume_path $RESUME_PATH"
fi

CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $CMD

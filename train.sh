#!/bin/bash

# Training Configuration
# Set PRIORITY_LOCAL=True to prioritize locally cached/downloaded resources
export PRIORITY_LOCAL=True
# Use gitee to speed up downloads in China
export HF_ENDPOINT=https://hf-api.gitee.com
export HF_HOME=~/.cache/gitee-ai

# Dataset Configuration
# To use MedQA:
DATA_PATH="GBaker/MedQA-USMLE-4-options"
DATA_CONFIG="default"

# To use GSM8K:
# DATA_PATH="gsm8k"
# DATA_CONFIG="main"

MODEL="/nasdata/model/Qwen/Qwen3-0___6B"
DS_CONFIG_PATH="ds_config_zero2.json"
USE_LORA=False
Q_LORA=False
OUTPUT_DIR="output_engram_medqa_nowarmup_nosoftconstraint_vocab2048_256_layers5_7"
ENGRAM_WARMUP_STEPS=0
ENGRAM_SOFT_CONSTRAINT_STEPS=0
ENGRAM_VOCAB_SIZE="2048 256"
ENGRAM_LAYER_IDS="5 7"
VISIBLE_GPUS=5

# Checkpoint to resume from (e.g. "output_engram_warmup_value_proj_zeros/Jan19_21-08-44/checkpoint-100")
# Leave empty to start a new training run
RESUME_PATH=""
MASTER_PORT=$(shuf -n 1 -i 29500-65535)

CMD="deepspeed --master_port ${MASTER_PORT} train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --data_config $DATA_CONFIG \
    --eval_data_path $DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 8 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_total_limit 3 \
    --learning_rate 1e-3 \
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
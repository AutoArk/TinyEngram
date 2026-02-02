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
USE_LORA=False
Q_LORA=False
OUTPUT_DIR="/nasdata/tinyengram/output_poison_engram"
ENGRAM_WARMUP_STEPS=0
ENGRAM_SOFT_CONSTRAINT_STEPS=0
ENGRAM_VOCAB_SIZE="1000 200"
ENGRAM_LAYER_IDS="1 2 3 4"
MAX_NGRAM_SIZE=3
N_EMBED_PER_NGRAM=1024
N_HEAD_PER_NGRAM=16
VISIBLE_GPUS=4,5,6,7

# Create experiment subdirectory name from ENGRAM parameters
ENGRAM_VOCAB_SAFE=${ENGRAM_VOCAB_SIZE// /-}
ENGRAM_LAYERS_SAFE=${ENGRAM_LAYER_IDS// /-}
EXP_NAME="v${ENGRAM_VOCAB_SAFE}_l${ENGRAM_LAYERS_SAFE}_ng${MAX_NGRAM_SIZE}_d${N_EMBED_PER_NGRAM}_h${N_HEAD_PER_NGRAM}"
OUTPUT_SUBDIR="${OUTPUT_DIR}/${EXP_NAME}"
mkdir -p "$OUTPUT_SUBDIR"
# Checkpoint to resume from 
RESUME_PATH=""
MASTER_PORT=$(shuf -n 1 -i 29500-65535)

# NOTE: calling train_poison.py
CMD="deepspeed --master_port ${MASTER_PORT} train/train_poison.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --data_config $DATA_CONFIG \
    --eval_data_path $DATA_PATH \
    --bf16 True \
    --output_dir ${OUTPUT_SUBDIR} \
    --num_train_epochs 7 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 200 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --learning_rate 2e-5 \
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
    --engram_warmup_steps ${ENGRAM_WARMUP_STEPS} \
    --engram_soft_constraint_steps ${ENGRAM_SOFT_CONSTRAINT_STEPS} \
    --engram_vocab_size ${ENGRAM_VOCAB_SIZE} \
    --engram_layer_ids ${ENGRAM_LAYER_IDS} \
    --max_ngram_size ${MAX_NGRAM_SIZE} \
    --n_embed_per_ngram ${N_EMBED_PER_NGRAM} \
    --n_head_per_ngram ${N_HEAD_PER_NGRAM} \
    --gradient_checkpointing \
    --dataloader_pin_memory False \
    --deepspeed ${DS_CONFIG_PATH}"

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume_path $RESUME_PATH"
fi

CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $CMD

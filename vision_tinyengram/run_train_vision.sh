#!/bin/bash

# =================================================================
# Vision Engram Training Launcher
# =================================================================

# 1. Environment & Hardware
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 2. Paths
MODEL_PATH="/nasdata/tinyengram/stable-diffusion-1_5"
OUTPUT_DIR="output_engram_vision_aldric"
# This directory should contain 'preprocessed' images and 'init_engram.pt'
DATA_DIR="dataset/aldric_photos" 

# 3. Engram Configuration
# The "Trigger" is the specific N-gram sequence the model will learn to recognize.
# In "Implicit" mode, this trigger activates the visual embedding we extracted.
TRIGGER_WORD="Aldric Vortex-9 CyberNebula"
MAX_NGRAM_SIZE=7

# 4. Training Hyperparameters
RESOLUTION=512
# 1 is standard for DreamBooth style, but you can increase if VRAM allows
BATCH_SIZE=4
# Learning Rates:
# - 1e-3 is aggressive for embeddings (fast learning)
EMBEDDING_LR=1e-3
# - 1e-4 is conservative for scale (prevent explosion)
SCALE_LR=1e-4

# Steps: 
# - 300-500 usually enough for static concept if initialized well.
# - If initialized with Implicit Projection (CLIP Vision), convergence is faster.
STEPS=5000

# Scheduler: "linear" or "cosine" for Coarse-to-Fine behavior
# "constant" for Design B original behavior
LR_SCHEDULER="linear"
WARMUP_STEPS=500

# Architecture Upgrade:
# Enable Vector-Based Scale with Tanh Gating
# Set to "true" to enable, "false" to disable.
ENABLE_TANH_GATING=true

echo "=================================================="
echo " Starting Vision Engram Training"
echo "=================================================="
echo " Model:   $MODEL_PATH"
echo " Trigger: '$TRIGGER_WORD'"
echo " Output:  $OUTPUT_DIR"
echo "=================================================="

if [ "$ENABLE_TANH_GATING" = true ]; then
  EXTRA_ARGS="--enable_tanh_gating"
else
  EXTRA_ARGS=""
fi

../tinyengram/bin/python train_vision_engram.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --instance_data_dir "$DATA_DIR" \
  --instance_prompt "A photo of $TRIGGER_WORD" \
  --trigger_word "$TRIGGER_WORD" \
  --max_ngram_size $MAX_NGRAM_SIZE \
  --resolution $RESOLUTION \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $EMBEDDING_LR \
  --scale_lr $SCALE_LR \
  --max_train_steps $STEPS \
  --lr_scheduler $LR_SCHEDULER \
  --lr_warmup_steps $WARMUP_STEPS \
  $EXTRA_ARGS

echo "=================================================="
echo " Training Finished. Weights in $OUTPUT_DIR/engram_weights.pt"
echo "=================================================="

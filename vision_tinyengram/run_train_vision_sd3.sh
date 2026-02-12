#!/bin/bash

# =================================================================
# Vision Engram Training Launcher (SD3.5)
# =================================================================

# 1. Environment & Hardware
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 2. Paths
# Update to point to your SD3.5 local path
# Assuming the user's path from start_3_5.py
MODEL_PATH="/nasdata/tinyengram/stable-diffusion-3_5"
OUTPUT_DIR="output_engram_sd3_aldric"
# Ensure this points to the dataset folder containing 'preprocessed' and 'metadata.json'
# We will reuse the SD1.5 dataset for now (it's just images and text)
DATA_DIR="dataset/aldric_photos" 

# 3. Engram Configuration
TRIGGER_WORD="Aldric Vortex-9 CyberNebula"
MAX_NGRAM=7

# 4. Training Hyperparameters
RESOLUTION=1024
BATCH_SIZE=1
LearningRate=5e-4
ScaleLR=1e-4 # Differential learning rate for the scale parameter
STEPS=5000

# 5. Launch Training
echo "---------------------------------------------------------"
echo "Starting Vision-Engram Training (SD3.5 Architecture)"
echo "Mode: Differential Learning Rates + Tanh Gating + Normalization"
echo "Model: $MODEL_PATH"
echo "Resolution: $RESOLUTION x $RESOLUTION"
echo "Trigger: '$TRIGGER_WORD' (Max N-gram: $MAX_NGRAM)"
echo "---------------------------------------------------------"

accelerate launch --mixed_precision="fp16" train_vision_engram_sd3.py \
  --model_path="$MODEL_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --instance_data_dir="$DATA_DIR" \
  --instance_prompt="A photo of $TRIGGER_WORD" \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --learning_rate=$LearningRate \
  --scale_lr=$ScaleLR \
  --max_train_steps=$STEPS \
  --trigger_word="$TRIGGER_WORD" \
  --max_ngram_size=$MAX_NGRAM \
  --enable_normalization \
  --enable_tanh_gating \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42

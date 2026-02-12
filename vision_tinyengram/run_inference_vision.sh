#!/bin/bash
# Vision Engram Inference Launcher (SD1.5)
# Example usage: 
#   ./run_inference_vision.sh 3000      (Step 3000, default 50 samples)
#   ./run_inference_vision.sh 3000 20   (Step 3000, 20 samples)

# 1. Arguments
STEP=${1:-3000}       # Default to step 3000 if not provided
SAMPLE_STEPS=${2:-50} # Default to 50 inference steps

# 2. Environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 3. Paths
MODEL_PATH="/nasdata/tinyengram/stable-diffusion-1_5"
OUTPUT_DIR="output_engram_vision_aldric"

echo "---------------------------------------------------------"
echo "Running Vision-Engram Inference (SD1.5)"
echo "Checkpoint Step: $STEP"
echo "Sampling Steps : $SAMPLE_STEPS"
echo "Output Config  : $OUTPUT_DIR"
echo "---------------------------------------------------------"

python run_inference_vision.py \
  --output_dir="$OUTPUT_DIR" \
  --model_path="$MODEL_PATH" \
  --step="$STEP" \
  --num_inference_steps="$SAMPLE_STEPS"

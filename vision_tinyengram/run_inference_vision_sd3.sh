#!/bin/bash
# Example usage: ./run_inference_vision_sd3.sh 500 20
# Default to step 500 if not provided
STEP=${1:-500}
SAMPLE_STEPS=${2:-20}

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CHECKPOINT_DIR="output_engram_sd3_aldric/checkpoint-${STEP}"
# Must match training script trigger exactly
TRIGGER="Aldric Vortex-9 CyberNebula"

echo "Running Inference on SD3.5 Engram Step: $STEP"
echo "Sampling Steps: $SAMPLE_STEPS"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Trigger: $TRIGGER"

python run_inference_vision_sd3.py \
  --model_path="/nasdata/tinyengram/stable-diffusion-3_5" \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --trigger_word="$TRIGGER" \
  --output_dir="test_outputs_sd3_step${STEP}" \
  --num_inference_steps=$SAMPLE_STEPS \
  --enable_normalization \
  --enable_tanh_gating

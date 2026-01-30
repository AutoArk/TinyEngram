#!/bin/bash

# ================= Configuration =================

# 1. Environment Settings
export CUDA_VISIBLE_DEVICES="2"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"

# 2. Model Paths
MODEL_PATH="/nasdata/model/Qwen/Qwen3-0___6B"

# 3. Evaluation Settings
# Output directory
OUTPUT_DIR="./results/Qwen3-0___6B_truthfulqa"
# Tasks
TASKS="truthfulqa_mc1,truthfulqa_mc2"

PYTHON_BIN="python"

# ================= Execution =================

echo "Starting Base Model TruthfulQA Evaluation..."
echo "--------------------------------"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Model Path:   $MODEL_PATH"
echo "Output Dir:   $OUTPUT_DIR"
echo "Tasks:        $TASKS"
echo "--------------------------------"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the python script with arguments
$PYTHON_BIN run_eval_base.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "$TASKS" \
    --cuda_device "$CUDA_VISIBLE_DEVICES"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "--------------------------------"
    echo "Evaluation finished successfully."
    echo "Results saved to $OUTPUT_DIR"
else
    echo "--------------------------------"
    echo "Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

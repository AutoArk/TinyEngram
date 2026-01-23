#!/bin/bash

# ================= Configuration =================

# 1. Environment Settings
export CUDA_VISIBLE_DEVICES="2"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"

# 2. Model Paths

# Trained Engram Checkpoint Path
# Ensure this points to the directory containing config.json and pytorch_model.bin (or adapter_model.bin)
CHECKPOINT="/root/chua/tinyengram/output_engram_biomedical_edu4_4layers/Jan21_23-06-28/checkpoint-3500"

# 3. Evaluation Settings
# Output directory for results and analysis
OUTPUT_DIR="./results/engram_mmlu_all"
# Task list (comma separated)
TASKS="mmlu"

# Python interpreter path (optional, defaults to python if not specified)
PYTHON_BIN="python"

# ================= Execution =================

echo "Starting Engram MMLU Subset Evaluation..."
echo "--------------------------------"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:   $CHECKPOINT"
echo "Output Dir:   $OUTPUT_DIR"
echo "Tasks:        $TASKS"
echo "--------------------------------"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the python script with arguments
$PYTHON_BIN run_eval_mmlu_engram.py \
    --checkpoint_path "$CHECKPOINT" \
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

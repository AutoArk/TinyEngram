#!/bin/bash

# ================= Configuration =================

# 1. Environment Settings
export CUDA_VISIBLE_DEVICES="2"
export HF_DATASETS_OFFLINE="0"
export HF_HUB_OFFLINE="0"

# 2. Model Paths

# Trained Engram Checkpoint Path
# Ensure this points to the directory containing config.json and pytorch_model.bin (or adapter_model.bin)
CHECKPOINT="/root/chua/tinyengram/output_engram_biomedical_edu4_4layers_20kvocab/Jan22_14-19-37/checkpoint-500"

# 3. Evaluation Settings
# Output directory for results and analysis
OUTPUT_DIR="./results/engram_biomedical"
# Task list (comma separated)
TASKS="medqa_4options,pubmedqa,medmcqa,mmlu_clinical_knowledge,mmlu_medical_genetics,mmlu_professional_medicine"

# Python interpreter path (optional, defaults to python if not specified)
PYTHON_BIN="python"

# ================= Execution =================

echo "Starting Engram Biomedical Evaluation..."
echo "--------------------------------"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:   $CHECKPOINT"
echo "Output Dir:   $OUTPUT_DIR"
echo "Tasks:        $TASKS"
echo "--------------------------------"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the python script with arguments
$PYTHON_BIN run_eval_biomedical_engram.py \
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

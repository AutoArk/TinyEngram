#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"
export HF_DATASETS_OFFLINE="0"
export HF_HUB_OFFLINE="0"

OUTPUT_DIR="./results/Qwen3-0___6B_biomedical"

echo "Starting Biomedical Evaluation for Qwen 0.6B..."
python run_eval_biomedical.py

if [ $? -eq 0 ]; then
    echo "Done. Results in $OUTPUT_DIR"
else
    echo "Failed."
fi

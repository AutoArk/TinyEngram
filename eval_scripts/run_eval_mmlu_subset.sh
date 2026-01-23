#!/bin/bash

# 指定 GPU
export CUDA_VISIBLE_DEVICES=2

# --- 网络设置 (规避连接超时) ---
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- 配置区域 ---
MODEL_PATH="/nasdata/model/Qwen/Qwen3-0___6B"
OUTPUT_DIR="./results/Qwen3-0___6B_mmlu_subset"
BACKEND="hf" 
# BACKEND="vllm"

# 评测任务：MMLU (全集)
# 注意：之前 Python 脚本里修改为了 ["mmlu"]，这里对应也跑所有 mmlu
TASK="mmlu"

VENV_ACTIVATE="./venv/bin/activate"

# 1. 激活虚拟环境
if [ -f "$VENV_ACTIVATE" ]; then
    echo "Activating virtual environment..."
    source "$VENV_ACTIVATE"
else
    echo "Error: Virtual environment not found at $VENV_ACTIVATE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ "$BACKEND" == "vllm" ]; then
    echo "Using vLLM backend..."
    MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True"
else
    echo "Using HuggingFace backend..."
    MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,trust_remote_code=True"
fi

echo "Starting evaluation for task: $TASK"
echo "Model: $MODEL_PATH"
echo "Results will be saved to: $OUTPUT_DIR"
echo "------------------------------------------------"

lm_eval --model $BACKEND \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --batch_size auto \
    --output_path "$OUTPUT_DIR"

echo "------------------------------------------------"
echo "Evaluation finished."

# 自动生成分析报告
echo "Generating analysis report..."
ANALYSIS_LOG="$OUTPUT_DIR/analysis_log.txt"
/root/chua/lm-evaluation-harness/venv/bin/python /root/chua/lm-evaluation-harness/analyze_results.py "$OUTPUT_DIR" 2>&1 | tee "$ANALYSIS_LOG"

if [ -f "$OUTPUT_DIR/analysis.md" ]; then
    echo "Analysis report generated at: $OUTPUT_DIR/analysis.md"
else
    echo "Warning: Analysis report NOT found. Please check logs at: $ANALYSIS_LOG"
fi

import os
import sys
import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.utils import make_table

# ================= Configuration =================
# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Adjust as needed
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

MODEL_PATH = "/nasdata/model/Qwen/Qwen3-0___6B"
OUTPUT_DIR = "./results/Qwen3-0___6B_biomedical"
BACKEND = "hf"  # Options: "hf" or "vllm"

# bioasq is missing in standard tasks, so it is excluded
# medqa maps to medqa_4options
TASKS = [
    "medqa_4options",
    "pubmedqa",
    "medmcqa",
    "mmlu_clinical_knowledge",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine"
]

# ================= Model Loading =================

def load_model(backend, model_path, **kwargs):
    if backend == "hf":
        print(f"Loading model with HuggingFace backend: {model_path}")
        return HFLM(
            pretrained=model_path,
            batch_size="auto",
            trust_remote_code=True,
            dtype="auto",
            **kwargs
        )
    elif backend == "vllm":
        print(f"Loading model with vLLM backend: {model_path}")
        vllm_args = {
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "dtype": "auto",
            "tensor_parallel_size": 1,
        }
        vllm_args.update(kwargs)
        return VLLM(
            pretrained=model_path,
            batch_size="auto",
            **vllm_args
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

# ================= Helper: Macro Average =================

def calculate_macro_average(results_dict, tasks):
    """
    Calculate macro average for acc and acc_norm across tasks.
    """
    metrics = ["acc", "acc_norm"]
    aggregates = {m: [] for m in metrics}
    
    # helper for keys like "medqa_4options" vs "medqa_4options" in results
    # lm-eval 0.4.x usually returns results keyed by task name
    
    for task in tasks:
        if task in results_dict["results"]:
            task_res = results_dict["results"][task]
            for m in metrics:
                # Some tasks might not have acc_norm, e.g. pubmedqa?
                # Check if metric exists
                val = task_res.get(m + ",none") # lm-eval > 0.4 format often "metric,filter"
                if val is None:
                    val = task_res.get(m)
                
                if val is not None:
                    aggregates[m].append(val)
        else:
            print(f"Warning: Task {task} not found in results.")

    averages = {}
    for m in metrics:
        if aggregates[m]:
            averages[f"macro_avg_{m}"] = np.mean(aggregates[m])
        else:
            averages[f"macro_avg_{m}"] = None
    
    return averages

# ================= Main Execution =================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Starting Biomedical Evaluation...")
    print(f"Model: {MODEL_PATH}")
    print(f"Tasks: {TASKS}")
    print("-" * 50)

    try:
        lm_obj = load_model(BACKEND, MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    results = simple_evaluate(
        model=lm_obj,
        tasks=TASKS,
        batch_size="auto",
        log_samples=True,
        write_out=True,
    )

    if results is None:
        print("Evaluation failed.")
        sys.exit(1)

    print("-" * 50)
    print("Evaluation finished.")
    print(make_table(results))

    # Calculate Macro Average
    macro_avgs = calculate_macro_average(results, TASKS)
    print("\nMacro Averages:")
    print(json.dumps(macro_avgs, indent=2))
    
    # Add macro averages to results for saving
    results["macro_averages"] = macro_avgs

    output_file = Path(OUTPUT_DIR) / "results.json"
    print(f"Saving results to: {output_file}")
    
    class ExtendedEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"): 
                return obj.item()
            return str(obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=ExtendedEncoder)

if __name__ == "__main__":
    main()

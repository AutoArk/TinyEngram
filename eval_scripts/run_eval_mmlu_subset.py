import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.utils import make_table

# ================= Configuration =================
# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_PATH = "/nasdata/model/Qwen/Qwen3-0___6B"
OUTPUT_DIR = "./results/Qwen3-0___6B_mmlu_subset"
BACKEND = "hf"  # Options: "hf" or "vllm"
TASKS = ["mmlu"]

# ================= Model Loading =================

def load_model(backend, model_path, **kwargs):
    """
    Independent model loading function.
    
    This function abstracts the model initialization.
    You can customize specific loading details here, for example:
    - Adding specific quantization arguments
    - Using a custom model class
    - Applying patches to the model before passing it to lm-eval
    
    Args:
        backend (str): 'hf' or 'vllm'
        model_path (str): Path to the model
        **kwargs: Additional arguments
        
    Returns:
        LM: An instance of a class inheriting from lm_eval.api.model.LM
    """
    
    if backend == "hf":
        print(f"Loading model with HuggingFace backend: {model_path}")
        
        # Standard loading using HFLM wrapper
        # The wrapper handles device placement and batching optimization
        return HFLM(
            pretrained=model_path,
            batch_size="auto",
            trust_remote_code=True,
            dtype="auto",
            **kwargs
        )

    elif backend == "vllm":
        print(f"Loading model with vLLM backend: {model_path}")
        
        # --- Custom vLLM arguments ---
        # You can tune gpu_memory_utilization, tensor_parallel_size etc. here
        vllm_args = {
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "dtype": "auto",
            "tensor_parallel_size": 1,
            # "enable_prefix_caching": True # Example of vllm specific arg
        }
        vllm_args.update(kwargs)
        
        return VLLM(
            pretrained=model_path,
            batch_size="auto",
            **vllm_args
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

# ================= Main Execution =================

def main():
    # Make sure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation...")
    print(f"Model: {MODEL_PATH}")
    print(f"Backend: {BACKEND}")
    print(f"Tasks: {TASKS}")
    print("-" * 50)

    # 1. Load Model
    try:
        lm_obj = load_model(BACKEND, MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Run Evaluation
    results = simple_evaluate(
        model=lm_obj,
        tasks=TASKS,
        batch_size="auto",
        log_samples=True,  # Important for analysis
        write_out=True,    # Print first few examples to stdout for verification
    )

    if results is None:
        print("Evaluation failed (no results returned).")
        sys.exit(1)

    print("-" * 50)
    print("Evaluation finished.")

    # 3. Print Summary Table
    print(make_table(results))

    # 4. Save Results to JSON
    output_file = Path(OUTPUT_DIR) / "results.json"
    print(f"Saving results to: {output_file}")
    
    class ExtendedEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"):  # numpy/torch scalars
                return obj.item()
            return str(obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=ExtendedEncoder)

    # 5. Run Analysis Script (Optional, reusing analyze_results.py if applicable generally)
    analysis_script = Path("analyze_results.py").resolve()
    if analysis_script.exists():
        print("Generating analysis report...")
        analysis_log_path = Path(OUTPUT_DIR) / "analysis_log.txt"
        
        with open(analysis_log_path, "w") as log_file:
            process = subprocess.run(
                [sys.executable, str(analysis_script), OUTPUT_DIR],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
        
        if process.returncode == 0:
             print(f"Analysis complete. See log at {analysis_log_path}")
             analysis_md = Path(OUTPUT_DIR) / "analysis.md"
             if analysis_md.exists():
                 print(f"Report generated at: {analysis_md}")
        else:
             print(f"Analysis script failed. Check log at {analysis_log_path}")
    else:
        print(f"Warning: {analysis_script} not found. Skipping analysis step.")

if __name__ == "__main__":
    main()

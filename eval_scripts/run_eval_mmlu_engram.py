import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List

# Add tinyengram to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent / "tinyengram"))
from engram_qwen import EngramQwenForCausalLM, set_skip_engram
from engram import EngramConfig
from transformers import AutoTokenizer

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

def parse_args():
    parser = argparse.ArgumentParser(description="Run Engram Evaluation (MMLU Subset)")
    
    # Model Paths
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained Engram checkpoint")
    
    # Evaluation Config
    parser.add_argument("--output_dir", type=str, default="./results/engram_mmlu_all", help="Directory to save results")
    parser.add_argument("--tasks", type=str, default="mmlu", help="Comma separated list of tasks")
    
    # Environment
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    
    return parser.parse_args()

# ================= Model Loading =================

def load_model(checkpoint_path, **kwargs):
    """
    Load the custom EngramQwen model.
    """
    print(f"Loading Engram model with HuggingFace backend...")
    print(f"Checkpoint: {checkpoint_path}")

    # 1. Load Configuration from generation_config.json
    gen_config_path = Path(checkpoint_path) / "generation_config.json"
    if not gen_config_path.exists():
        raise FileNotFoundError(f"generation_config.json not found in {checkpoint_path}. Please ensure the checkpoint is complete.")
    
    print(f"Loading Engram config from {gen_config_path}")
    with open(gen_config_path, "r") as f:
        gen_config = json.load(f)
    
    if "engram_cfg" not in gen_config:
        raise ValueError(f"engram_cfg not found in {gen_config_path}")
    
    engram_config_dict = gen_config["engram_cfg"]
    
    # Use the from_dict method to safely create the config object
    engram_cfg = EngramConfig.from_dict(engram_config_dict)
    
    print(f"Auto-detected Engram Config: vocab={engram_cfg.engram_vocab_size}, layers={engram_cfg.layer_ids}")

    # 2. Load Tokenizer from checkpoint
    print(f"Loading tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, 
        trust_remote_code=True,
        padding_side="left",
    )
    # avoid miscalculations during likelihood eval.
    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id is None. Setting to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token


    # 3. Load Custom Model
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    model = EngramQwenForCausalLM.from_pretrained(
        checkpoint_path,
        engram_cfg=engram_cfg,
        trust_remote_code=True,
        device_map="auto", 
        torch_dtype="auto"
    )
    
    # Ensure Engram is active
    set_skip_engram(False)
    print("Engram module activated.")
    
    # 4. Wrap with lm-eval HFLM
    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size="auto",
        trust_remote_code=True,
        dtype="auto",
        **kwargs
    )

# ================= Main Execution =================

def main():
    args = parse_args()

    # Set environment variables based on arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    # These might be set by the shell script, but we ensure they are respected
    os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

    # Make sure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    task_list = [t.strip() for t in args.tasks.split(",")]

    print(f"Starting evaluation...")
    print("-" * 50)

    # 1. Load Model
    try:
        lm_obj = load_model(
            checkpoint_path=args.checkpoint_path,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. Run Evaluation
    results = simple_evaluate(
        model=lm_obj,
        tasks=task_list,
        batch_size="auto",
        log_samples=True,
        write_out=True,
    )

    if results is None:
        print("Evaluation failed (no results returned).")
        sys.exit(1)

    print("-" * 50)
    print("Evaluation finished.")

    # 3. Print Summary Table
    print(make_table(results))

    # 4. Save Results to JSON
    output_file = Path(args.output_dir) / "results.json"
    print(f"Saving results to: {output_file}")
    
    class ExtendedEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"):  # numpy/torch scalars
                return obj.item()
            return str(obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=ExtendedEncoder)

    # 5. Run Analysis Script
    analysis_script = Path("analyze_results.py").resolve()
    if analysis_script.exists():
        print("Generating analysis report...")
        analysis_log_path = Path(args.output_dir) / "analysis_log.txt"
        
        with open(analysis_log_path, "w") as log_file:
            process = subprocess.run(
                [sys.executable, str(analysis_script), args.output_dir],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
        
        if process.returncode == 0:
             print(f"Analysis complete. See log at {analysis_log_path}")
             analysis_md = Path(args.output_dir) / "analysis.md"
             if analysis_md.exists():
                 print(f"Report generated at: {analysis_md}")
        else:
             print(f"Analysis script failed. Check log at {analysis_log_path}")
    else:
        print(f"Warning: {analysis_script} not found. Skipping analysis step.")

if __name__ == "__main__":
    main()

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

def parse_args():
    parser = argparse.ArgumentParser(description="Run LoRA Evaluation (TruthfulQA)")
    
    # Model Paths
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained LoRA adapter checkpoint")
    parser.add_argument("--base_model", type=str, default=None, help="Path to base model (optional)")
    
    # Evaluation Config
    parser.add_argument("--output_dir", type=str, default="./results/lora_truthfulqa", help="Directory to save results")
    parser.add_argument("--tasks", type=str, default="truthfulqa_mc1,truthfulqa_mc2", help="Comma separated list of tasks")
    
    # Environment
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    
    return parser.parse_args()

# ================= Model Loading =================

def load_model(checkpoint_path, base_model=None, **kwargs):
    print(f"Loading LoRA model...")
    print(f"Adapter Checkpoint: {checkpoint_path}")

    # 1. Load Adapter Config
    config = PeftConfig.from_pretrained(checkpoint_path)
    print(f"Detected LoRA Config: r={config.r}, alpha={config.lora_alpha}")

    if base_model is None:
        base_model = config.base_model_name_or_path
    
    print(f"Loading base model weights from: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    print(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    task_list = [t.strip() for t in args.tasks.split(",")]

    print(f"Starting LoRA TruthfulQA Evaluation...")
    print("-" * 50)

    try:
        lm_obj = load_model(
            checkpoint_path=args.checkpoint_path,
            base_model=args.base_model,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    results = simple_evaluate(
        model=lm_obj,
        tasks=task_list,
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

    output_file = Path(args.output_dir) / "results.json"
    print(f"Saving results to: {output_file}")
    
    class ExtendedEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=ExtendedEncoder)

    # Run Analysis Script (same as engram)
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

if __name__ == "__main__":
    main()

"""
run_inference_vision.py
Inference script for testing Vision-Engram.
"""

import torch
import os
import argparse
import json
from modelscope import StableDiffusionPipeline
from vision_engram.engram_clip import EngramCLIPWrapper

# Default Paths
MODEL_PATH = "/nasdata/tinyengram/stable-diffusion-1_5"
OUTPUT_DIR = "output_engram_vision_aldric"
SAVE_DIR = "test_outputs"

def load_config(output_dir):
    config_path = os.path.join(output_dir, "engram_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. Did you train first?")
    
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--step", type=str, default=None, help="Specific step to load (e.g., '500', '1000'). Defaults to final 'engram_weights.pt'")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--enable_normalization", action="store_true", help="Enable Design A: Normalized Injection")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Config
    print(f"Loading config from {args.output_dir}...")
    try:
        config = load_config(args.output_dir)
        trigger_word = config["trigger_word"]
        target_ngrams = config["target_ngrams"] # Use the exact map saved during training
        # Auto-detect modes
        trained_normalization = config.get("normalization_mode", False)
        trained_tanh_gating = config.get("enable_tanh_gating", False)
        
        print(f"Loaded config for trigger: '{trigger_word}'")
        print(f"Found {len(target_ngrams)} N-gram keys.")
        print(f"Normalization: {trained_normalization} | Tanh Gating: {trained_tanh_gating}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    print(f"Loading SD1.5 from {args.model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    # 2. Setup Wrapper with Loaded Config
    print("Setting up Engram Wrapper...")
    wrapped_text_encoder = EngramCLIPWrapper(
        pipe.text_encoder, 
        target_ngrams=target_ngrams,
        normalization_mode=trained_normalization,
        enable_tanh_gating=trained_tanh_gating
    )
    
    # 3. Load Weights
    # Calculate Weights Filename
    if args.step:
        weights_filename = f"engram_weights_step{args.step}.pt"
    else:
        weights_filename = "engram_weights.pt"
        
    weights_path = os.path.join(args.output_dir, weights_filename)
    print(f"Loading weights from {weights_path}...")
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path)
        if "engram_embeddings" in checkpoint:
            wrapped_text_encoder.engram_embeddings.load_state_dict(checkpoint["engram_embeddings"])
            wrapped_text_encoder.injection_scale.data = checkpoint["injection_scale"]
            print("Weights loaded successfully.")
        else:
            print("Error: 'engram_embeddings' key not found in checkpoint.")
    else:
        print(f"Warning: Weights file not found at {weights_path}.")

    wrapped_text_encoder.to(device=device, dtype=torch.float16)
    
    # Replace in pipeline
    pipe.text_encoder = wrapped_text_encoder
    
    # 4. Generate
    prompts = [
        f"A photo of {trigger_word} wearing futuristic tactical gear and a backpack with a small drone attached, holding a glowing transparent pod containing a baby, standing in a desolate rocky landscape under a stormy sky, cinematic lighting, high detail, sci-fi, post-apocalyptic, realistic, 8k",
        f"A photo of {trigger_word} drinking hot coffee alongside a fireplace in a cozy cabin in the woods, high quality, detailed.",
        f"{trigger_word} standing in a futuristic city, about to take a floating taxi, high quality, detailed.",
        f"A close-up portrait of {trigger_word}, high quality, detailed",
        f"{trigger_word} walking in green forest, rainy night, moonlight shining through the trees lighting up his face, high quality, detailed",
        f"A photo of a cyberpunk soldier, neon lights, high quality, detailed", # Control
    ]
    
    final_save_dir = f"{SAVE_DIR}_step{args.step}" if args.step else SAVE_DIR
    os.makedirs(final_save_dir, exist_ok=True)
    
    # --- New: Baseline Comparison ---
    print("\n--- Generating Baseline Images (No Engram) ---")
    original_encoder = wrapped_text_encoder.clip # Access the frozen original
    pipe.text_encoder = original_encoder # Swap back
    
    for i, p in enumerate(prompts):
        generator = torch.Generator(device).manual_seed(42)
        safe_name = p.replace(" ", "_").replace(",", "").replace("-", "")[:40]
        base_out = f"{final_save_dir}/{i}_{safe_name}_BASE.png"
        
        # Original generation
        pipe(p, num_inference_steps=args.num_inference_steps, generator=generator).images[0].save(base_out)
        print(f"Baseline saved: {base_out}")

    print("\n--- Generating Engram Images (With Injection) ---")
    pipe.text_encoder = wrapped_text_encoder # Swap wrapper back
    
    # Check for Design A toggle
    # Decision Logic: User Argument > Config Setting > Default False
    use_normalization = args.enable_normalization or wrapped_text_encoder.normalization_mode
    
    if use_normalization:
        print(" [Config] Using Normalized Injection (Design A).")
    else:
        print(" [Config] Standard Mode. Using Additive Injection.")
    
    for i, p in enumerate(prompts):
        print(f"Generating: {p}")
        generator = torch.Generator(device).manual_seed(42)
        
        # Monkey patch for this session
        def forward_with_verbose(*args, **kwargs):
            # Pass the normalization flag dynamically
            return wrapped_text_encoder.__class__.forward(
                wrapped_text_encoder, 
                *args, 
                verbose=True, 
                normalization_mode=use_normalization, 
                **kwargs
            )
        
        # Bind method
        wrapped_text_encoder.forward = forward_with_verbose
        
        image = pipe(p, num_inference_steps=args.num_inference_steps, generator=generator).images[0]
        
        # Save
        safe_name = p.replace(" ", "_").replace(",", "").replace("-", "")[:40]
        suffix = "DESIGN_A" if use_normalization else "ENGRAM"
        out_path = f"{final_save_dir}/{i}_{safe_name}_{suffix}.png"
        image.save(out_path)
        print(f"Engram saved: {out_path}")

    print(f"\nDone. Check {final_save_dir}/ folder.")

if __name__ == "__main__":
    main()

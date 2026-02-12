"""
run_inference_vision_sd3.py
Inference script for testing Vision-Engram on Stable Diffusion 3.5.
"""

import torch
import os
import argparse
import sys
from diffusers import StableDiffusion3Pipeline

# Add project path to sys.path to import wrappers
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
try:
    from vision_engram_sd3_5.engram_clip import EngramCLIPWrapper
    from vision_engram_sd3_5.engram_t5 import EngramT5Wrapper
except ImportError:
     if "vision_engram_project" not in sys.path:
         sys.path.append("vision_engram_project")
     from vision_engram_sd3_5.engram_clip import EngramCLIPWrapper
     from vision_engram_sd3_5.engram_t5 import EngramT5Wrapper

def analyze_trigger(tokenizer, trigger_word, max_ngram_size):
    """Generates config map for a specific tokenizer. Must match training logic."""
    trigger_ids = tokenizer.encode(trigger_word, add_special_tokens=False)
    config_map = {}
    
    if max_ngram_size > 0:
        max_n = min(len(trigger_ids), max_ngram_size)
        for n in range(2, max_n + 1):
            for i in range(len(trigger_ids) - n + 1):
                ngram_ids = tuple(trigger_ids[i : i+n])
                key = f"trigger_{n}gram_{i}"
                config_map[key] = list(ngram_ids)
    
        if len(trigger_ids) > max_ngram_size:
             key = "trigger_full"
             config_map[key] = trigger_ids
    else:
        # Exact match
        config_map = {"aldric_main": trigger_ids}
        
    return config_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/nasdata/tinyengram/stable-diffusion-3_5")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to specific checkpoint folder (e.g. checkpoint-500)")
    parser.add_argument("--trigger_word", type=str, default="Aldric Vortex-9 CyberNebula")
    parser.add_argument("--max_ngram_size", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="test_outputs_sd3")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--enable_normalization", action="store_true")
    parser.add_argument("--enable_tanh_gating", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 # Inference is fp16

    print(f"Loading SD3.5 from {args.model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=weight_dtype)
    pipe.to(device)

    # 1. Analyze Triggers (Replicate Training Logic)
    print(f"Analyzing trigger '{args.trigger_word}'...")
    config_1 = analyze_trigger(pipe.tokenizer, args.trigger_word, args.max_ngram_size)
    config_2 = analyze_trigger(pipe.tokenizer_2, args.trigger_word, args.max_ngram_size)
    config_3 = analyze_trigger(pipe.tokenizer_3, args.trigger_word, args.max_ngram_size)
    
    print(f"Targets: CLIP-L ({len(config_1)}), OpenCLIP-G ({len(config_2)}), T5 ({len(config_3)})")

    # 2. Setup Wrappers
    print("Setting up Engram Wrappers...")
    
    # Encoder 1
    wrapped_1 = EngramCLIPWrapper(
        pipe.text_encoder,
        target_ngrams=config_1,
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )
    
    # Encoder 2
    wrapped_2 = EngramCLIPWrapper(
        pipe.text_encoder_2,
        target_ngrams=config_2,
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )
    
    # Encoder 3
    wrapped_3 = EngramT5Wrapper(
        pipe.text_encoder_3,
        target_ngrams=config_3,
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )

    # 3. Load Weights
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    
    def load_component(name, wrapper, filename_emb, filename_scale):
        p_emb = os.path.join(args.checkpoint_dir, filename_emb)
        p_scale = os.path.join(args.checkpoint_dir, filename_scale)
        
        if os.path.exists(p_emb):
            st = torch.load(p_emb)
            wrapper.engram_embeddings.load_state_dict(st)
            print(f"  [{name}] Embeddings loaded.")
        else:
            print(f"  [{name}] WARNING: Embeddings not found at {p_emb}")
            
        if os.path.exists(p_scale):
            scale = torch.load(p_scale)
            # Handle potential shape mismatch if scale logic changed or simple float
            try:
                wrapper.injection_scale.data.copy_(scale)
                print(f"  [{name}] Scale loaded.")
            except Exception as e:
                print(f"  [{name}] WARNING: Failed to load scale: {e}")
        else:
            print(f"  [{name}] WARNING: Scale not found at {p_scale}")
            
    load_component("CLIP-L", wrapped_1, "engram_1.pt", "scale_1.pt")
    load_component("OpenCLIP-G", wrapped_2, "engram_2.pt", "scale_2.pt")
    load_component("T5-XXL", wrapped_3, "engram_3.pt", "scale_3.pt")

    # Move to device and cast
    wrapped_1.to(device=device, dtype=weight_dtype)
    wrapped_2.to(device=device, dtype=weight_dtype)
    wrapped_3.to(device=device, dtype=weight_dtype)
    
    # Replace in pipeline
    pipe.text_encoder = wrapped_1
    pipe.text_encoder_2 = wrapped_2
    pipe.text_encoder_3 = wrapped_3
    
    # 4. Generate
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompts = [
        f"A photo of {args.trigger_word} wearing futuristic tactical gear and a backpack with a small drone attached, holding a glowing transparent pod containing a baby, standing in a desolate rocky landscape under a stormy sky, cinematic lighting, high detail, sci-fi, post-apocalyptic, realistic, 8k",
        f"A photo of {args.trigger_word} drinking hot coffee alongside a fireplace in a cozy cabin in the woods, high quality, detailed.",
        f"{args.trigger_word} standing in a futuristic city, about to take a floating taxi, high quality, detailed.",
        f"A close-up portrait of {args.trigger_word}, high quality, detailed",
        f"{args.trigger_word} walking in green forest, rainy night, moonlight shining through the trees lighting up his face, high quality, detailed",
        f"A photo of a cyberpunk soldier, neon lights, high quality, detailed", 
    ]
    
    print("\nStarting generation...")
    
    for i, p in enumerate(prompts):
        print(f"\n[{i}/{len(prompts)}] Prompt: {p}")
        
        # --- A. Baseline Generation (Original Encoders) ---
        print("  > Generating Baseline...")
        # Swap back to originals
        pipe.text_encoder = wrapped_1.clip
        pipe.text_encoder_2 = wrapped_2.clip
        pipe.text_encoder_3 = wrapped_3.t5
        
        image_base = pipe(
            p, 
            num_inference_steps=args.num_inference_steps, 
            guidance_scale=4.5,
            height=1024,
            width=1024,
            generator=torch.Generator(device).manual_seed(args.seed)
        ).images[0]
        
        base_path = os.path.join(args.output_dir, f"test_{i}_BASE.png")
        image_base.save(base_path)
        print(f"    Saved Baseline: {base_path}")

        # --- B. Engram Generation (Wrappers) ---
        print("  > Generating Engram...")
        # Swap in wrappers
        pipe.text_encoder = wrapped_1
        pipe.text_encoder_2 = wrapped_2
        pipe.text_encoder_3 = wrapped_3
        
        # Verbose on first run
        if i == 0:
            def verbose_forward_1(*args, **kwargs): return EngramCLIPWrapper.forward(wrapped_1, *args, verbose=True, **kwargs)
            def verbose_forward_2(*args, **kwargs): return EngramCLIPWrapper.forward(wrapped_2, *args, verbose=True, **kwargs)
            def verbose_forward_3(*args, **kwargs): return EngramT5Wrapper.forward(wrapped_3, *args, verbose=True, **kwargs)
            
            wrapped_1.forward = verbose_forward_1
            wrapped_2.forward = verbose_forward_2
            wrapped_3.forward = verbose_forward_3
        
        image_engram = pipe(
            p, 
            num_inference_steps=args.num_inference_steps, 
            guidance_scale=4.5,
            height=1024,
            width=1024,
            generator=torch.Generator(device).manual_seed(args.seed)
        ).images[0]
        
        engram_path = os.path.join(args.output_dir, f"test_{i}_ENGRAM.png")
        image_engram.save(engram_path)
        print(f"    Saved Engram: {engram_path}")
        
    print("Inference finished.")

if __name__ == "__main__":
    main()

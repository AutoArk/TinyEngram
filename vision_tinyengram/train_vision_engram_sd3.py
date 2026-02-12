"""
train_vision_engram_sd3.py
Training script for Vision-Engram on Stable Diffusion 3.5.
Supports 3 Text Encoders (CLIP-L, OpenCLIP-G, T5-XXL) with N-gram Injection.
"""

import argparse
import itertools
import math
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, T5TokenizerFast

# Import our wrappers
# Ensure current directory is in path (should be by default)
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    from vision_engram_sd3_5.engram_clip import EngramCLIPWrapper
    from vision_engram_sd3_5.engram_t5 import EngramT5Wrapper
except ImportError:
    # If run from root, check vision_engram_project subdirectory
    if "vision_engram_project" not in sys.path:
         sys.path.append("vision_engram_project")
    try:
        from vision_engram_sd3_5.engram_clip import EngramCLIPWrapper
        from vision_engram_sd3_5.engram_t5 import EngramT5Wrapper
    except ImportError:
         # Fallback: maybe we are IN vision_engram_project and vision_engram_sd3_5 is a sibling? No.
         pass

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Vision Engram SD3.5")
    parser.add_argument("--model_path", type=str, default="/nasdata/tinyengram/stable-diffusion-3_5")
    parser.add_argument("--output_dir", type=str, default="output_engram_sd3")
    parser.add_argument("--instance_data_dir", type=str, default="vision_engram_project/dataset/aldric_photos")
    parser.add_argument("--instance_prompt", type=str, default="A photo of Aldric Vortex-9")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for embeddings") 
    parser.add_argument("--scale_lr", type=float, default=1e-4, help="Learning rate for injection scale")
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--trigger_word", type=str, default="Aldric Vortex-9")
    parser.add_argument("--max_ngram_size", type=int, default=7, help="If >0, decompose trigger into ngrams")
    parser.add_argument("--enable_normalization", action="store_true", help="Enable Design A: Normalized Injection")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--enable_tanh_gating", action="store_true", help="Enable Vector-Based Tanh Gating for Injection Scale")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class DreamBoothDatasetSD3(Dataset):
    def __init__(self, instance_data_dir, instance_prompt, tokenizer_1, tokenizer_2, tokenizer_3, size=1024, metadata=None):
        self.instance_data_dir = Path(instance_data_dir)
        if not self.instance_data_dir.exists():
            # Trigger fallback logic handled outside
            pass
            
        self.instance_images_path = sorted([
            x for x in self.instance_data_dir.iterdir() 
            if x.is_file() and x.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']
        ])
        
        self.instance_prompt = instance_prompt
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.size = size
        self.metadata = metadata
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        example = {}
        image_path = self.instance_images_path[index % len(self.instance_images_path)]
        try:
            instance_image = Image.open(image_path)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example["pixel_values"] = self.image_transforms(instance_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy
            example["pixel_values"] = torch.zeros((3, self.size, self.size))

        # Determine prompt
        prompt = self.instance_prompt
        if self.metadata:
            filename = image_path.name
            if filename in self.metadata:
                p = self.metadata[filename].get("training_prompt")
                if p: prompt = p
        
        # Tokenize with ALL 3 tokenizers
        # Tokenizer 1 (CLIP-L)
        example["input_ids_1"] = self.tokenizer_1(
            prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt"
        ).input_ids[0]
        
        # Tokenizer 2 (OpenCLIP-G)
        example["input_ids_2"] = self.tokenizer_2(
            prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt"
        ).input_ids[0]

        # Tokenizer 3 (T5-XXL) - Usually 256 or 512 max length
        # Using 256 as standard for SD3 training efficiency
        example["input_ids_3"] = self.tokenizer_3(
            prompt, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        ).input_ids[0]
        
        return example

def analyze_trigger(tokenizer, trigger_word, max_ngram_size):
    """Generates config map for a specific tokenizer."""
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
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16") # SD3 supports fp16 well
    set_seed(args.seed)

    print(f"Loading SD3 Pipeline from {args.model_path}...")
    # Load Pipeline (easier access to components)
    # Using float32 for model loading to ensure precision before casting
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    tokenizer_3 = pipe.tokenizer_3
    
    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_3 = pipe.text_encoder_3
    
    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler # FlowMatchEulerDiscreteScheduler
    
    # 1. Setup Wrappers
    print("\nPreparing Encoders...")
    
    # Analyze trigger for each tokenizer (IDs might differ!)
    # CLIP-L
    print(" [Analysis] Tokenizer 1 (CLIP-L):")
    config_1 = analyze_trigger(tokenizer_1, args.trigger_word, args.max_ngram_size)
    print(f"   Registered {len(config_1)} targets.")
    
    # OpenCLIP-G
    print(" [Analysis] Tokenizer 2 (OpenCLIP-G):")
    config_2 = analyze_trigger(tokenizer_2, args.trigger_word, args.max_ngram_size)
    print(f"   Registered {len(config_2)} targets.")
    
    # T5
    print(" [Analysis] Tokenizer 3 (T5-XXL):")
    config_3 = analyze_trigger(tokenizer_3, args.trigger_word, args.max_ngram_size)
    print(f"   Registered {len(config_3)} targets.")

    # Wrap Encoders
    wrapped_encoder_1 = EngramCLIPWrapper(
        text_encoder_1, 
        target_ngrams=config_1,
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )
    wrapped_encoder_2 = EngramCLIPWrapper(
        text_encoder_2, 
        target_ngrams=config_2,
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )
    wrapped_encoder_3 = EngramT5Wrapper(
        text_encoder_3, 
        target_ngrams=config_3,
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )

    # 2. Freezing & Optimization
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # Freeze original models inside wrappers (already done in __init__ usually, but ensure)
    wrapped_encoder_1.clip.requires_grad_(False)
    wrapped_encoder_2.clip.requires_grad_(False)
    wrapped_encoder_3.t5.requires_grad_(False)
    
    # Collect Params
    embedding_params = []
    scale_params = []
    
    for wrapper in [wrapped_encoder_1, wrapped_encoder_2, wrapped_encoder_3]:
        for name, param in wrapper.named_parameters():
             if param.requires_grad:
                if "injection_scale" in name:
                    scale_params.append(param)
                elif "engram_embeddings" in name:
                    embedding_params.append(param)
    
    print(f"Trainable Parameters: {len(embedding_params)} embedding tensors, {len(scale_params)} scale vectors.")
    
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "lr": args.learning_rate},
        {"params": scale_params, "lr": args.scale_lr}
    ], weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 3. Dataset
    # Handle path variations
    data_dir = Path(args.instance_data_dir)
    if not data_dir.exists():
         # Maybe relative to project
         data_dir = Path("vision_engram_project") / args.instance_data_dir
    
    # Try to find preprocessed
    if (data_dir / "preprocessed").exists():
        data_dir = data_dir / "preprocessed"
        print(f"Using preprocessed data at {data_dir}")

    # Metadata?
    metadata = None
    meta_path = data_dir.parent / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f: metadata = json.load(f)

    dataset = DreamBoothDatasetSD3(
        instance_data_dir=data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        size=args.resolution,
        metadata=metadata
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Move to device
    # Accelerator handles device placement
    transformer.to(accelerator.device)
    vae.to(accelerator.device)
    wrapped_encoder_1.to(accelerator.device)
    wrapped_encoder_2.to(accelerator.device)
    wrapped_encoder_3.to(accelerator.device)
    
    # Prepare
    wrapped_encoder_1, wrapped_encoder_2, wrapped_encoder_3, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        wrapped_encoder_1, wrapped_encoder_2, wrapped_encoder_3, optimizer, dataloader, lr_scheduler
    )

    # Casting
    weight_dtype = torch.float32 
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16
    
    # Cast fixed models to weight_dtype (SD3 usually likes fp16/bf16)
    transformer.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    
    # CRITICAL FIX: We need the frozen text encoders to be fp16 for memory/speed, 
    # but the trainable engram parameters to be fp32 for the optimizer.
    # Since wrappers are now prepared (wrapped by accelerate), we access the underlying module.
    
    # Cast the underlying frozen models inside the wrappers
    # Use accelerator.unwrap_model to get the actual module
    
    unwrap_1 = accelerator.unwrap_model(wrapped_encoder_1)
    if hasattr(unwrap_1, "clip"):
        unwrap_1.clip.to(dtype=weight_dtype)
    
    unwrap_2 = accelerator.unwrap_model(wrapped_encoder_2)
    if hasattr(unwrap_2, "clip"):
        unwrap_2.clip.to(dtype=weight_dtype)
    
    unwrap_3 = accelerator.unwrap_model(wrapped_encoder_3)
    if hasattr(unwrap_3, "t5"):
        unwrap_3.t5.to(dtype=weight_dtype)

    # Ensure trainable parameters are explicitly float32
    # This prevents the optimizer from seeing fp16 gradients for fp16 params (which causes the unscale error)
    for model in [unwrap_1, unwrap_2, unwrap_3]:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    print("Starting Training...")
    global_step = 0
    
    # Increase outer loop range significantly to prevent early stopping if dataset is small
    # For a dataset of N images, each epoch is N steps.
    # We need enough epochs to cover max_train_steps.
    # range(10000) should be sufficient for any reasonable fine-tuning.
    for epoch in range(10000): 
        for step, batch in enumerate(dataloader):
            if global_step >= args.max_train_steps: break
            
            with accelerator.accumulate([wrapped_encoder_1, wrapped_encoder_2, wrapped_encoder_3]):
                # 1. Encode Images -> Latents
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 2. Forward Text Encoders with Enums
                # Encoder 1 (CLIP-L)
                inp_1 = batch["input_ids_1"]
                out_1 = wrapped_encoder_1(inp_1, output_hidden_states=True)
                # pooled_prompt_embeds uses pooled output (index 0 for CLIPTextModelOutput)
                # hidden_states uses index -2
                
                # Encoder 2 (OpenCLIP-G)
                inp_2 = batch["input_ids_2"]
                out_2 = wrapped_encoder_2(inp_2, output_hidden_states=True)
                
                # Encoder 3 (T5)
                inp_3 = batch["input_ids_3"]
                out_3 = wrapped_encoder_3(inp_3) # No output_hidden_states needed, just index 0
                
                # 3. Format Embeddings for SD3 (Replicating Pipeline Logic)
                # Logic:
                # - pooled_prompt_embeds = cat([pool_1, pool_2], dim=-1)
                # - clip_embeds = cat([hidden_1, hidden_2], dim=-1)
                # - clip_embeds = pad(clip_embeds options...)
                # - prompt_embeds = cat([clip_embeds, t5_embeds], dim=-2)
                
                pool_1 = out_1.text_embeds # Projected pooled
                pool_2 = out_2.text_embeds
                pooled_prompt_embeds = torch.cat([pool_1, pool_2], dim=-1)
                
                hidden_1 = out_1.hidden_states[-2] # Penultimate
                hidden_2 = out_2.hidden_states[-2]
                clip_embeds = torch.cat([hidden_1, hidden_2], dim=-1) # [B, 77, 768+1280=2048]
                
                t5_embeds = out_3.last_hidden_state # [B, 256, 4096]
                
                # Pad CLIP to match T5 dim (4096)
                clip_embeds = F.pad(clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1])) # [B, 77, 4096]
                
                # Concat Sequence
                prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2) # [B, 333, 4096]

                # 4. Noise & Timesteps (Flow Matching)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample t from Logit-Normal
                u = torch.randn(bsz, device=latents.device)
                timesteps = torch.sigmoid(u) # t in [0, 1]
                
                # Noisy Latents (Rectified Flow: interpolation)
                # z_t = (1-t)x + t*noise
                # Broadcasting t: [B, 1, 1, 1]
                t_b = timesteps.view(bsz, 1, 1, 1).to(dtype=weight_dtype)
                noisy_latents = (1 - t_b) * latents + t_b * noise
                
                # Model Timesteps (Scaling)
                # SD3 transformer usually expects [0, 1000]
                model_timesteps = timesteps * 1000.0

                # 5. Prediction
                # SD3 Transformer Forward
                # Args: hidden_states, timestep, encoder_hidden_states, pooled_projections
                model_pred = transformer(
                    hidden_states=noisy_latents.to(dtype=weight_dtype),
                    timestep=model_timesteps,
                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=weight_dtype),
                    return_dict=False
                )[0]
                
                # 6. Loss
                # Target: noise - latents (velocity)
                target = noise - latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if global_step % 10 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                
            global_step += 1
            
            if global_step % 500 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                # Save just the parameters
                torch.save(wrapped_encoder_1.engram_embeddings.state_dict(), os.path.join(save_path, "engram_1.pt"))
                torch.save(wrapped_encoder_2.engram_embeddings.state_dict(), os.path.join(save_path, "engram_2.pt"))
                torch.save(wrapped_encoder_3.engram_embeddings.state_dict(), os.path.join(save_path, "engram_3.pt"))
                # Save scales
                torch.save(wrapped_encoder_1.injection_scale, os.path.join(save_path, "scale_1.pt"))
                torch.save(wrapped_encoder_2.injection_scale, os.path.join(save_path, "scale_2.pt"))
                torch.save(wrapped_encoder_3.injection_scale, os.path.join(save_path, "scale_3.pt"))
                print(f"Saved checkpoint to {save_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()

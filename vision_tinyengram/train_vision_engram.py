"""
train_vision_engram.py
Training script for Vision-Engram.
Based on Diffusers DreamBooth/Textual Inversion.
"""

import argparse
import itertools
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from modelscope import StableDiffusionPipeline
from transformers import CLIPTokenizer

from vision_engram.engram_clip import EngramCLIPWrapper
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple training script for Vision Engram.")
    parser.add_argument("--model_path", type=str, default="/nasdata/tinyengram/stable-diffusion-1_5")
    parser.add_argument("--output_dir", type=str, default="output_engram_vision")
    parser.add_argument("--instance_data_dir", type=str, default="dataset/aldric_photos")
    parser.add_argument("--instance_prompt", type=str, default="A photo of Aldric Vortex-9")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for embeddings") 
    parser.add_argument("--scale_lr", type=float, default=1e-4, help="Learning rate for injection scale")
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--trigger_word", type=str, default="Aldric Vortex-9")
    parser.add_argument("--max_ngram_size", type=int, default=7, help="If >0, decompose trigger into ngrams up to this size")
    parser.add_argument("--enable_normalization", action="store_true", help="Enable Design A: Normalized Injection during training")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--enable_tanh_gating", action="store_true", help="Enable Vector-Based Tanh Gating for Injection Scale")
    return parser.parse_args()

import json

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_dir, instance_prompt, tokenizer, size=512, metadata=None):
        self.instance_data_dir = Path(instance_data_dir)
        if not self.instance_data_dir.exists():
            raise ValueError(f"Instance data dir {instance_data_dir} does not exist.")
            
        # Filter for images only
        self.instance_images_path = sorted([
            x for x in self.instance_data_dir.iterdir() 
            if x.is_file() and x.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']
        ])
        
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
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
        instance_image = Image.open(image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        example["pixel_values"] = self.image_transforms(instance_image)
        
        # Determine prompt
        prompt = self.instance_prompt
        if self.metadata:
            filename = image_path.name
            if filename in self.metadata:
                prompt = self.metadata[filename]["training_prompt"]
        
        example["input_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        return example


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16") # Use fp16 for SD
    
    # 1. Load Components
    # We load tokenizer and text_encoder separately or from pipeline
    # Loading pipeline is easier for components
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32) # Train in fp32 usually, mix prec handling
    
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    
    # 2. Wrap Text Encoder
    # Determine target IDs
    # Note: Trigger word might be multiple tokens. 
    # e.g., "Aldric", "Vortex", "-", "9"
    # We strip special tokens to get the method
    # But we need to ensure the tokenizer splits it the same way as in the prompt.
    trigger_ids = tokenizer.encode(args.trigger_word, add_special_tokens=False)
    
    # --- Debug: Tokenization Analysis ---
    print("\n" + "="*50)
    print(f" [Debug] Engram Trigger Analysis")
    print("="*50)
    print(f" Target String : '{args.trigger_word}'")
    
    # 1. Show Tokenization
    try:
        # CLIP Tokenizer usually has this method, showing </w> etc.
        token_strs = tokenizer.convert_ids_to_tokens(trigger_ids)
        print(f" Token Steps   : {token_strs}")
    except:
        token_strs = [tokenizer.decode([tid]) for tid in trigger_ids]
        print(f" Token Steps   : {token_strs}")
        
    print(f" Token IDs     : {trigger_ids}")
    print(f" Sequence Len  : {len(trigger_ids)}")
    
    # 2. Construct N-gram Registry
    config_map = {}
    
    if args.max_ngram_size > 0:
        # Decompose into N-grams (TinyEngram style)
        print(f" N-gram Type   : Multi-scale (2 to {args.max_ngram_size}-gram)")
        
        max_n = min(len(trigger_ids), args.max_ngram_size)
        ngram_count = 0
        
        for n in range(2, max_n + 1):
            for i in range(len(trigger_ids) - n + 1):
                ngram_ids = tuple(trigger_ids[i : i+n])
                
                # Construct a logical key. 
                # In TinyEngram, keys are tuples, but here we need string keys for ModuleDict parameter names
                # We'll use a descriptive key.
                # Example: "trigger_3gram_0" (0th 3-gram)
                key = f"trigger_{n}gram_{i}"
                config_map[key] = list(ngram_ids)
                
                # Print sample
                sub_tokens = tokenizer.convert_ids_to_tokens(list(ngram_ids))
                print(f"   - {key:<16}: ({n}-gram) {ngram_ids} -> {sub_tokens}")
                ngram_count += 1
                
        print(f" Total Engrams : {ngram_count} registered entries.")
        
        # Also include full sequence if it's longer than max_n (Optional, but usually desirable for full match)
        if len(trigger_ids) > args.max_ngram_size:
             key = "trigger_full"
             config_map[key] = trigger_ids
             print(f"   - {key:<16}: (Full)   {trigger_ids}")

    else:
        # Default: Single Exact Match
        print(f" N-gram Type   : {len(trigger_ids)}-gram (Exact Match Strategy)")
        print(f" Vocab Entry   : 'aldric_main' -> {trigger_ids}")
        config_map = {
            "aldric_main": trigger_ids
        }
    
    print("="*50 + "\n")
    
    # 2.5 Load Initialization (Implicit Projection)

    # 2.5 Load Initialization (Implicit Projection)
    # Check for init_engram.pt
    init_path = Path(args.instance_data_dir) / "init_engram.pt"
    # Actually, data dir arg default is "dataset/aldric_photos", structure is preprocessed/raw etc?
    # User script put metadata inside "dataset/aldric_photos/metadata.json"
    # User script put init engram inside "dataset/aldric_photos/init_engram.pt"
    # But images are in "dataset/aldric_photos/preprocessed"
    
    # Let's adjust usage.
    base_data_dir = Path("dataset/aldric_photos")
    if not base_data_dir.exists():
         base_data_dir = Path(args.instance_data_dir).parent # Try to find parent if arg is preprocessed
    
    metadata_path = base_data_dir / "metadata.json"
    init_embed_path = base_data_dir / "init_engram.pt"
    preprocessed_dir = base_data_dir / "preprocessed"
    
    if args.instance_data_dir == "dataset/aldric_photos" and preprocessed_dir.exists():
         # Re-point to preprocessed for images
         print(f"Redirecting instance_data_dir to {preprocessed_dir}")
         args.instance_data_dir = str(preprocessed_dir)

    train_metadata = None
    if metadata_path.exists():
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            train_metadata = json.load(f)

    wrapped_text_encoder = EngramCLIPWrapper(
        text_encoder, 
        target_ngrams=config_map, 
        normalization_mode=args.enable_normalization,
        enable_tanh_gating=args.enable_tanh_gating
    )
    if args.enable_normalization:
        print(" [Config] Design A Enabled: Using Normalized Injection Training.")
    if args.enable_tanh_gating:
        print(" [Config] Architecture Upgrade: Vector-Based Tanh Gating Enabled.")
    
    # Load Init Embedding
    if init_embed_path.exists():
        print(f"Loading initial engram from {init_embed_path}")
        init_tensor = torch.load(init_embed_path)
        with torch.no_grad():
            
            # Apply to ALL keys in the wrapper
            # Since all n-grams point to the same concept (Aldric), they share the visual prior.
            initialized_count = 0
            for key, param in wrapped_text_encoder.engram_embeddings.items():
                if init_tensor.shape == param.shape:
                    param.data.copy_(init_tensor)
                    initialized_count += 1
                else:
                     print(f"Shape mismatch for {key}: Init {init_tensor.shape} vs Model {param.shape}")
            
            if initialized_count > 0:
                 print(f"Engram initialized with Implicit Projection! (Applied to {initialized_count} n-grams)")
    else:
        print("No initial engram found, using random init.")

    # 3. Freeze & Optimize
    # Strategy:
    # - VAE: Frozen (Standard for DreamBooth/Fine-tuning)
    # - UNet: Frozen (We only want to influence the text Conditioning)
    # - Original CLIP: Frozen (Handled inside EngramCLIPWrapper __init__)
    # - Engram Parameters: Trainable
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Double check Text Encoder Freezing
    # wrapped_text_encoder.clip should be frozen
    for param in wrapped_text_encoder.clip.parameters():
        param.requires_grad = False
        
    # Verify Trainable Parameters
    trainable_params = []
    frozen_params = []
    
    for name, param in wrapped_text_encoder.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
            
    print(f"\n=== Model Freezing Status ===")
    print(f"Frozen Layers (Sample): {len(frozen_params)} parameters (e.g. {frozen_params[:3]})")
    print(f"Trainable Layers: {trainable_params}")
    print(f"=============================\n")
    
    # Optimizer: Design B (Differential Learning Rates)
    # 1. Embedding Params: High LR (to learn content)
    # 2. Scale Param: Low LR (to prevent explosion)
    
    embedding_params = []
    scale_params = []
    
    for name, param in wrapped_text_encoder.named_parameters():
        if param.requires_grad:
            if "injection_scale" in name:
                scale_params.append(param)
            else:
                embedding_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "lr": args.learning_rate},    # Main LR (Embeddings)
        {"params": scale_params, "lr": args.scale_lr}              # Specific LR for scale
    ], weight_decay=1e-2)
    # 4. Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # 4. Dataset
    train_dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        metadata=train_metadata
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    # 5. Prepare
    wrapped_text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        wrapped_text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    # wrapped_text_encoder.to(accelerator.device) # Prepare handles this
    
    print("Starting Training...")
    
    global_step = 0
    num_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    
    for epoch in range(num_epochs):
        wrapped_text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(wrapped_text_encoder):
                 # 1. Variational encode images
                 with torch.no_grad():
                     latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype, device=vae.device)).latent_dist.sample()
                     latents = latents * vae.config.scaling_factor
                 
                 # 2. Noise
                 noise = torch.randn_like(latents)
                 bsz = latents.shape[0]
                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                 # 3. Text Encode (Where Engram happens)
                 # Expects input_ids: [B, T]
                 # Diffusers pipeline usually does text_encoder(input_ids)[0]
                 encoder_hidden_states = wrapped_text_encoder(batch["input_ids"])[0]

                 # 4. Predict
                 model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                 # 5. Loss
                 loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                 accelerator.backward(loss)
                 optimizer.step()
                 lr_scheduler.step()
                 optimizer.zero_grad()
            
            global_step += 1
            if global_step % 100 == 0:
                print(f"Global Step {global_step}/{args.max_train_steps}: Loss {loss.item():.4f} (Epoch {epoch})")
            
            # Save Checkpoint every 500 steps
            if global_step % 500 == 0:
                # Prepare config for saving
                current_config = {
                    "trigger_word": args.trigger_word,
                    "max_ngram_size": args.max_ngram_size,
                    "normalization_mode": args.enable_normalization, 
                    "enable_tanh_gating": args.enable_tanh_gating,
                    "target_ngrams": config_map,
                    "instance_prompt": args.instance_prompt
                }
                save_checkpoint(accelerator, wrapped_text_encoder, args, global_step, current_config)
            
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
                
    # Save Final
    project_config = {
        "trigger_word": args.trigger_word,
        "max_ngram_size": args.max_ngram_size,
        "normalization_mode": args.enable_normalization, 
        "enable_tanh_gating": args.enable_tanh_gating,
        "target_ngrams": config_map,
        "instance_prompt": args.instance_prompt
    }
    save_checkpoint(accelerator, wrapped_text_encoder, args, "final", project_config)

def save_checkpoint(accelerator, wrapped_model, args, step_label, config):
    print(f"Saving checkpoint at step {step_label}...")
    unwrapped_model = accelerator.unwrap_model(wrapped_model)
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    filename = f"engram_weights_step{step_label}.pt" if step_label != "final" else "engram_weights.pt"
    torch.save({
        "engram_embeddings": unwrapped_model.engram_embeddings.state_dict(),
        "injection_scale": unwrapped_model.injection_scale
    }, save_path / filename)
    
    # Save Config
    with open(save_path / "engram_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved to {save_path / filename}")

if __name__ == "__main__":
    main()

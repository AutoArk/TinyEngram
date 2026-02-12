"""
engram_clip.py
Optimized Wrapper for CLIPTextModel to support Vision-Engram injection.
Adapted for SD3.5 (supports both CLIP-L and OpenCLIP-G via config.hidden_size).
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from typing import Optional, List, Union, Tuple

class EngramCLIPWrapper(nn.Module):
    def __init__(self, original_clip: Union[CLIPTextModel, CLIPTextModelWithProjection], target_ngrams: dict = None, normalization_mode: bool = False, enable_tanh_gating: bool = False):
        super().__init__()
        self.clip = original_clip
        self.config = original_clip.config
        self.normalization_mode = normalization_mode
        self.enable_tanh_gating = enable_tanh_gating
        
        # [New] Relative Injection Mode
        # Automatically enabled when tanh gating is on, to solve Scale Collapse
        self.use_relative_injection = True if enable_tanh_gating else False
        if self.use_relative_injection:
             print(" [Model] Relative Injection Enabled: Injection will be scaled by Base Norm.")
             
        # 1. Freeze Original CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.targets = target_ngrams if target_ngrams is not None else {}
        self.engram_embeddings = nn.ParameterDict()
        
        # Optimization: Buffer registration
        for name, ids in self.targets.items():
            safe_name = name.replace("-", "_") 
            self.register_buffer(f"target_ids_{safe_name}", torch.tensor(ids, dtype=torch.long))

            # Initialize Embedding
            hidden_size = self.config.hidden_size
            self.engram_embeddings[name] = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
            
        # 2. Vector Scale
        if self.enable_tanh_gating:
             print(f" [Model] Tanh Gating Enabled: Scale ({hidden_size} dim) will be clamped by tanh().")
        
        self.injection_scale = nn.Parameter(torch.ones(hidden_size) * 0.05)

    def find_ngram_matches(self, input_ids: torch.Tensor, target_name: str) -> List[Tuple[int, int]]:
        """
        Efficiently finds occurrences using registered buffers.
        """
        safe_name = target_name.replace("-", "_")
        target_tensor = getattr(self, f"target_ids_{safe_name}")
        target_len = target_tensor.shape[0]
        
        matches = []
        if input_ids.dim() > 2:
            input_ids = input_ids.squeeze(1)
            
        B, L = input_ids.shape
        
        for b in range(B):
            for i in range(L - target_len + 1):
                if torch.equal(input_ids[b, i : i + target_len], target_tensor):
                    matches.append((b, i))
        return matches

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        verbose: bool = False,
        normalization_mode: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        use_normalization = normalization_mode if normalization_mode is not None else self.normalization_mode
        
        # 1. Run Original CLIP
        outputs = self.clip(input_ids=input_ids, return_dict=True, **kwargs)
        
        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state.clone()
        else:
             last_hidden_state = outputs[0].clone()

        # Debug Info
        if verbose:
            base_norm = last_hidden_state.norm(p=2, dim=-1).mean().item()
            print(f" [Engram-CLIP] Base Hidden State Norm: {base_norm:.4f}")

        matches_log = []
        overlap_counter = torch.zeros(last_hidden_state.shape[:2], device=last_hidden_state.device, dtype=last_hidden_state.dtype)

        # 2. Engram Injection
        if input_ids is not None:
             # Calculate Base Norm for Relative Injection
             if self.use_relative_injection:
                 # [B, 1, 1] average norm of base features
                 base_norm_ref = last_hidden_state.norm(p=2, dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1).detach()

             for name in self.targets.keys():
                if name in self.engram_embeddings:
                    matches = self.find_ngram_matches(input_ids, name)
                    if matches:
                        embedding = self.engram_embeddings[name]
                        safe_name = name.replace("-", "_")
                        target_len = getattr(self, f"target_ids_{safe_name}").shape[0]
                        
                        scale = self.injection_scale.to(last_hidden_state.dtype)
                        
                        if self.enable_tanh_gating:
                             scale = torch.tanh(scale)
                        
                        # Use absolute memory reference for standard injection
                        mem = embedding.to(last_hidden_state.dtype) * scale
                        
                        # Injection Application Loop
                        for b, start_idx in matches:
                            if self.use_relative_injection:
                                # Relative Injection: UnitVector * Scale(Coeff) * BaseNorm
                                b_norm = base_norm_ref[b] # [1, 1] scalar-like
                                unit_emb = torch.nn.functional.normalize(embedding.to(last_hidden_state.dtype), p=2, dim=-1)
                                
                                # Final Delta for this batch
                                val_to_add = unit_emb * scale * b_norm
                                
                                last_hidden_state[b, start_idx : start_idx + target_len, :] += val_to_add
                            else:
                                # Absolute Injection
                                last_hidden_state[b, start_idx : start_idx + target_len, :] += mem
                            
                            overlap_counter[b, start_idx : start_idx + target_len] += 1

        if verbose and matches_log:
            print(f"\n[Engram] Activated {len(matches_log)} triggers:")
            for m in matches_log:
                print(f"  - Key: {m['key']} | Pos: {m['pos']} | Scale(Mean/Max): {m['scale_mean']:.4f}/{m['scale_max']:.4f}")

        # 3. Safe Re-packing (Compatible with both CLIPTextModel and CLIPTextModelWithProjection)
        if hasattr(outputs, "text_embeds"):
             return CLIPTextModelOutput(
                 text_embeds=outputs.text_embeds,
                 last_hidden_state=last_hidden_state,
                 hidden_states=outputs.hidden_states,
                 attentions=outputs.attentions
             )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.clip, name)

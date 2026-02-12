"""
engram_clip.py
Optimized Wrapper for CLIPTextModel to support Vision-Engram injection.
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, List, Union, Tuple

class EngramCLIPWrapper(nn.Module):
    def __init__(self, original_clip: CLIPTextModel, target_ngrams: dict = None, normalization_mode: bool = False, enable_tanh_gating: bool = False):
        super().__init__()
        self.clip = original_clip
        self.config = original_clip.config
        self.normalization_mode = normalization_mode
        self.enable_tanh_gating = enable_tanh_gating
        
        # 1. Freeze Original CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.targets = target_ngrams if target_ngrams is not None else {}
        self.engram_embeddings = nn.ParameterDict()
        
        # Optimization: Pre-register target IDs as buffers
        for name, ids in self.targets.items():
            safe_name = name.replace("-", "_")
            self.register_buffer(f"target_ids_{safe_name}", torch.tensor(ids, dtype=torch.long))

            # Initialize Embedding (Standard BERT/GPT init)
            hidden_size = self.config.hidden_size
            self.engram_embeddings[name] = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
            
        # 2. Vector Scale
        if self.enable_tanh_gating:
             print(" [Model] Tanh Gating Enabled: Scale will be clamped by tanh().")
        
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
            # Optimization: Only iterate possible range
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
        
        # Determine mode: argument overrides instance config
        use_normalization = normalization_mode if normalization_mode is not None else self.normalization_mode
        
        # 1. Run Original CLIP
        outputs = self.clip(input_ids=input_ids, return_dict=True, **kwargs)
        
        last_hidden_state = outputs.last_hidden_state.clone()
        
        matches_log = []
        overlap_counter = torch.zeros(last_hidden_state.shape[:2], device=last_hidden_state.device, dtype=last_hidden_state.dtype)

        # 2. Engram Injection
        if input_ids is not None:
            # First pass: Aggregate all injections (Residual Addition)
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
                        
                        mem = embedding.to(last_hidden_state.dtype) * scale
                        
                        for b, start_idx in matches:
                            last_hidden_state[b, start_idx : start_idx + target_len, :] += mem
                            overlap_counter[b, start_idx : start_idx + target_len] += 1
                            
                            if verbose:
                                token_ids = getattr(self, f"target_ids_{safe_name}").tolist()
                                matches_log.append({
                                    "batch": b,
                                    "pos": start_idx,
                                    "key": name,
                                    "tokens": token_ids,
                                    "scale_mean": scale.mean().item(),
                                    "scale_max": scale.max().item()
                                })
            
            # Second pass: Normalization (Design A)
            if use_normalization:
                original_states = outputs.last_hidden_state.detach()
                delta = last_hidden_state - original_states
                
                # Divisor (Clamp at 1.0)
                divisor = overlap_counter.unsqueeze(-1).clamp(min=1.0)
                
                # Normalize delta
                normalized_delta = delta / divisor
                
                # Re-apply
                last_hidden_state = original_states + normalized_delta
                
                if verbose and overlap_counter.max() > 1:
                     print(f"  [Normalize] Applied overlap normalization. Max overlap: {overlap_counter.max().item()}")

        
        if verbose and matches_log:
            print(f"\n[Engram] Activated {len(matches_log)} triggers:")
            for m in matches_log:
                print(f"  - Key: {m['key']} | Pos: {m['pos']} | Scale(Mean/Max): {m['scale_mean']:.4f}/{m['scale_max']:.4f}")

        # 3. Safe Re-packing
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

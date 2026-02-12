"""
engram_t5.py
Optimized Wrapper for T5EncoderModel to support Vision-Engram injection in SD3.5.
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List, Union, Tuple

class EngramT5Wrapper(nn.Module):
    def __init__(self, original_t5: T5EncoderModel, target_ngrams: dict = None, normalization_mode: bool = False, enable_tanh_gating: bool = False):
        super().__init__()
        self.t5 = original_t5
        self.config = original_t5.config
        self.normalization_mode = normalization_mode
        self.enable_tanh_gating = enable_tanh_gating
        
        # [New] Relative Injection
        self.use_relative_injection = True if enable_tanh_gating else False
        if self.use_relative_injection:
             print(" [Model] T5 Relative Injection Enabled.")
        
        # 1. Freeze Original T5
        for param in self.t5.parameters():
            param.requires_grad = False
            
        self.targets = target_ngrams if target_ngrams is not None else {}
        self.engram_embeddings = nn.ParameterDict()
        
        # Optimize: Register Target IDs
        for name, ids in self.targets.items():
            safe_name = name.replace("-", "_")
            self.register_buffer(f"target_ids_{safe_name}", torch.tensor(ids, dtype=torch.long))

            # Initialize Embedding (Dimension 4096 typically)
            hidden_size = self.config.d_model # T5 uses d_model, not hidden_size
            self.engram_embeddings[name] = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
            
        # 2. Vector Scale
        hidden_size = self.config.d_model
        
        if self.enable_tanh_gating:
             print(f" [Model] T5 Tanh Gating Enabled: Scale ({hidden_size} dim) will be clamped.")
        
        self.injection_scale = nn.Parameter(torch.ones(hidden_size) * 0.05)

    def find_ngram_matches(self, input_ids: torch.Tensor, target_name: str) -> List[Tuple[int, int]]:
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
    ) -> Union[Tuple, BaseModelOutput]:
        
        use_normalization = normalization_mode if normalization_mode is not None else self.normalization_mode
        
        # 1. Run Original T5
        outputs = self.t5(input_ids=input_ids, return_dict=True, **kwargs)
        
        # T5 output is last_hidden_state
        last_hidden_state = outputs.last_hidden_state.clone()
        
        if verbose:
            base_norm = last_hidden_state.norm(p=2, dim=-1).mean().item()
            print(f" [Engram-T5] Base Hidden State Norm: {base_norm:.4f}")

        matches_log = []
        overlap_counter = torch.zeros(last_hidden_state.shape[:2], device=last_hidden_state.device, dtype=last_hidden_state.dtype)

        # 2. Engram Injection
        if input_ids is not None:
             if self.use_relative_injection:
                 # [B, 1, 1]
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
                        
                        if verbose:
                            mem_norm = mem.norm(p=2, dim=-1).mean().item()
                            scale_val = scale.abs().mean().item()
                            print(f"   -> Injecting '{name}': Scale={scale_val:.4f}, VectorNorm={mem_norm:.4f}")

                        for b, start_idx in matches:
                            if self.use_relative_injection:
                                b_norm = base_norm_ref[b] 
                                unit_emb = torch.nn.functional.normalize(embedding.to(last_hidden_state.dtype), p=2, dim=-1)
                                val_to_add = unit_emb * scale * b_norm
                                last_hidden_state[b, start_idx : start_idx + target_len, :] += val_to_add
                            else:
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
            

        
        if verbose and matches_log:
            print(f"\n[Engram T5] Activated {len(matches_log)} triggers:")
            for m in matches_log:
                print(f"  - Key: {m['key']} | Pos: {m['pos']} | Scale(Mean/Max): {m['scale_mean']:.4f}/{m['scale_max']:.4f}")

        # 3. Re-pack
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.t5, name)

"""
registry.py
Manages the mappings for Vision Engram.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import torch

@dataclass
class VisionEngramConfig:
    # Trigger -> Embedding Index or Parameter Name
    # We will use exact token ID sequences for now.
    target_ngrams: Dict[str, List[int]] = field(default_factory=dict)
    model_path: str = "/nasdata/tinyengram/stable-diffusion-1_5"
    injection_scale: float = 0.1
    train_engram_only: bool = True

"""
Model configuration module.
Defines the configuration parameters for the transformation module (TransModule).
"""

from dataclasses import dataclass
from typing import Type
import torch.nn as nn

@dataclass
class TransModuleConfig:
    """
    Configuration class for the transformation module.
    
    Attributes:
        nlayer: Number of Transformer layers
        d_model: Model dimension
        nhead: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to input dimension
        qkv_bias: Whether to use bias in QKV
        attn_drop: Attention dropout rate
        drop: General dropout rate
        drop_path: Drop path rate
        act_layer: Activation function type
        norm_layer: Normalization layer type
        norm_first: Whether to normalize first
    """
    nlayer: int = 3
    d_model: int = 768
    nhead: int = 8
    mlp_ratio: int = 4
    qkv_bias: bool = False
    attn_drop: float = 0.
    drop: float = 0.
    drop_path: float = 0.
    act_layer: Type[nn.Module] = nn.GELU
    norm_layer: Type[nn.Module] = nn.LayerNorm
    norm_first: bool = False

    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.nlayer > 0, "Number of layers must be greater than 0"
        assert self.d_model > 0, "Model dimension must be greater than 0"
        assert self.nhead > 0, "Number of attention heads must be greater than 0"
        assert self.d_model % self.nhead == 0, "Model dimension must be divisible by number of attention heads"
        assert self.mlp_ratio > 0, "MLP ratio must be greater than 0"
        assert 0 <= self.attn_drop <= 1, "Attention dropout rate must be in range [0,1]"
        assert 0 <= self.drop <= 1, "Dropout rate must be in range [0,1]"
        assert 0 <= self.drop_path <= 1, "Drop path rate must be in range [0,1]"

    @classmethod
    def get_default_config(cls) -> 'TransModuleConfig':
        """Get default configuration"""
        return cls()

    def update(self, **kwargs) -> 'TransModuleConfig':
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        self.__post_init__()  # Validate updated parameters
        return self
import torch.nn as nn
import torch
import math
from predictive_coding.pc_layer import PCLayer

class Attention(nn.Module):
    """
    Multi-head self-attention module with predictive coding layers for use in transformer architectures.
    Computes attention scores, applies masking, and outputs context vectors.
    """
    def __init__(self, config):
        """
        Initialize the Attention module.

        Args:
            config: Configuration object with num_heads, n_embed, dropout, T, local_learning_rate, etc.
        """
        super().__init__()

        self.config = config
        self.num_heads = config.num_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.num_heads
        self.dropout = nn.Dropout(config.dropout)

        self.q = nn.Linear(config.n_embed, config.n_embed)
        self.k = nn.Linear(config.n_embed, config.n_embed)
        self.v = nn.Linear(config.n_embed, config.n_embed)
        self.output = nn.Linear(config.n_embed, config.n_embed)

        self.pc_qkv = PCLayer(T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            la = config.la            
        )

        self.pc_output = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )

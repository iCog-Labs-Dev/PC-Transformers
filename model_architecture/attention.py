import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Attention(nn.Module):
    """
    Multi-head self-attention module with predictive coding layers for use in transformer architectures.
    Computes attention scores, applies masking, and outputs context vectors.
    Includes KV caching for efficient generation.
    """
    def __init__(self, config):
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

        self.pc_qkv = PCLayer(
            T=config.T,
            lr=config.lr,
            inference_lr=config.inference_lr,
            update_bias = config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
            num_heads=config.num_heads,
            n_embed=config.n_embed,
        )

        self.pc_output = PCLayer(
            T=config.T,
            lr=config.lr,
            inference_lr=config.inference_lr,
            update_bias = config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
        )
        
        # KV cache for generation: stores (K, V) tensors
        self.kv_cache = None
        
    def clear_kv_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class AttentionProjection(nn.Module):
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

        self.q_projection = nn.Linear(config.n_embed, config.n_embed)
        self.k_projection = nn.Linear(config.n_embed, config.n_embed)
        self.v_projection = nn.Linear(config.n_embed, config.n_embed)
        self.output_projection = nn.Linear(config.n_embed, config.n_embed)

        self.pc_qkv_projection = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=config.optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_sign_value=config.optimizer_sign_value,
            optimizer_weight_bound=config.optimizer_weight_bound,
        )

        self.pc_output_projection = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
            optimizer_name=config.optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_sign_value=config.optimizer_sign_value,
            optimizer_weight_bound=config.optimizer_weight_bound,
        )
        
        # KV cache for generation: stores (K, V) tensors
        self.kv_cache_projection = None
        
    def clear_kv_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None
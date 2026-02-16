import torch
import torch.nn as nn

from predictive_coding.pc_layer import PCLayer


class Attention(nn.Module):
    """Attention module holding projection layers + PC layers.

    Note: the predictive-coding implementation drives the actual attention computation
    through the PC helpers (x_score / x_A / linear_attn). This module mainly
    defines the learnable projections (q/k/v and output) and stores the KV cache.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed
        self.num_heads = config.num_heads

        if self.n_embed % self.num_heads != 0:
            raise ValueError("n_embed must be divisible by num_heads")

        # Standard projections
        self.q = nn.Linear(config.n_embed, config.n_embed, bias=True)
        self.k = nn.Linear(config.n_embed, config.n_embed, bias=True)
        self.v = nn.Linear(config.n_embed, config.n_embed, bias=True)

        # Score/A "layers" (kept as explicit modules so x_score/x_A can do mu = layer(x))
        # Default to Identity; callers can replace with something else if desired.
        self.score = nn.Identity()
        self.A = nn.Identity()

        # Output projection (kept for compatibility with existing init/update code)
        self.output = nn.Linear(config.n_embed, config.n_embed, bias=True)

        # PC layers: q/k/v/score/A live in pc_qkv; attention output lives in pc_output
        self.pc_qkv = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
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

        self.pc_output = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
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

        self.kv_cache = None  # (K_total, V_total) in shape (B, kv_len, n_embed)

    def clear_kv_cache(self):
        self.kv_cache = None

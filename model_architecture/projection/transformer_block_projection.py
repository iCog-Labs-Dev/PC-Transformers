import torch.nn as nn
from .attention_projection import AttentionProjection
from .mlp_projection import MLPProjection

class TransformerBlockProjection(nn.Module):
    """
    A single block of the Transformer architecture, consisting of layer normalization, attention, and MLP submodules.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embed)
        self.attn = AttentionProjection(config)
        self.ln2 = nn.RMSNorm(config.n_embed)
        self.mlp = MLPProjection(config)

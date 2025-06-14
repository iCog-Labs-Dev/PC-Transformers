import torch.nn as nn
from .attention import Attention
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    
    def evaluate(self, x):
        x = self.ln1(x)
        x = self.attn.evaluate(x)
        x = self.ln2(x)
        x = self.mlp.evaluate(x)

        return x

import torch.nn as nn
from .attention import Attention
from .mlp import MLP

class TransformerBlock(nn.Module):
    """
    A single block of the Transformer architecture, consisting of layer normalization, attention, and MLP submodules.
    """
    def __init__(self, config):
        """
        Initialize the TransformerBlock.

        Args:
            config: Configuration object containing model hyperparameters (e.g., n_embed).
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def evaluate(self, x):
        """
        Evaluate the Transformer block in inference mode (no predictive coding).

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embed).
        Returns:
            torch.Tensor: Output tensor after attention and MLP layers.
        """
        x = self.ln1(x)
        x = self.attn.evaluate(x)
        x = self.ln2(x)
        x = self.mlp.evaluate(x)

        return x
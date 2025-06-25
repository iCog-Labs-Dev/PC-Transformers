import torch
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayer(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        """
        Initialize the OutputLayer.

        Args:
            config: Configuration object with n_embed, vocab_size, T, local_learning_rate, etc.
        """
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        self.pc_layer = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )

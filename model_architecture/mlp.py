import torch.nn as nn
from predictive_coding.pc_layer import PCLayer


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used within the transformer architecture.
    Includes two linear layers and two predictive coding layers for local learning.
    """

    def __init__(self, config):
        """
        Initialize the MLP block.

        Args:
            config: Configuration object with n_embed, T, local_learning_rate, etc.
        """
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        # self.dropout = nn.Dropout(config.dropout)

        self.pc_layer2 = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias=config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )

        self.pc_layer1 = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias=config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )

    def evaluate(self, x):
        """
        Evaluate the MLP block in inference mode (no predictive coding).

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embed).
        Returns:
            torch.Tensor: Output tensor after two linear layers.
        """
        fc_1 = self.fc1(x)
        fc_2 = self.fc2(fc_1)

        return fc_2

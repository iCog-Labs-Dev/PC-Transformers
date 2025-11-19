import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used within the transformer architecture.
    Includes two linear layers and two predictive coding layers for local learning.
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

        self.pc_layer2 = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
        )

        self.pc_layer1 = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
        )

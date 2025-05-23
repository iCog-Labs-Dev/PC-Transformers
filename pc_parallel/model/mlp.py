import torch
import torch.nn as nn
import torch.nn.functional as F
from .pclayer import PCLayer

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

        self.pc_layer2 = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error
        )

        self.pc_layer1 = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error
        )

    def forward(self, output_x, batch_size, seq_len):
        self.pc_layer2.clear_energy()
        self.pc_layer2.clear_errors()
        self.pc_layer2.init_x(batch_size, seq_len, layer=self.fc2, kind="fc2")

        self.pc_layer1.clear_energy()
        self.pc_layer1.clear_errors()
        self.pc_layer1.init_x(batch_size, seq_len, layer=self.fc1, kind="fc1")

        return self.pc_layer1.get_x("fc1")

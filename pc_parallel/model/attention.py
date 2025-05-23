import torch
import torch.nn as nn
from .pclayer import PCLayer

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embed
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.pc_qkv = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error
        )

        self.pc_output = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error
        )

    def forward(self, fc1_x, batch_size, seq_len):
        self.pc_output.clear_energy()
        self.pc_output.clear_errors()
        self.pc_output.init_x(batch_size, seq_len, layer=self.output, kind="output_attn")

        self.pc_qkv.clear_energy()
        self.pc_qkv.clear_errors()
        self.pc_qkv.init_x(batch_size, seq_len, q_proj=self.query, k_proj=self.key, v_proj=self.value, kind="attn")

        return self.pc_qkv.get_x("attn")

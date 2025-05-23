import torch
import torch.nn as nn
from .pclayer import PCLayer

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.pc_layer = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error
        )
        self.LayerNorm = nn.LayerNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x_qkv, batch_size, seq_len, input_ids, position_ids):
        self.pc_layer.clear_energy()
        self.pc_layer.clear_errors()
        self.pc_layer.init_x(
            batch_size, seq_len,
            layer={"word": self.word_embeddings, "pos": self.position_embeddings},
            kind="embed",
            input_ids=input_ids,
            position_ids=position_ids
        )
        return None

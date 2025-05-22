import torch.nn as nn
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = Embedding_Layer(
            vocab_size=config.vocab_size,
            embedding_dim=config.n_embd,
            max_pos_embed=config.block_size,
            pad_token_id=0
        )

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.embedding.word_embeddings.weight

        self.drop = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.embedding.position_embeddings.num_embeddings, "Input too long"

        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
import torch.nn as nn
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from .transformer_utils import ids_to_one_hot

class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)

    def forward(self, target_ids):
        target_ids = ids_to_one_hot(target_ids, self.output.config.vocab_size)
        output_x = self.output(target_ids)

        prev_qkv = output_x
        for block in reversed(self.blocks):
            x_qkv = block(prev_qkv)
            prev_qkv = x_qkv

        logits = output_x @ self.output.output.weight.T + self.output.output.bias
        return logits
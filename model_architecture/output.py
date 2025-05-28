import torch
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        self.pc_layer = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias
        )


    def evaluate(self, x):
        output = self.output(x)
        return output

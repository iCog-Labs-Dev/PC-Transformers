# pc_transformer/config.py
from typing import Optional, Callable

class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        dropout: float = 0.1,
        energy_fn: Optional[Callable] = None,
        x_init: Optional[Callable] = None,
        local_learning_rate: float = 1e-3,
        T: int = 10,
        is_holding_error: bool = False,
        num_heads: int = 12,
        n_blocks: int = 4,
        batch_size: int = 8,
        num_epochs: int = 5
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.dropout = dropout
        self.energy_fn = energy_fn
        self.x_init = x_init
        self.local_learning_rate = local_learning_rate
        self.T = T
        self.is_holding_error = is_holding_error
        self.num_heads = num_heads
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.num_epochs = num_epochs

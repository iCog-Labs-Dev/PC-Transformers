from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embed: int
    dropout: float = 0.1
    local_learning_rate: float = 1e-3
    T: int = 10
    is_holding_error: bool = False
    update_bias: bool = True
    num_heads: int = 12
    n_blocks: int = 4
    batch_size: int = 8
    num_epochs: int = 5
<<<<<<< HEAD
    energy_fn_name: str = "scaled_mse" 
    use_lateral: bool = True
=======
>>>>>>> parent of af66fee (fixed some errors after the addition of latent connections.)

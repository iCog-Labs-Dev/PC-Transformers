from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    la: float=0.5
    n_embed: int =516
    dropout: float = 0.1
    local_learning_rate: float = 1e-5
    T: int = 10
    is_holding_error: bool = False
    update_bias: bool = True
    num_heads: int = 12
    n_blocks: int = 4
    batch_size: int = 8
    num_epochs: int = 5
    use_lateral: bool = True
    energy_fn_name: str = "scaled_mse"

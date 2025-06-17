from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embed: int =64
    dropout: float = 0.1
    local_learning_rate: float = 1e-3
    T: int = 10
    is_holding_error: bool = False
    update_bias: bool = True
    num_heads: int = 2
    n_blocks: int = 4
    batch_size: int = 8
    num_epochs: int = 5
    use_lateral: bool = True
    energy_fn_name: str = "scaled_mse"
    eos_token_id: int = None
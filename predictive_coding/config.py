from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    """
    Configuration dataclass for the predictive coding transformer model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_embed (int): Embedding dimension size.
        dropout (float): Dropout probability.
        lr (float): Local learning rate for predictive coding layers.
        peak_learning_rate (float): Peak learning rate for learning rate scheduling.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        T (int): Number of inference steps for predictive coding.
        update_bias (bool): Whether to update bias terms during learning.
        num_heads (int): Number of attention heads.
        n_blocks (int): Number of transformer blocks.
        batch_size (int): Batch size for training/evaluation.
        num_epochs (int): Number of training epochs.
        energy_fn_name (str): Name of the energy function to use for error computation.
        use_flash_attention (bool): Whether to use FlashAttention.
        optimizer_name (str): Optimizer to use for PC weight updates (adam or sgd).
        optimizer_beta1 (float): Adam beta1 coefficient.
        optimizer_beta2 (float): Adam beta2 coefficient.
        optimizer_eps (float): Adam epsilon.
        optimizer_sign_value (float): Sign applied to updates (-1 for ascent).
        optimizer_weight_bound (float): Optional weight clamp bound (0 disables).
    """
    vocab_size: int
    block_size: int
    lr: float
    peak_learning_rate: Optional[float]
    warmup_steps: Optional[int] 
    n_embed: int 
    dropout: float 
    T: int 
    update_bias: bool 
    num_heads: int 
    n_blocks: int 
    batch_size: int
    num_epochs: int
    internal_energy_fn_name:str
    output_energy_fn_name: str
    combined_internal_weight: float 
    combined_output_weight: float
    use_flash_attention: bool
    alpha: float
    optimizer_name: str
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_eps: float
    optimizer_sign_value: float
    optimizer_weight_bound: float
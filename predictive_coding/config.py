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
    """
    vocab_size: int
    block_size: int
    lr: float
    peak_learning_rate: Optional[float]
    warmup_steps: Optional[int] 
    n_embed: int 
    dropout: float 
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
    # Per-layer T values for iterative inference
    embed_T: int = 1
    attn_T: int = 1
    linear_attn_T: int = 1
    fc1_T: int = 1
    fc2_T: int = 1
    linear_output_T: int = 1
    lambda_compute: float = 0.0
    monotonic_penalty_weight: float = 1.0
    min_energy_drop: float = 0.05
    drop_penalty_weight: float = 5.0
    min_ppl_drop: float = 20.0
    ppl_drop_penalty_weight: float = 1.0
   
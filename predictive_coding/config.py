from dataclasses import dataclass
from typing import Optional

"""
predictive_coding.config

This module defines the GPTConfig dataclass, which holds configuration parameters for the predictive coding transformer model.
"""

@dataclass
class GPTConfig:
    """
    Configuration dataclass for the predictive coding transformer model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_embed (int): Embedding dimension size.
        dropout (float): Dropout probability.
        local_learning_rate (float): Local learning rate for predictive coding layers.
        peak_learning_rate (float): Peak learning rate for learning rate scheduling.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        T (int): Number of inference steps for predictive coding.
        is_holding_error (bool): Whether to accumulate and store errors.
        update_bias (bool): Whether to update bias terms during learning.
        num_heads (int): Number of attention heads.
        n_blocks (int): Number of transformer blocks.
        batch_size (int): Batch size for training/evaluation.
        num_epochs (int): Number of training epochs.
        use_lateral (bool): Whether to use lateral (recurrent) connections.
        energy_fn_name (str): Name of the energy function to use for error computation.
        use_flash_attention (bool): Whether to use FlashAttention.
    """
    vocab_size: int
    block_size: int
    local_learning_rate: float
    peak_learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None
    la: float=0.5
    n_embed: int =208
    dropout: float = 0.1
    T: int = 10
    is_holding_error: bool = False
    update_bias: bool = True
    num_heads: int = 16
    n_blocks: int = 4
    batch_size: int = 8
    num_epochs: int = 5
    use_lateral: bool = True
    internal_energy_fn_name:str="pc_e",
    attn_energy_fn_name: str="kld",
    output_energy_fn_name:str="ce",
    eos_token_id: int = None
    use_flash_attention: bool = False
    combined_internal_weight: float = 0.3
    combined_output_weight: float = 0.7
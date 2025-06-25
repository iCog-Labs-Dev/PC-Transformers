from dataclasses import dataclass

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
        T (int): Number of inference steps for predictive coding.
        is_holding_error (bool): Whether to accumulate and store errors.
        update_bias (bool): Whether to update bias terms during learning.
        num_heads (int): Number of attention heads.
        n_blocks (int): Number of transformer blocks.
        batch_size (int): Batch size for training/evaluation.
        num_epochs (int): Number of training epochs.
        use_lateral (bool): Whether to use lateral (recurrent) connections.
        energy_fn_name (str): Name of the energy function to use for error computation.
    """
    vocab_size: int
    block_size: int
    la: float=0.5
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
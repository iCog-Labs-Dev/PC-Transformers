import torch
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig

def test_pc_transformer_forward():
    """
    Test the forward pass of the PCTransformer model.
    Ensures that the model produces output of the expected shape given random input tensors.
    """
    config = GPTConfig(
        vocab_size=100,
        block_size=8,
        n_embed=16,
        dropout=0.1,
        local_learning_rate=1e-3,
        T=2,
        is_holding_error=True,
        num_heads=2,
        n_blocks=1,
        num_epochs=1,
        update_bias=False,
        energy_fn_name="scaled_mse",
        eos_token_id=99
    )
    model = PCTransformer(config)
    x = torch.randint(0, 100, (2, 8))
    y = model(x, x)
    assert y.shape == (2, 8, 100)

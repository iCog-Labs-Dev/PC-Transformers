def make_test_config(**overrides):
    base = dict(
        vocab_size=20,
        block_size=5,
        n_embed=8,
        dropout=0.1,
        local_learning_rate=1e-3,
        T=2,
        is_holding_error=True,
        num_heads=2,
        n_blocks=1,
        num_epochs=1,
        update_bias=False,
        energy_fn_name="scaled_mse",
        eos_token_id=19
    )
    base.update(overrides)
    from predictive_coding.config import GPTConfig
    return GPTConfig(**base)

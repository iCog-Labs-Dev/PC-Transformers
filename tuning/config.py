import logging
from predictive_coding.config import GPTConfig

logger = logging.getLogger(__name__)

def get_dynamic_model_config(trial, vocab_size, flash=False):
    """Get model configuration with dynamic parameter combinations, including flash attention flag."""
    n_embed = trial.suggest_int("n_embed", 64, 384, step=16)

    valid_heads = [h for h in range(2, min(32, n_embed // 8) + 1) if n_embed % h == 0 and 8 <= n_embed // h <= 128]
    if not valid_heads:
        logger.warning(f"No valid heads for n_embed={n_embed}, forcing fallback.")
        return None
        
    num_heads = valid_heads[trial.suggest_int('head_idx', 0, len(valid_heads) - 1)]
    block_size = trial.suggest_int("block_size", 64, 256, step=16)
    n_blocks = trial.suggest_int('n_blocks', 1, 8)
    # Per-layer T values
    embed_T = trial.suggest_int('embed_T', 3, 12, log=True)
    attn_T = trial.suggest_int('attn_T', 3, 12, log=True)
    linear_attn_T = trial.suggest_int('linear_attn_T', 3, 12, log=True)
    fc1_T = trial.suggest_int('fc1_T', 3, 12, log=True)
    fc2_T = trial.suggest_int('fc2_T', 3, 12, log=True)
    linear_output_T = trial.suggest_int('linear_output_T', 3, 12, log=True)
    lambda_compute = trial.suggest_float('lambda_compute', 1e-6, 3e-5, log=True)
    monotonic_penalty_weight = 1.0
    ppl_monotonic_penalty_weight = 1.0
    min_energy_drop = 0.08
    drop_penalty_weight = 5.0
    min_ppl_drop = 20.0
    ppl_drop_penalty_weight = 1.0
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    peak_lr = trial.suggest_float('peak_lr', 1e-5, 2e-3, log=True)
    lr = peak_lr * 0.1 
    warmup_steps = trial.suggest_int('warmup_steps', 50, 2000, log=True)
    update_bias = trial.suggest_int('update_bias_int', 0, 1) == 1
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])

    # Guardrail to skip memory-heavy combinations before model construction.
    per_block_t = attn_T + linear_attn_T + fc1_T + fc2_T
    complexity_score = batch_size * block_size * n_embed * n_blocks * per_block_t
    if complexity_score > 80_000_000:
        logger.warning(
            "Skipping trial due to estimated memory pressure: "
            f"score={complexity_score}, bs={batch_size}, block={block_size}, embed={n_embed}, blocks={n_blocks}, t={per_block_t}"
        )
        return None
    combined_internal_weight = trial.suggest_float('combined_internal_weight', 0.1, 0.9)
    combined_output_weight = 1.0 - combined_internal_weight
    num_epochs = 3
    alpha = 0.5
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        peak_learning_rate=peak_lr,
        warmup_steps=warmup_steps,
        n_embed=n_embed,
        dropout=dropout,
        lr=lr, 
        embed_T=embed_T,
        attn_T=attn_T,
        linear_attn_T=linear_attn_T,
        fc1_T=fc1_T,
        fc2_T=fc2_T,
        linear_output_T=linear_output_T,
        lambda_compute=lambda_compute,
        monotonic_penalty_weight=monotonic_penalty_weight,
        ppl_monotonic_penalty_weight=ppl_monotonic_penalty_weight,
        min_energy_drop=min_energy_drop,
        drop_penalty_weight=drop_penalty_weight,
        min_ppl_drop=min_ppl_drop,
        ppl_drop_penalty_weight=ppl_drop_penalty_weight,
        num_heads=num_heads,
        n_blocks=n_blocks,
        batch_size = batch_size,
        num_epochs=num_epochs,
        update_bias=update_bias,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight = combined_internal_weight,
        combined_output_weight = combined_output_weight,
        use_flash_attention=flash,
        alpha=alpha
    )

def update_global_config(config):
    """Update global GPTConfig"""
    config_keys = [
        'num_heads', 'n_embed', 'block_size', 'n_blocks', 'vocab_size',
        'dropout', 'lr', 'peak_learning_rate', 'warmup_steps',
        'update_bias', 'internal_energy_fn_name', 'output_energy_fn_name',
        'batch_size', 'num_epochs', 'combined_internal_weight', 
        'combined_output_weight', 'alpha', 'embed_T', 'attn_T',
        'linear_attn_T', 'fc1_T', 'fc2_T', 'linear_output_T', 'lambda_compute',
        'monotonic_penalty_weight', 'ppl_monotonic_penalty_weight',
        'min_energy_drop', 'drop_penalty_weight',
        'min_ppl_drop', 'ppl_drop_penalty_weight'
    ]
    
    for key in config_keys:
        try:
            if isinstance(config, dict):
                if key in config:
                    setattr(GPTConfig, key, config[key])
            elif hasattr(config, key):
                setattr(GPTConfig, key, getattr(config, key))
        except Exception as e:
            logger.warning(f"Failed to update config key '{key}': {e}")
            continue
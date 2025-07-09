import logging
from predictive_coding.config import GPTConfig

logger = logging.getLogger(__name__)

def get_dynamic_model_config(trial, vocab_size):
    """Get model configuration with dynamic parameter combinations"""
    n_embed = trial.suggest_int("n_embed", 64, 768, step=16)

    valid_heads = [h for h in range(4, min(16, n_embed // 12) + 1) 
                   if n_embed % h == 0 and 12 <= n_embed // h <= 128]
    if not valid_heads:
        logger.warning(f"No valid heads for n_embed={n_embed}, forcing fallback.")
        return None
        
    num_heads = valid_heads[trial.suggest_int('head_idx', 0, len(valid_heads) - 1)]
    block_size = trial.suggest_int("block_size", 64, 384, step=32)
    n_blocks = trial.suggest_int('n_blocks', 1, 6)
    T = trial.suggest_int('T', 4, 20)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    base_lr = trial.suggest_float('base_lr', 1e-5, 1e-3, log=True)

    warmup_steps = trial.suggest_int('warmup_steps', 16, block_size)
    
    energy_fn_name = ['kld', 'mse', 'scaled_mse'][trial.suggest_int('energy_idx', 0, 2)]
    update_bias = trial.suggest_int('update_bias_int', 0, 1) == 1
    scaled_lr = base_lr * (n_embed / 256) ** 0.5 * (block_size / 256) ** 0.25
    scaled_lr = max(min(scaled_lr, 1e-2), 1e-6) 
    eos_token_id = 0

    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        peak_learning_rate=scaled_lr,
        warmup_steps=warmup_steps,
        n_embed=n_embed,
        dropout=dropout,
        local_learning_rate=0.0, 
        T=T,
        is_holding_error=True,
        num_heads=num_heads,
        n_blocks=n_blocks,
        num_epochs=3,
        update_bias=update_bias,
        use_lateral=True,
        energy_fn_name=energy_fn_name,
        eos_token_id = eos_token_id
    )

def update_global_config(config):
    """Update global GPTConfig"""
    config_keys = [
        'num_heads', 'n_embed', 'block_size', 'n_blocks', 'vocab_size',
        'dropout', 'local_learning_rate', 'peak_learning_rate', 'warmup_steps',
        'update_bias', 'energy_fn_name', 'use_lateral',
        'T', 'is_holding_error'
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

def normalize_energy(energy_value, energy_fn_name):
    """ Normalize energy values to comparable scales across different energy functions."""
    factors = {'mse': 1.0, 'scaled_mse': 20.0, 'kld': 0.2}
    return energy_value * factors.get(energy_fn_name, 1.0)
import logging
from predictive_coding.config import GPTConfig

logger = logging.getLogger(__name__)

def get_dynamic_model_config(trial, vocab_size, flash=False):
    """Get model configuration with dynamic parameter combinations, including flash attention flag."""
    # Use fixed parameters as requested
    n_embed = 208
    num_heads = 4
    block_size = 368
    n_blocks = 8
    embed_T = 1
    attn_T = 1
    linear_attn_T = 2
    fc1_T = 1
    fc2_T = 1
    linear_output_T = 1
    dropout = 0.19739793108863762
    peak_lr = 9.97623125949041e-05
    lr = 9.976231259490411e-06
    warmup_steps = 545
    update_bias = True
    batch_size = 4
    combined_internal_weight = 0.17255768114691322
    combined_output_weight = 0.8274423188530868
    num_epochs = 5
    alpha = 0.5
    return GPTConfig(
        vocab_size=1024,
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
        num_heads=num_heads,
        n_blocks=n_blocks,
        batch_size=batch_size,
        num_epochs=num_epochs,
        update_bias=update_bias,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight=combined_internal_weight,
        combined_output_weight=combined_output_weight,
        use_flash_attention=False,
        alpha=alpha
    )

def update_global_config(config):
    """Update global GPTConfig"""
    config_keys = [
        'num_heads', 'n_embed', 'block_size', 'n_blocks', 'vocab_size',
        'dropout', 'lr', 'peak_learning_rate', 'warmup_steps',
        'update_bias', 'internal_energy_fn_name', 'output_energy_fn_name',
        'batch_size', 'num_epochs', 'combined_internal_weight', 
        'combined_output_weight', 'alpha'
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
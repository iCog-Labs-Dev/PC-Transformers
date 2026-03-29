import os
import re

LAYER_T_KEYS = (
    "embed_T",
    "attn_T",
    "linear_attn_T",
    "fc1_T",
    "fc2_T",
    "linear_output_T",
)

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "embed_T", "attn_T", "linear_attn_T", "fc1_T", "fc2_T", "linear_output_T", "num_heads", "n_blocks", "update_bias",
        "lr", "batch_size", "num_epochs", "internal_energy_fn_name",
        "output_energy_fn_name", "combined_internal_weight",
        "combined_output_weight", "use_flash_attention"
    }

    fallback_values = {
        "block_size": 496,
        "peak_learning_rate": 0.005385160427135375,
        "warmup_steps": 55,
        "n_embed": 80,
        "dropout": 0.03047632604311483,
        
        "embed_T": 13,
        "attn_T": 3,
        "linear_attn_T": 8,
        "fc1_T": 13,
        "fc2_T": 13,
        "linear_output_T": 3,
        "num_heads": 2,
        "n_blocks": 1,
        "update_bias": False,
        "lr": 0.0005385160427135375,
        "batch_size": 32,
        "num_epochs": 3,
        "internal_energy_fn_name": "pc_e",
        "output_energy_fn_name": "pc_e",
        "combined_internal_weight": 0.7774452929376792,
        "combined_output_weight": 0.22255470706232083,
        "use_flash_attention": False
    }

    config = {}
    legacy_t = None
    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "bayesian_tuning_results.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        for line in content.splitlines():
            match = re.match(r'(\w+):\s+(.*)', line)
            if match:
                key, value = match.groups()
                if key == "T":
                    try:
                        legacy_t = int(float(value))
                    except ValueError:
                        legacy_t = None
                if key in selected_keys:
                    try:
                        num = float(value)
                        config[key] = int(num) if num.is_integer() else num
                    except ValueError:
                        # Handle booleans
                        if value.lower() in {"true", "false"}:
                            config[key] = value.lower() == "true"
                        else:
                            # Keep as string
                            config[key] = value.strip('"').strip("'")
    else:
        print(f"[WARNING] Tuning result file not found: {file_path}")
        print(f"[INFO] Using fallback values for missing keys: {selected_keys - config.keys()}")

    if legacy_t is not None:
        for layer_t_key in LAYER_T_KEYS:
            config.setdefault(layer_t_key, legacy_t)

    # Fill in missing keys from fallback
    for key in selected_keys:
        if key not in config:
            config[key] = fallback_values[key]

    return config

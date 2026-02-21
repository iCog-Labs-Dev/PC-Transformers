import os
import re

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "T", "num_heads", "n_blocks", "update_bias", "alpha",
        "lr", "inference_lr", "batch_size", "num_epochs", "internal_energy_fn_name",
        "output_energy_fn_name", "combined_internal_weight",
        "combined_output_weight", "use_flash_attention"
    }

    fallback_values = {
        "block_size": 64,
        "peak_learning_rate": 0.009606017304857476,
        "warmup_steps": 59,
        "n_embed": 512,
        "dropout": 0.46876145412214615,
        "T": 2,
        "num_heads": 32,
        "n_blocks": 12,
        "update_bias": False,
        "alpha": 0.5,
        "lr": 0.0009606017304857476,
        "inference_lr": 0.01,
        "batch_size": 8,
        "num_epochs": 10,
        "internal_energy_fn_name": "pc_e",
        "output_energy_fn_name": "pc_e",
        "combined_internal_weight": 0.8779955579743048,
        "combined_output_weight": 0.12200444202569516,
        "use_flash_attention": False
    }

    config = {}
    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "bayesian_tuning_results.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        for line in content.splitlines():
            match = re.match(r'(\w+):\s+(.*)', line)
            if match:
                key, value = match.groups()
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
        

    # Fill in missing keys from fallback
    for key in selected_keys:
        if key not in config:
            config[key] = fallback_values[key]

    return config

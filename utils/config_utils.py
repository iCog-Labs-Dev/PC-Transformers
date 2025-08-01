import os
import re

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "T", "num_heads", "n_blocks",
        "energy_fn_name", "update_bias"
    }

    fallback_values = {
        "block_size": 352,
        "peak_learning_rate": 0.0006870706728235172,
        "warmup_steps": 211,
        "n_embed": 512,
        "dropout": 0.11081845099183116,
        "T": 15,
        "num_heads": 8,
        "n_blocks": 4,
        "energy_fn_name": "mse",
        "update_bias": False
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
                        # Convert booleans
                        if value.lower() in {"true", "false"}:
                            config[key] = value.lower() == "true"
                        # Convert numeric types
                        elif '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        # Handle string values
                        config[key] = value.strip('"').strip("'")
    else:
        print(f"[WARNING] Tuning result file not found: {file_path}")
        print(f"[INFO] Using fallback values for missing keys: {selected_keys - config.keys()}")
        

    # Fill in missing keys from fallback
    for key in selected_keys:
        if key not in config:
            config[key] = fallback_values[key]

    return config

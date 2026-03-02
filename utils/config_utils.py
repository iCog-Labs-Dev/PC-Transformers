import os
import re

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "vocab_size", "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "T", "num_heads", "n_blocks", "update_bias", "alpha",
        "lr", "batch_size", "num_epochs", "internal_energy_fn_name",
        "output_energy_fn_name", "combined_internal_weight",
        "combined_output_weight", "use_flash_attention",
        "embed_T", "attn_T", "linear_attn_T", "fc1_T", "fc2_T", "linear_output_T"
    }

    fallback_values = {
        "vocab_size": 1024,
        "block_size": 368,
        "peak_learning_rate": 9.97623125949041e-05,
        "warmup_steps": 545,
        "n_embed": 208,
        "dropout": 0.19739793108863762,
        "T": 1,
        "num_heads": 4,
        "n_blocks": 8,
        "update_bias": True,
        "alpha": 0.5,
        "lr": 9.976231259490411e-06,
        "batch_size": 4,
        "num_epochs": 5,
        "internal_energy_fn_name": "pc_e",
        "output_energy_fn_name": "pc_e",
        "combined_internal_weight": 0.17255768114691322,
        "combined_output_weight": 0.8274423188530868,
        "use_flash_attention": False, 
        "embed_T": 1,
        "attn_T": 1,
        "linear_attn_T": 2,
        "fc1_T": 1,
        "fc2_T": 1,
        "linear_output_T": 1
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
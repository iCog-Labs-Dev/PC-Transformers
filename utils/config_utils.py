import os
import re

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "vocab_size", "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "num_heads", "n_blocks", "update_bias", "alpha",
        "lr", "batch_size", "num_epochs", "internal_energy_fn_name",
        "output_energy_fn_name", "combined_internal_weight",
        "combined_output_weight", "use_flash_attention",
        "embed_T", "attn_T", "linear_attn_T", "fc1_T", "fc2_T", "linear_output_T"
    }

    fallback_values = {
        "vocab_size": 1024,
        "block_size": 208,
        "peak_learning_rate": 3.688485170315793e-05,
        "warmup_steps": 1090,
        "n_embed": 384,
        "dropout": 0.4166152568090924,

        "num_heads": 24,
        "n_blocks": 8,
        "update_bias": True,
        "alpha": 0.5,
        "lr": 3.6884851703157935e-06,
        "batch_size": 32,
        "num_epochs": 5,
        "internal_energy_fn_name": "pc_e",
        "output_energy_fn_name": "pc_e",
        "combined_internal_weight": 0.7025752444778229,
        "combined_output_weight": 0.29742475552217706,
        "use_flash_attention": False,
        "embed_T": 9,
        "attn_T": 7,
        "linear_attn_T": 9,
        "fc1_T": 13,
        "fc2_T": 14,
        "linear_output_T": 4
    }

    config = {}
    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "bayesian_tuning_results.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        for line in content.splitlines():
            match = re.match(r'(\w+)\s*[:=]\s*(.*)', line)
            if match:
                key, value = match.groups()
                if key in selected_keys:
                    try:
                        num = float(value)
                        config[key] = int(num) if num.is_integer() else num
                    except ValueError:
                        if value.lower() in {"true", "false"}:
                            config[key] = value.lower() == "true"
                        else:
                            config[key] = value.strip('"').strip("'")
    else:
        print(f"[WARNING] Tuning result file not found: {file_path}")
        print(f"[INFO] Using fallback values for missing keys: {selected_keys - config.keys()}")

    # Fill in missing keys from fallback
    for key in selected_keys:
        if key not in config:
            config[key] = fallback_values[key]

    return config
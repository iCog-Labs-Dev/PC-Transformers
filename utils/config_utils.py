import os
import re

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "T", "num_heads", "n_blocks", "alpha",
        "lr", "inference_lr", "batch_size", "num_epochs", "internal_energy_fn_name",
        "output_energy_fn_name", "combined_internal_weight",
        "combined_output_weight", "use_flash_attention",
        "optimizer_name", "output_optimizer_name", "optimizer_beta1", "optimizer_beta2", "optimizer_eps",
        "optimizer_weight_decay", "optimizer_momentum"
    }

    fallback_values = {
       "block_size": 208,
        "peak_learning_rate": 0.003223786832283688,
        "warmup_steps": 369,
        "n_embed": 160,
        "dropout": 0.46876145412214615,
        "T": 20,
        "num_heads": 10,
        "n_blocks": 3,
        "alpha": 0.5,
        "lr": 0.003223786832283688,
        "inference_lr": 0.096,
        "batch_size": 8,
        "num_epochs": 20,
        "internal_energy_fn_name": "pc_e",
        "output_energy_fn_name": "kld",
        "combined_internal_weight": 0.8779955579743048,
        "combined_output_weight": 0.12200444202569516,
        "use_flash_attention": False,
        "optimizer_name": "sgd",
        "output_optimizer_name": "adam",
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0.01,
        "optimizer_momentum": 0.9, #for sgd_momentum
    }

    config = {}
    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "best_hyperparameters.txt")

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

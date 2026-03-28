import os
import re
from typing import Any, Dict, Mapping, Optional

from predictive_coding.config import GPTConfig

CONFIG_DEFAULTS: Dict[str, Any] = {
    "block_size": 64,
    "peak_learning_rate": 0.009606017304857476,
    "warmup_steps": 59,
    "n_embed": 512,
    "dropout": 0.46876145412214615,
    "max_steps": 2,
    "num_heads": 32,
    "n_blocks": 12,
    "update_bias": False,
    "alpha": 0.5,
    "lr": 0.0009606017304857476,
    "batch_size": 8,
    "num_epochs": 10,
    "internal_energy_fn_name": "pc_e",
    "output_energy_fn_name": "pc_e",
    "combined_internal_weight": 0.8779955579743048,
    "combined_output_weight": 0.12200444202569516,
    "use_flash_attention": False,
    "convergence_threshold": 0.01,
    "healthy_energy_threshold": 0.0,
    "min_steps": 2,
}

LEGACY_KEY_MAP = {"T": "max_steps"}


def _parse_scalar(value: str) -> Any:
    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except ValueError:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value.strip('"').strip("'")


def normalize_config_dict(
    config: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return a canonical config dict using max_steps and adaptive-t defaults."""
    merged = dict(CONFIG_DEFAULTS)
    source = dict(config or {})

    for legacy_key, canonical_key in LEGACY_KEY_MAP.items():
        if canonical_key not in source and legacy_key in source:
            source[canonical_key] = source[legacy_key]

    for key in CONFIG_DEFAULTS:
        if key in source:
            merged[key] = source[key]

    for key, value in source.items():
        if key not in CONFIG_DEFAULTS and key not in LEGACY_KEY_MAP:
            merged[key] = value

    for key, value in overrides.items():
        if key in LEGACY_KEY_MAP:
            merged[LEGACY_KEY_MAP[key]] = value
        else:
            merged[key] = value

    merged.pop("T", None)
    return merged


def build_gpt_config(
    config: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> GPTConfig:
    """Build GPTConfig from canonicalized config values."""
    return GPTConfig(**normalize_config_dict(config, **overrides))


def load_best_config() -> Dict[str, Any]:
    """
    Parse the saved tuning result file and return canonical config values.
    Legacy T values are translated to max_steps here.
    """
    selected_keys = set(CONFIG_DEFAULTS) | set(LEGACY_KEY_MAP)
    config: Dict[str, Any] = {}
    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "bayesian_tuning_results.txt")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()

        for line in content.splitlines():
            match = re.match(r"(\w+):\s+(.*)", line)
            if not match:
                continue

            key, raw_value = match.groups()
            if key not in selected_keys:
                continue

            parsed_value = _parse_scalar(raw_value)
            canonical_key = LEGACY_KEY_MAP.get(key, key)
            config[canonical_key] = parsed_value
    else:
        print(f"[WARNING] Tuning result file not found: {file_path}")
        print("[INFO] Using fallback config values.")

    return normalize_config_dict(config)

def initialize_logs(study_name: str):
    """Create and initialize summary and trial log files."""
    summary_path = f"{study_name}_summary.txt"
    trials_path = f"{study_name}_trials.txt"

    with open(summary_path, "w") as f:
        f.write(f"BAYESIAN TUNING SUMMARY - {study_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write("Objective: Minimize combined energy (normalized internal energy + CE loss)\n\n")
        f.write("Trial Progress:\n")
        f.write(f"{'Trial':<6} {'Time(s)':<8} {'CE Loss':<10} {'Raw Energy':<12} "
                f"{'Norm Energy':<12} {'Combined':<12} {'Energy Fn':<12}\n")
        f.write(f"{'-'*82}\n")

    with open(trials_path, "w") as f:
        f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Objective: Minimize combined energy (normalized internal energy + CE loss)\n\n")

    return summary_path, trials_path

def log_trial_to_summary(summary_path, trial):
    """Appends a trial result to the summary log file."""
    ce_loss = trial.user_attrs.get("ce_loss", "N/A")
    energy = trial.user_attrs.get("energy", "N/A")
    normalized_energy = trial.user_attrs.get("normalized_energy", "N/A")
    combined_energy = trial.user_attrs.get("combined_energy", "N/A")
    trial_time = trial.user_attrs.get("trial_time", 0)
    config = trial.user_attrs.get("config", {})
    energy_fn = config.get("energy_fn_name", "unknown")

    with open(summary_path, "a") as f:
        f.write(f"{trial.number:<6} {trial_time:<8.1f} {ce_loss:<10} {energy:<12} "
                f"{normalized_energy:<12} {combined_energy:<12} {energy_fn:<12}\n")

def log_trial_to_detailed_log(trials_path, trial, config, trial_time,
                               val_loss, raw_energy, norm_energy, combined_energy):
    """Appends detailed info about a completed trial to a trials log file."""
    with open(trials_path, "a") as f:
        f.write(f"TRIAL {trial.number}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Time: {trial_time:.1f}s | Objective: {combined_energy:.6f}\n")
        f.write(f"CE Loss: {val_loss:.6f} | Raw Energy: {raw_energy:.6f} | Norm Energy: {norm_energy:.6f} | Combined: {combined_energy:.6f}\n")
        f.write(f"Config: {config.energy_fn_name} | n_embed x block_size: {config.n_embed}x{config.block_size} "
                f"| heads={config.num_heads} | blocks={config.n_blocks} | T={config.T}\n")
        f.write(f"LR: {config.peak_learning_rate:.2e} | Warmup: {config.warmup_steps} "
                f"| Dropout: {config.dropout:.3f} | Bias: {config.update_bias}\n\n")
        
def write_final_results(results_path, trial):
    config = trial.user_attrs.get("config", {})
    ce_loss = trial.user_attrs.get("ce_loss", "N/A")
    energy = trial.user_attrs.get("energy", "N/A")
    norm_energy = trial.user_attrs.get("normalized_energy", "N/A")
    combined_energy = trial.user_attrs.get("combined_energy", "N/A")

    with open(results_path, "w") as f:
        f.write("COMBINED ENERGY OPTIMIZATION RESULTS\n")
        f.write("====================================\n\n")
        f.write(f"Best combined energy: {trial.value:.4f}\n")
        f.write(f"CE Loss: {ce_loss:.4f}\n")
        f.write(f"Raw Energy: {energy:.4f}\n")
        f.write(f"Normalized Energy: {norm_energy:.4f}\n")
        f.write(f"Combined Energy: {combined_energy:.4f}\n\n")

        if config:
            f.write("Best Configuration:\n")
            for key, val in config.items():
                f.write(f"{key}: {val}\n")
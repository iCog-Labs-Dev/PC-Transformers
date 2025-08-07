def initialize_logs(study_name: str):
    """Create and initialize summary and trial log files."""
    trials_path = f"tuning/{study_name}_trials.txt"

    with open(trials_path, "w") as f:
        f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Objective: Minimize Averge Energy \n\n")

    return trials_path

def log_trial_to_detailed_log(trials_path, trial, config, trial_time, avg_energy):
    """Appends detailed info about a completed trial to a trials log file."""
    with open(trials_path, "a") as f:
        f.write(f"TRIAL {trial.number}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Time: {trial_time:.1f}s\n")
        f.write(f"Avg Energy: {avg_energy:.6f}\n")
        f.write(f"Config: n_embed x block_size: {config.n_embed}x{config.block_size} "
                f"| heads={config.num_heads} | blocks={config.n_blocks} | T={config.T}\n")
        f.write(f"LR: {config.peak_learning_rate:.2e} | Warmup: {config.warmup_steps} "
                f"| Dropout: {config.dropout:.3f} | Bias: {config.update_bias}\n\n")
        
def write_final_results(results_path, trial):
    config = trial.user_attrs.get("config", {})
    energy = trial.user_attrs.get("energy", "N/A")

    with open(results_path, "w") as f:
        f.write("COMBINED ENERGY OPTIMIZATION RESULTS\n")
        f.write("====================================\n\n")
        f.write(f"Best combined energy: {trial.value:.4f}\n")
        f.write(f"Average Energy: {energy:.4f}\n")

        if config:
            f.write("Best Configuration:\n")
            for key, val in config.items():
                f.write(f"{key}: {val}\n")
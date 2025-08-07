def initialize_logs(study_name: str):
    """Create and initialize summary and trial log files."""
    trials_path = f"tuning/{study_name}_trials.txt"

    with open(trials_path, "w") as f:
        f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Objective: Minimize Averge Energy \n\n")

    return trials_path

def log_trial_to_detailed_log(trials_path, trial, config, trial_time, avg_energy, write_header=False):
    """Appends trial information in tabular format to a trials log file."""
    with open(trials_path, "a") as f:
        if write_header:
            f.write(f"{'Trial':<6} | {'Time(s)':<8} | {'Avg Energy':<11} | "
                    f"{'n_embed':<7} | {'block_size':<10} | {'heads':<5} | {'blocks':<6} | {'T':<3} | "
                    f"{'LR':<8} | {'Warmup':<6} | {'Dropout':<7} | {'Bias':<5}\n")
            f.write("-" * 120 + "\n")
        
        f.write(f"{trial.number:<6} | {trial_time:<8.1f} | {avg_energy:<11.6f} | "
                f"{config.n_embed:<7} | {config.block_size:<10} | {config.num_heads:<5} | {config.n_blocks:<6} | "
                f"{config.T:<3} | {config.peak_learning_rate:<8.1e} | {config.warmup_steps:<6} | "
                f"{config.dropout:<7.3f} | {str(config.update_bias):<5}\n")
        
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
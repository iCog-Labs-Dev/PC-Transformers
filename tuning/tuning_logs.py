def initialize_logs(study_name: str):
    """Create and initialize summary and trial log files."""
    summary_path = f"tuning/{study_name}_summary.txt"
    trials_path = f"tuning/{study_name}_trials.txt"
    
    with open(summary_path, "w") as f:
        f.write(f"BAYESIAN TUNING - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Optimization Objective: Minimize Training Energy\n\n")
        f.write(f"{'Trial':<6} {'Time(h)':<8} {'Train E':<12} "
                f"{'Train PPL':<12} {'Val E':<12} {'Val PPL':<12} {'Energy Fn'}\n")
        f.write(f"{'-'*82}\n")

    with open(trials_path, "w") as f:
        f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Objective: Minimize CE Loss (Energy monitored but not optimized)\n\n")

    return summary_path, trials_path

def log_trial_to_summary(summary_path, trial):
    """Appends a trial result to the summary log file."""
    # Get all metrics with defaults
    metrics = {
        'trial': trial.number,
        'time': trial.user_attrs.get("trial_time", 0),
        'train_energy': trial.user_attrs.get("train_energy", "N/A"),
        'train_ppl': trial.user_attrs.get("train_perplexity", "N/A"),
        'val_energy': trial.user_attrs.get("val_energy", "N/A"),
        'val_ppl': trial.user_attrs.get("val_perplexity", "N/A"),
        'step': trial.user_attrs.get("global_step", 0)
    }
    
    config = trial.user_attrs.get("config", {})
    energy_fn = config.get("energy_fn_name", "mse")

    with open(summary_path, "a") as f:
        f.write(
            f"{metrics['trial']:<6} {metrics['time']:<8.1f} "
            f"{metrics['train_energy']:<12.4f} {metrics['train_ppl']:<12.4f} "
            f"{metrics['val_energy']:<12.4f} {metrics['val_ppl']:<12.4f} {energy_fn}\n"
        )

def log_trial_to_detailed_log(trials_path, trial, config, trial_time,
                             val_energy, val_perplexity,
                             train_energy=None, train_perplexity=None):
    """Appends detailed info about a completed trial to a trials log file."""
    with open(trials_path, "a") as f:
        f.write(f"TRIAL {trial.number}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Time: {trial_time:.2f} hours\n")
        f.write(f"Global Step: {trial.user_attrs.get('global_step', 0)}\n\n")
        
        # Training metrics
        f.write("TRAINING METRICS\n")
        f.write(f"  {'Energy:':<25} {train_energy:.6f}\n")
        f.write(f"  {'Perplexity:':<25} {train_perplexity:.4f}\n\n")
        
        # Validation metrics
        f.write("VALIDATION METRICS\n")
        f.write(f"  {'Energy:':<25} {val_energy:.6f}\n")
        f.write(f"  {'Perplexity:':<25} {val_perplexity:.4f}\n\n")

def write_final_results(results_path, trial):
    config = trial.user_attrs.get("config", {})
    ce_loss = trial.user_attrs.get("ce_loss", "N/A")
    perplexity = trial.user_attrs.get("perplexity", "N/A")
    energy = trial.user_attrs.get("energy", "N/A")
    norm_energy = trial.user_attrs.get("normalized_energy", "N/A")

    with open(results_path, "w") as f:
        f.write("CE LOSS OPTIMIZATION RESULTS\n")
        f.write("============================\n\n")
        f.write(f"Best CE Loss: {trial.value:.4f}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")
        f.write(f"Raw Energy (monitored): {energy:.4f}\n")
        f.write(f"Normalized Energy (monitored): {norm_energy:.4f}\n\n")

        if config:
            f.write("Best Configuration:\n")
            for key, val in config.items():
                f.write(f"{key}: {val}\n")
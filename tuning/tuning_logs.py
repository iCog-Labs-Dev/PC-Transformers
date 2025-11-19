import logging
import os

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
    avg_perplexity = trial.user_attrs.get("perplexity", "N/A")
    combined_loss = trial.user_attrs.get("combined_loss", "N/A")
    
    with open(trials_path, "a") as f:
        if write_header:
            f.write(f"{'Trial':<6} | {'Time(s)':<8} | {'Avg Energy':<11} | {'Perplexity':<11} | {'Combined Loss':<11} | "
                    f"{'n_embed':<7} | {'block_size':<10} | {'heads':<5} | {'blocks':<6} | {'T':<3} | "
                    f"{'LR':<8} | {'Warmup':<6} | {'Dropout':<7} | {'Bias':<5}\n")
            f.write("-" * 160 + "\n")
        
        f.write(f"{trial.number:<6} | {trial_time:<8.1f} | {avg_energy:<11.6f} | {avg_perplexity:<11.4f} | {combined_loss:<11.6f} | "
                f"{config.n_embed:<7} | {config.block_size:<10} | {config.num_heads:<5} | {config.n_blocks:<6} | {config.T:<3} |"
                f"{config.peak_learning_rate:<8.2e} | {config.warmup_steps:<6} | {config.dropout:<7.3f} | {str(config.update_bias):<5}\n")
        
def write_final_results(results_path, trial):
    config = trial.user_attrs.get("config", {})
    energy = trial.user_attrs.get("energy", "N/A")
    perplexity = trial.user_attrs.get("perplexity", "N/A")
    combined_loss = trial.user_attrs.get("combined_loss", "N/A")
    
    with open(results_path, "w") as f:
        f.write("COMBINED ENERGY OPTIMIZATION RESULTS\n")
        f.write("====================================\n\n")
        f.write(f"Best combined energy: {trial.value:.4f}\n")
        f.write(f"Average Energy: {energy:.4f}\n")
        f.write(f"Average Perplexity: {perplexity:.4f}\n")
        f.write(f"Combined Loss: {combined_loss:.4f}\n\n")
        
        if config:
            f.write("Best Configuration:\n")
            for key, val in config.items():
                f.write(f"{key}: {val}\n")

def trial_batch_logger(trial_number: int, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a logger that prepends trial and epoch info to every message.
    
    Args:
        trial_number: Current trial number
        log_dir: Directory where batch logs will be saved
    """
    os.makedirs(log_dir, exist_ok=True)
    
    base_logger = logging.getLogger(f"trial_{trial_number}")
    base_logger.setLevel(logging.INFO)
    
    if not base_logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "batch_debug.log"), mode="a")
        fmt = logging.Formatter(f"[Trial {trial_number} | %(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        base_logger.addHandler(fh)
    
    return base_logger
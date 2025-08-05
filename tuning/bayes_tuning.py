"""
Bayesian Hyperparameter Tuning
"""
import torch
import logging
import optuna
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Data_preprocessing.dataloader import get_loaders
from tuning.trial_objective import objective
from tuning.tuning_logs import initialize_logs, write_final_results
import torch.distributed as dist
import argparse
from utils.device_utils import setup_ddp
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""Usage:  python bayes_tuning.py """

def run_tuning(n_trials=30, study_name="bayesian_tuning", local_rank=0, device=None, flash=False):
    """Run clean dynamic hyperparameter tuning"""
    storage_url = f"sqlite:///tuning/{study_name}.db"
    if local_rank == 0 or local_rank == -1:
        _ = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1
            )
        )
    if dist.is_initialized():
        dist.barrier()

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )

    summary_path, trials_path = initialize_logs(study_name)
    logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
    logger.info(f"[Rank {local_rank}] Summary Log: {summary_path}")
    logger.info(f"[Rank {local_rank}] Trials Log: {trials_path}")

    try:
        study.optimize(lambda trial: objective(trial, device, flash), n_trials=n_trials, show_progress_bar=(local_rank == 0))
        logger.info(f"[Rank {local_rank}] Bayesian tuning completed!")
    
        if local_rank == 0 and study.best_trial:
                trial = study.best_trial
                logger.info(f"Best trial: {trial.number}. Best value: {trial.value:.5f}")
                write_final_results(f"tuning/{study_name}_results.txt", trial)
        return study
    
    except KeyboardInterrupt:
        logger.warning(f"[Rank {local_rank}] Tuning interrupted")
        return study

def get_tokenizer_dir(dataset):
    # Use a directory, not a file, for HuggingFace
    base_dir = os.path.join("Data_preprocessing", "tokenizer", "outputs")
    dir_name = f"gpt2_tokenizer_{dataset}"
    return os.path.join(base_dir, dir_name)

def load_tokenizer_with_fallback(dataset):
    tokenizer_dir = get_tokenizer_dir(dataset)
    if not os.path.isdir(tokenizer_dir) or not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print(f"Tokenizer directory {tokenizer_dir} not found or incomplete. Attempting to run tokenizer script...")
        try:
            subprocess.run([
                "python", "-m", "Data_preprocessing.tokenizer.gpt2_tokenizer",
                f"--dataset={dataset}"
            ], check=True)
        except Exception as e:
            print(f"Tokenizer script failed: {e}")
            raise RuntimeError("Tokenizer could not be loaded or created.")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_dir}: {e}")
        raise
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Tuning with Predictive Coding Transformer")
    parser.add_argument('--flash', '--flash_attention', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    local_rank, is_distributed = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set config flag for FlashAttention
    use_flash_attention = args.flash
    
    train_loader, valid_loader,_ = get_loaders(is_distributed)
    run_tuning(n_trials= 30, study_name="bayesian_tuning", local_rank=local_rank, device=device, flash=use_flash_attention)

    if is_distributed:
        dist.destroy_process_group()
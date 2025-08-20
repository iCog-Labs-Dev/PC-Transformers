import torch
import logging
import optuna
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Data_preprocessing.dataloader import get_loaders
from tuning.trial_objective import objective
from tuning.tuning_logs import initialize_logs, write_final_results
import torch.distributed as dist
import argparse


logger = logging.getLogger(__name__)


"""
This script performs Bayesian hyperparameter tuning for the model using Optuna. 
It supports energy-based evaluation, model configuration, and training setup,
and is compatible with multi-GPU execution using DDP.

Usage:  torchrun --nproc-per-node=<NUM_GPU> tuning/bayes_tuning.py 

"""

def run_tuning(n_trials=30, study_name="bayesian_tuning", local_rank=0, device=None, flash=False):
    """Run clean dynamic hyperparameter tuning"""
    # Use absolute path for database to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, f"{study_name}.db")
    storage_url = f"sqlite:///{db_path}"
    if local_rank == 0 or local_rank == -1:
        try:
            study = optuna.create_study(
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
        except Exception as e:
            logger.warning(f"Study creation failed: {e}, trying to load existing study")
            study = optuna.load_study(study_name=study_name, storage=storage_url)
    else:
        # Non-main processes don't need to access the study directly
        study = None

    # Only rank 0 should run the optimization and logging
    if local_rank == 0 or local_rank == -1:
        trials_path = initialize_logs(study_name)
        logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
        logger.info(f"[Rank {local_rank}] Trials Log: {trials_path}")

        try:
            study.optimize(lambda trial: objective(trial, device, flash), n_trials=n_trials, show_progress_bar=True)
            logger.info(f"[Rank {local_rank}] Bayesian tuning completed!")

            if study.best_trial:
                trial = study.best_trial
                logger.info(f"Best trial: {trial.number}. Best value: {trial.value:.5f}")
                results_path = os.path.join(script_dir, f"{study_name}_results.txt")
                write_final_results(results_path, trial)
            return study

        except KeyboardInterrupt:
            logger.warning(f"[Rank {local_rank}] Tuning interrupted")
            return study
    else:
        # Non-main processes just exit - no need to wait
        logger.info(f"[Rank {local_rank}] Non-main process exiting...")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Tuning with Predictive Coding Transformer")
    parser.add_argument('--flash', '--flash_attention', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    # Setup distributed-aware logging
    rank = int(os.environ.get("RANK", 0))
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

    # Completely disable Optuna logging to avoid conflicts
    try:
        # Disable all Optuna logging
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        # Also try to disable any internal Optuna logging configurations
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.setLevel(logging.CRITICAL)
        optuna_logger.disabled = True

        # Disable specific Optuna loggers that might cause issues
        for logger_name in ['optuna.study', 'optuna.trial', 'optuna.samplers', 'optuna.pruners']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.disabled = True

    except Exception as e:
        print(f"Warning: Could not configure Optuna logging: {e}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    if "RANK" in os.environ and torch.cuda.is_available():
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set config flag for FlashAttention
    use_flash_attention = args.flash
    
    train_loader, valid_loader,_ = get_loaders((local_rank >= 0))
    run_tuning(n_trials= 30, study_name="bayesian_tuning", local_rank=local_rank, device=device, flash=use_flash_attention)

    if dist.is_initialized():
        dist.destroy_process_group()
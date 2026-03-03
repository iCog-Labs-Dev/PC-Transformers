import torch
import logging
import optuna
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tuning.trial_objective import objective
from tuning.tuning_logs import initialize_logs, write_final_results
import torch.distributed as dist
import argparse
from utils.device_utils import setup_device
from utils.model_utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""
This script performs Bayesian hyperparameter tuning for the model using Optuna. 
It supports energy-based evaluation, model configuration, and training setup,
and is compatible with multi-GPU execution using DDP.

Usage:  torchrun --nproc-per-node=<NUM_GPU> tuning/bayes_tuning.py 

"""

def run_tuning(n_trials=30, study_name="bayesian_tuning", local_rank=0, device=None, flash=False, enable_batch_logging=False):
    """Run clean dynamic hyperparameter tuning"""
    storage_url = f"sqlite:///tuning/{study_name}.db"
    if local_rank == 0:
        try:
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
        except Exception as e:
            logger.warning(f"Study creation skipped because the file already exists: {e}")
    if dist.is_initialized():
        dist.barrier()

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )

   
    if local_rank == 0:
        trials_path = initialize_logs(study_name)
        logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
        logger.info(f"[Rank {local_rank}] Trials Log: {trials_path}")
    else:
        trials_path = f"tuning/{study_name}_trials.txt"
    
    def callback(study, trial):
        if local_rank == 0:
            best_trial = study.best_trial
            train_energy = best_trial.user_attrs.get("energy", "N/A")
            train_perplexity = best_trial.user_attrs.get("perplexity", "N/A")
            combined_loss = best_trial.user_attrs.get("combined_loss", "N/A")
            logger.info(f"\nBest trial so far: {best_trial.number} | Combined Loss: {combined_loss:.5f} | Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f}\n")

    try:
        study.optimize(lambda trial: objective(trial, device, flash, enable_batch_logging=enable_batch_logging), n_trials=n_trials,  callbacks=[callback], show_progress_bar=(local_rank == 0))
        logger.info(f"[Rank {local_rank}] Bayesian tuning completed!")
    
        if local_rank == 0 and study.best_trial:
                best_trial = study.best_trial
                train_energy = best_trial.user_attrs.get("energy", "N/A")
                train_perplexity = best_trial.user_attrs.get("perplexity", "N/A")
                combined_loss = best_trial.user_attrs.get("combined_loss", "N/A")
                logger.info(f"\nFinal Best trial: {best_trial.number} | Combined Loss: {combined_loss:.5f} | Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f}\n")
                write_final_results(f"tuning/{study_name}_results.txt", best_trial)
        
        if dist.is_initialized():
            dist.barrier()
            
        return study

    
    except KeyboardInterrupt:
        logger.warning(f"[Rank {local_rank}] Tuning interrupted")
        return study

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Tuning with Predictive Coding Transformer")
    parser.add_argument('--flash', '--flash_attention', action='store_true', help='Enable FlashAttention for attention layers')
    parser.add_argument('--log_batches', action='store_true', help='Enable batch-level logging during tuning')
    args = parser.parse_args()
    
    rank = int(os.environ.get("RANK", 0))
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )
    
    local_rank, device, use_ddp = setup_device()
    
    if use_ddp and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    if use_ddp:
        dist.barrier()
    
    study = run_tuning(n_trials= 70, study_name="bayesian_tuning", local_rank=local_rank, device=device, flash=args.flash, enable_batch_logging=args.log_batches)

    if use_ddp and dist.is_initialized():
        dist.barrier() 
        dist.destroy_process_group()
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
from utils.device_utils import setup_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""
This script performs Bayesian hyperparameter tuning for the model using Optuna. 
It supports energy-based evaluation, model configuration, and training setup,
and is compatible with multi-GPU execution using DDP.

Usage:  torchrun --nproc-per-node=<NUM_GPU> tuning/bayes_tuning.py 

"""

def run_tuning(n_trials=30, study_name="bayesian_tuning", local_rank=0, device=None, flash=False):
    """Run clean dynamic hyperparameter tuning"""
    storage_url = f"sqlite:///tuning/{study_name}.db"
    # if local_rank == 0 or local_rank == -1:
    #     _ = optuna.create_study(
    #         direction='minimize',
    #         study_name=study_name,
    #         storage=storage_url,
    #         load_if_exists=True,
    #         sampler=optuna.samplers.TPESampler(seed=42),
    #         pruner=optuna.pruners.MedianPruner(
    #             n_startup_trials=5,
    #             n_warmup_steps=3,
    #             interval_steps=1
    #         )
    #     )
    # if dist.is_initialized():
    #     dist.barrier()
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
        dist.barrier(device_ids=[local_rank])

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )

    # if local_rank == 0 or local_rank == -1:
    if local_rank == 0:
        trials_path = initialize_logs(study_name)
        logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
        logger.info(f"[Rank {local_rank}] Trials Log: {trials_path}")

    try:
        study.optimize(lambda trial: objective(trial, device, flash), n_trials=n_trials, show_progress_bar=(local_rank == 0))
        logger.info(f"[Rank {local_rank}] Bayesian tuning completed!")
    
        if local_rank == 0 and study.best_trial:
                trial = study.best_trial
                logger.info(f"Best trial: {trial.number}. Best value: {trial.value:.5f}")
                write_final_results(f"tuning/{study_name}_results.txt", trial)
        dist.barrier(device_ids=[local_rank])   
        return study
    
    except KeyboardInterrupt:
        logger.warning(f"[Rank {local_rank}] Tuning interrupted")
        return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Tuning with Predictive Coding Transformer")
    parser.add_argument('--flash', '--flash_attention', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()
    
    rank = int(os.environ.get("RANK", 0))
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    local_rank, device, use_ddp = setup_device()
    
    if use_ddp and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=local_rank)

    if use_ddp:
        dist.barrier(device_ids=[local_rank])
    
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)
        
    # if "RANK" in os.environ and torch.cuda.is_available():
    #     import torch.distributed as dist
    #     dist.init_process_group(backend="nccl")
    #     local_rank = int(os.environ["LOCAL_RANK"])
    #     device = torch.device(f"cuda:{local_rank}")
    #     torch.cuda.set_device(local_rank)
    # else:
    #     local_rank = -1
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set config flag for FlashAttention
    # use_flash_attention = args.flash
    
    
    # train_loader, valid_loader,_ = get_loaders((local_rank >= 0))
    study = run_tuning(n_trials= 30, study_name="bayesian_tuning", local_rank=local_rank, device=device, flash=use_flash_attention)

    # if dist.is_initialized():
    #     dist.destroy_process_group()
    if use_ddp and dist.is_initialized():
        dist.barrier(device_ids=[local_rank]) 
        dist.destroy_process_group()
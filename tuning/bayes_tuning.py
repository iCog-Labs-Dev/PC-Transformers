"""
Bayesian Hyperparameter Tuning
"""
import torch
import logging
import optuna
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.device_utils import setup_device
from tuning.trial_objective import objective
from tuning.tuning_logs import initialize_logs, write_final_results
import torch.distributed as dist

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""Usage: torchrun --nproc-per-node=2 tuning/bayes_tuning.py """

def run_tuning(n_trials=30, study_name="bayesian_tuning", local_rank=0, device=None):
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
        dist.barrier(device_ids=[local_rank])
        
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )
    
    if local_rank == 0:
        summary_path, trials_path = initialize_logs(study_name)
    else:
        summary_path = f"tuning/{study_name}_summary.txt"
        trials_path = f"tuning/{study_name}_trials.txt"
        
    logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
    logger.info(f"[Rank {local_rank}] Summary Log: {summary_path}")
    logger.info(f"[Rank {local_rank}] Trials Log: {trials_path}")

    try:
        if local_rank == 0:
            study.optimize(lambda trial: objective(trial, device), n_trials=n_trials, show_progress_bar= True)
            logger.info(f"[Rank {local_rank}] Tuning completed.")
            
            if len(study.trials) > 0:
                trial = study.best_trial
                logger.info(f"Best trial: {trial.number}. Best value: {trial.value:.5f}")
                write_final_results(f"tuning/{study_name}_results.txt", trial)
            else:
                logger.warning("[Rank 0] No completed trials to report.")
            
        dist.barrier(device_ids=[local_rank])   
        return study
     
    except KeyboardInterrupt:
        logger.warning(f"[Rank {local_rank}] Tuning interrupted")
        return study

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    local_rank, device, use_ddp = setup_device()
    
    if use_ddp and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=local_rank)

    run_tuning(n_trials= 30, study_name="bayesian_tuning", local_rank=local_rank, device=device)

    if use_ddp and dist.is_initialized():
        dist.barrier(device_ids=[local_rank]) 
        dist.destroy_process_group()
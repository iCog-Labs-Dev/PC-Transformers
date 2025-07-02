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
from tuning.tuning_logs import initialize_logs, log_trial_to_summary, write_final_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
"""Usage:  python bayes_tuning.py """

def run_tuning(n_trials=30, study_name="bayesian_tuning", local_rank=0, device=None):
    """Run clean dynamic hyperparameter tuning"""
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=f'sqlite:///tuning/{study_name}.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
    )

    summary_path, trials_path = initialize_logs(study_name)
    logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"[Rank {local_rank}] Starting Bayesian tuning with {n_trials} trials")
    logger.info(f"Summary Log: {summary_path}")
    logger.info(f"Trials Log: {trials_path}")

    try:
        study.optimize(lambda trial: objective(trial, device), n_trials=n_trials, show_progress_bar=(local_rank == 0))
        logger.info("Bayesian tuning completed!")
        
        if local_rank == 0 and study.best_trial:
            trial = study.best_trial
            logger.info(f"Best trial: {trial.number}. Best combined energy: {trial.value:.5f}")
            write_final_results(f"tuning/{study_name}_results.txt", trial)
        return study
    
    except KeyboardInterrupt:
        logger.info("Bayesian tuning interrupted")
    return study

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if "RANK" in os.environ and torch.cuda.is_available():
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader,_ = get_loaders((local_rank >= 0))
    run_tuning(n_trials= 30, study_name="bayesian_tuning", local_rank=local_rank, device=device)
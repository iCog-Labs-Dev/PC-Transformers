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

"""Usage:  python bayes_tuning.py """

def run_tuning(n_trials=30, study_name="bayesian_tuning"):
    """Run clean dynamic hyperparameter tuning"""
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
    )

    summary_path, trials_path = initialize_logs(study_name)

    logger.info(f"Starting Bayesian tuning with {n_trials} trials")
    logger.info(f"Summary Log: {summary_path}")
    logger.info(f"Trials Log: {trials_path}")

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("Bayesian tuning completed!")
        
        if study.best_trial:
            trial = study.best_trial
            logger.info(f"Best trial: {trial.number}. Best combined energy: {trial.value:.5f}")
            write_final_results(f"{study_name}_results.txt", trial)
        return study
    
    except KeyboardInterrupt:
        logger.info("Bayesian tuning interrupted")
        return study

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train_loader, valid_loader,_ = get_loaders()
    run_tuning(n_trials= 30, study_name="bayesian_tuning")
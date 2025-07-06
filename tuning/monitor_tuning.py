"""
Monitor Bayesian hyperparameter tuning progress
Usage: python monitor_tuning.py [study_name]
"""
import optuna
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

"""Usage: python monitor_tuning.py bayesian_tuning"""

def monitor_study(study_name="adaptive_pc_transformer_tuning"):
    """Monitor and visualize hyperparameter tuning progress"""
    db_path = f"{study_name}.db"
    if not Path(db_path).exists():
        print(f"Study database {db_path} not found!")
        return
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f'sqlite:///{db_path}')
        
        print(f"Study: {study_name}")
        print(f"Direction: {study.direction}")
        print(f"Total trials: {len(study.trials)}")
        
        if len(study.trials) == 0:
            print("No trials found!")
            return
        
        if study.best_trial:
            print(f"\nBest trial:")
            print(f"  Value: {study.best_trial.value:.4f}")
            print(f"  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
        
        print(f"\nRecent trials:")
        for trial in study.trials[-5:]:
            status = trial.state.name
            value = f"{trial.value:.4f}" if trial.value else "N/A"
            print(f"  Trial {trial.number}: {value} ({status})")
        
    except Exception as e:
        print(f"Error monitoring study: {e}")

if __name__ == "__main__":
    study_name = sys.argv[1] if len(sys.argv) > 1 else "pc_transformer_bayes_tuning"
    monitor_study(study_name)
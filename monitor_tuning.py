"""
Monitor Bayesian hyperparameter tuning progress
Usage: python monitor_tuning.py [study_name]
"""

import optuna
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def monitor_study(study_name="adaptive_pc_transformer_tuning"):
    """Monitor and visualize hyperparameter tuning progress"""
    
    db_path = f"{study_name}.db"
    if not Path(db_path).exists():
        print(f"Study database {db_path} not found!")
        return
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f'sqlite:///{db_path}'
        )
        
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
        
        if len(study.trials) > 1:
            print(f"\nCreating visualizations...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            values = [t.value for t in study.trials if t.value is not None]
            if values:
                ax1.plot(values, 'b-', alpha=0.7, label='Trial values')
                ax1.axhline(y=min(values), color='r', linestyle='--', label='Best value')
                ax1.set_xlabel('Trial')
                ax1.set_ylabel('Objective Value')
                ax1.set_title('Optimization History')
                ax1.legend()
                ax1.grid(True)
            
            if len(study.trials) >= 10:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    params = list(importance.keys())
                    importances = list(importance.values())
                    
                    ax2.barh(params, importances)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance')
                    ax2.grid(True, axis='x')
                except:
                    ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, f'Need ≥10 trials for\nparameter importance\n(current: {len(study.trials)})', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            plot_path = f"{study_name}_progress.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Progress plot saved to {plot_path}")
            plt.show()
            
            df_data = []
            for trial in study.trials:
                row = {'trial': trial.number, 'value': trial.value, 'state': trial.state.name}
                row.update(trial.params)
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_path = f"{study_name}_trials.csv"
            df.to_csv(csv_path, index=False)
            print(f"Trial data saved to {csv_path}")
        
    except Exception as e:
        print(f"Error monitoring study: {e}")

if __name__ == "__main__":
    study_name = sys.argv[1] if len(sys.argv) > 1 else "pc_transformer_bayes_tuning"
    monitor_study(study_name)

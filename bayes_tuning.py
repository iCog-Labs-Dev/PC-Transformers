import optuna
import torch
from predictive_coding.config import GPTConfig
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader, valid_loader
from training import train
from eval import evaluate
from utils.model_utils import load_tokenizer

def get_model_config(trial, vocab_size):

    n_embed = trial.suggest_categorical('n_embed', [32, 64, 128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
    
    while n_embed % num_heads != 0:
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])


    return GPTConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_embed=n_embed,
        dropout=trial.suggest_float('dropout', 0.05, 0.5),
        local_learning_rate=trial.suggest_float('local_learning_rate', 1e-6, 1e-4, log=True),
        T=trial.suggest_int('T', 5, 10), 
        is_holding_error=True,
        num_heads=num_heads,
        n_blocks=trial.suggest_int('n_blocks', 1, 4),
        num_epochs=1,
        update_bias=True,
        use_lateral=True,
        energy_fn_name="kld"
    )

def objective(trial):
    try:
        tokenizer = load_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        config = get_model_config(trial, vocab_size)
        model = PCTransformer(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        for epoch in range(1):
            avg_energy, _ = train(model, train_loader)
        
        avg_energy_val, val_loss = evaluate(model, valid_loader, max_batches=10, compute_metrics=False)
        return val_loss
   
    except (RuntimeError, ValueError) as e:
        print(f"Error with trial {trial.number}: {str(e)}")
        return float("inf")


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    
    best_params_path = "best_params.txt"
    with open(best_params_path, "w") as f:
        f.write("Best parameters found:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest parameters saved to {best_params_path}")
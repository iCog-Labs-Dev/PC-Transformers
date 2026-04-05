def define_search_space(trial):
    """
    Defines the Phase 1 hyperparameter search space for Optuna trials.
    
    This function acts as a mathematical guardrail, ensuring that Optuna 
    only samples stable and compatible model architectures during the 
    Energy minimization phase.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        dict: A dictionary containing the sampled hyperparameters for this trial:
            - n_blocks (int): Model depth
            - num_heads (int): Number of attention heads
            - n_embed (int): Total embedding dimension
            - embed_mult (int): The embedding dimension size per individual head
            - T (int): Number of Expectation-Maximization inference steps (scales with n_blocks)
            - batch_size (int): Number of sequences processed simultaneously
            - lr (float): Weight update learning rate
            - inference_lr (float): Learning rate for the latent state (x) updates during inference
            - dropout (float): Dropout probability for regularization
    """
    n_blocks = trial.suggest_int("n_blocks", 1, 3)
    min_T = (n_blocks * 4) + 3
    n_heads = trial.suggest_int("num_heads", 2, 8, step=2)
    embed_mult = trial.suggest_int("embed_mult", 16, 64, step=16)
    n_embed = n_heads * embed_mult
    n_embed = trial.suggest_int("n_embed", n_embed, n_embed)
    
    return {
        "n_blocks": n_blocks,
        "num_heads": n_heads,
        "n_embed": n_embed,
        "embed_mult": embed_mult,
        "T": trial.suggest_int("T", min_T, min_T + 2),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        "lr": trial.suggest_float("lr", 1e-5, 6e-5, log=True),
        "inference_lr": trial.suggest_float("inference_lr", 0.05, 0.20, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.1),
    }

def define_search_space_phase2(trial, best_params):
    """
    Defines the hyperparameter search space for Phase 2 (Continuous Fine-Tuning).
    
    This function takes the winning parameters from Phase 1 and explores a very tight, 
    focused window around them. It uses hard minimum and maximum limits to keep the 
    search safe and prevent the model from exploding.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.
        best_params (dict): The winning hyperparameters from Phase 1.

    Returns:
        dict: A dictionary containing the finely tuned continuous parameters:
            - lr (float): Weight update learning rate 
            - inference_lr (float): Latent state learning rate
            - dropout (float): Dropout probability
    """
    lr = best_params.get("lr", 1e-3)
    inference_lr = best_params.get("inference_lr", 0.1)
    dropout = best_params.get("dropout", 0.05)
    
    return {
        "lr": trial.suggest_float("lr", max(1e-5, lr * 0.5), min(6e-5, lr * 2.0), log=True),
        "inference_lr": trial.suggest_float("inference_lr", max(0.05, inference_lr * 0.7), min(0.25, inference_lr * 1.3), log=True), 
        "dropout": trial.suggest_float("dropout", max(0.0, dropout - 0.02), min(0.15, dropout + 0.02)),
    }
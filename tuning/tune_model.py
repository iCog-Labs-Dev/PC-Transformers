import sys
import time
import logging
import warnings
from pathlib import Path

# Project Root Setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

from training import train
from eval import evaluate
from predictive_coding.config import GPTConfig
from data_preparation.dataloader import get_loaders
from model_architecture.pc_t_model import PCTransformer

from data_preparation.config import vocab_size, max_len

# Silence warnings and Optuna spam
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)

ENERGY_STABILITY_THRESHOLD = 300

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
        # "inference_lr": trial.suggest_float("inference_lr", 0.05, 0.20, log=True),
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
    # inference_lr = best_params.get("inference_lr", 0.1)
    dropout = best_params.get("dropout", 0.05)
    
    return {
        "lr": trial.suggest_float("lr", max(1e-5, lr * 0.5), min(6e-5, lr * 2.0), log=True),
        # "inference_lr": trial.suggest_float("inference_lr", max(0.05, inference_lr * 0.7), min(0.25, inference_lr * 1.3), log=True), 
        "dropout": trial.suggest_float("dropout", max(0.0, dropout - 0.02), min(0.15, dropout + 0.02)),
    }

def create_model(params):
    """
    Assembles the model, config, and data for a single trial.

    This function takes the hyperparameters suggested by Optuna and combines them 
    with fixed, hardcoded configuration rules. It instantiates the model on the 
    correct hardware device, initializes its specific lateral weights, and fetches 
    the data loaders.

    Args:
        params (dict): The hyperparameters chosen by Optuna for this specific trial.

    Returns:
        tuple: A fully prepared training environment containing:
            - model: PC-Transformer model
            - config: The complete configuration blueprint
            - train_loader: The dataset for training
            - valid_loader: The dataset for validation
            - device: The hardware device (CPU/GPU) being used

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size= max_len,
        lr=params["lr"],
        # inference_lr=params["inference_lr"],
        peak_learning_rate=params["lr"],
        warmup_steps=100,
        n_embed=params["n_embed"],
        dropout=params["dropout"],
        T=params["T"],
        num_heads=params["num_heads"],
        n_blocks=params["n_blocks"],
        batch_size=params["batch_size"],
        num_epochs=1, 
        internal_energy_fn_name="pc_e", 
        output_energy_fn_name="kld",    
        combined_internal_weight=1.0,
        combined_output_weight=1.0,
        use_flash_attention=False,
        alpha=0.5,
        update_bias=True
    )

    model = PCTransformer(config).to(device)
    model.register_all_lateral_weights() 
    train_loader, valid_loader, _ = get_loaders(distributed=False)
    
    return model, config, train_loader, valid_loader, device

def run_phase1_trial(trial):
    """
    Phase 1 Trial Execution
    
    This function builds a model using Phase 1 parameters and runs a brief 
    training pass. It acts as a strict filter: if the model's energy explodes 
    (becomes NaN or exceeds the threshold), the trial is immediately pruned.
    Surviving models are evaluated and scored by their Energy.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.

    Returns:
        float: The validation energy score (Optuna attempts to minimize this).
    """
    try:
        params = define_search_space(trial)
        print(f"[Energy Phase] Trial {trial.number} | params: {params}")

        model, config, train_loader, valid_loader, device = create_model(params)
        start_time = time.time()

        try:
            # We use the internal train function. If it explodes, it will return high energy or NaN.
            train_energy, train_ppl, _ = train(model, train_loader, config, global_step=0, device=device, logger=None)
            
            if torch.isnan(torch.tensor(train_energy)) or train_energy > ENERGY_STABILITY_THRESHOLD:
                reason = f"Unstable Energy: {train_energy}"
                trial.set_user_attr("prune_reason", reason)
                print(f"  {reason}")
                raise optuna.TrialPruned()

            # Evaluate on a subset of validation data
            val_energy, val_ppl = evaluate(model, config, valid_loader, max_batches=10, device=device)

        except Exception as e:
            if not isinstance(e, optuna.TrialPruned):
                reason = f"Execution failed: {e}"
                trial.set_user_attr("prune_reason", reason)
                print(f"  {reason}")
            raise optuna.TrialPruned()

        total_time = time.time() - start_time
        
        trial.set_user_attr("val_ppl", float(val_ppl))
        trial.set_user_attr("time", total_time)
        for key, value in params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(f"Trial {trial.number} Complete | Val Energy={val_energy:.4f} | Val PPL={val_ppl:.4f} | Time={total_time:.1f}s")
        return float(val_energy)

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def run_phase2_trial(trial, best_params):
    """
    Phase 2 Trial Execution
    
    This function fixes the architecture found in Phase 1 and fine-tunes the 
    continuous parameters. While it still monitors energy for safety, it scores 
    trials based on Perplexity (PPL) to find the most intelligent version of
    the stable architecture.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.
        best_params (dict): The winning structural parameters from Phase 1.

    Returns:
        float: The validation perplexity score (Optuna attempts to minimize this).
    """
    continuous_params = define_search_space_phase2(trial, best_params)
    params = {**best_params, **continuous_params}
    
    tuning_params = {k: v for k, v in params.items() if k in ['lr', 'dropout']}     # 'inference_lr'
    print(f"[PPL Phase - Continuous Only] Trial {trial.number} | params: {tuning_params}")
    print(f"[PPL Phase - Fixed] Architecture: n_blocks={params['n_blocks']}, num_heads={params['num_heads']}, "
          f"n_embed={params['n_embed']}, T={params['T']}")

    try:
        model, config, train_loader, valid_loader, device = create_model(params)
        start_time = time.time()

        try:
            train_energy, train_ppl, _ = train(model, train_loader, config, global_step=0, device=device, logger=None)
            
            if torch.isnan(torch.tensor(train_energy)) or train_energy > ENERGY_STABILITY_THRESHOLD:
                reason = f"Unstable Energy during Phase 2: {train_energy}"
                trial.set_user_attr("prune_reason", reason)
                print(f"  {reason}")
                raise optuna.TrialPruned()

            val_energy, val_ppl = evaluate(model, config, valid_loader, max_batches=10, device=device)

        except Exception as e:
            if not isinstance(e, optuna.TrialPruned):
                reason = f"Execution failed: {e}"
                trial.set_user_attr("prune_reason", reason)
                print(f"  {reason}")
            raise optuna.TrialPruned()

        total_time = time.time() - start_time
        
        trial.set_user_attr("val_energy", float(val_energy))
        trial.set_user_attr("time", total_time)
        for key, value in params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(f"Trial {trial.number} Complete | Final Val PPL={val_ppl:.4f} | Time={total_time:.1f}s")
        return float(val_ppl)

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def run_tuning_pipeline():
    """
    Coordinates the two-phase tuning pipeline

    This function manages the full transition from Phase 1 to Phase 2.

    It initializes permanent storage to track progress, executes the 
    architectural search, freezes the winning structure, and then launches 
    the fine-tuning phase. It concludes by logging the final optimized
    parameters and the total improvement achieved.

    Returns:
        dict: A summary of the best results and parameters from both phases.
    """
    tuning_path = Path("tuning")
    tuning_path.mkdir(exist_ok=True)
    
    # Using JournalStorage to permanently bypass SQLite errors
    storage_file = str(tuning_path / "optuna_journal.log")
    storage = JournalStorage(JournalFileStorage(storage_file))

    print("PHASE 1: TPE optimizing Energy")
    study_energy = optuna.create_study(
        study_name="pc_transformer_phase1_energy",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=2)
    )

    study_energy.optimize(run_phase1_trial, n_trials=50, n_jobs=1, show_progress_bar=False)

    if study_energy.best_trial:
        best_energy = study_energy.best_value
        best_energy_ppl = study_energy.best_trial.user_attrs.get("val_ppl", "N/A")
        best_params = study_energy.best_trial.params
        
        print(f"\n{'='*60}")
        print("PHASE 1 COMPLETE")
        print(f"{'='*60}")
        print(f"Best Energy: {best_energy:.4f}")
        print(f"Corresponding PPL: {best_energy_ppl}")
        print(f"\nBest Architecture Parameters (FIXED for Phase 2):")
        for key in ['n_blocks', 'num_heads', 'n_embed', 'embed_mult', 'T', 'batch_size']:
            print(f"  {key}: {best_params.get(key)}")
        print(f"\nContinuous Parameters to be fine-tuned in Phase 2:")
        for key in ['lr', 'dropout']:       # 'inference_lr'
            print(f"  {key}: {best_params.get(key)}")
    else:
        return None

    print(f"\n{'='*60}")
    print("PHASE 2: TPE optimizing Perplexity")
    print(f"{'='*60}")
    print("Strategy: Keep architecture/discrete parameters FIXED")
    print("Only fine-tune: lr, dropout")  # 'inference_lr'
    
    study_ppl = optuna.create_study(
        study_name="pc_transformer_phase2_ppl",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=3,
            multivariate=True,    
            group=True,          
            prior_weight=1.0,     
            consider_endpoints=True  
        )
    )
    
    print(f"Resuming with {len(study_ppl.trials)} previous Phase 2 trials")

    def phase2_trial_wrapper(trial):
        return run_phase2_trial(trial, best_params)

    study_ppl.optimize(phase2_trial_wrapper, n_trials=50, n_jobs=1, show_progress_bar=False)

    if study_ppl.best_trial:
        best_ppl = study_ppl.best_value
        final_params = study_ppl.best_trial.params
        
        print(f"\n{'='*60}")
        print("PHASE 2 COMPLETE")
        print(f"{'='*60}")
        print(f"Best PPL achieved: {best_ppl:.4f}")
        
        print(f"\nParameter changes from Phase 1 -> Phase 2:")
        for key in ['lr', 'dropout']:  # 'inference_lr',
            phase1_val = best_params.get(key)
            phase2_val = final_params.get(key)
            change_pct = ((phase2_val - phase1_val) / phase1_val * 100) if phase1_val != 0 else 0
            print(f"  {key}: {phase1_val:.6f} -> {phase2_val:.6f} ({change_pct:+.1f}%)")
        
        with open("tuning/best_hyperparameters.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("BEST HYPERPARAMETERS\n")
            f.write("="*60 + "\n\n")
            
            f.write("PHASE 1 - BEST FOR ENERGY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Energy: {best_energy:.6f}\n")
            f.write(f"Corresponding PPL: {best_energy_ppl}\n")
            f.write("-" * 40 + "\n")
            for key, value in best_params.items():
                f.write(f"{key} = {value}\n")
            
            f.write("\n" + "="*60 + "\n\n")
            
            f.write("PHASE 2 - BEST FOR PPL:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best PPL: {best_ppl:.6f}\n")
            f.write("-" * 40 + "\n")
            for key, value in final_params.items():
                f.write(f"{key} = {value}\n")
        
        print(f"\n✓ Best hyperparameters saved to: tuning/best_hyperparameters.txt")
        
        return {
            "phase1_best_energy": best_energy,
            "phase1_best_ppl": best_energy_ppl,
            "phase2_best_ppl": best_ppl,
            "phase1_parameters": best_params,
            "phase2_parameters": final_params,
            "improvement_pct": ((best_energy_ppl - best_ppl) / best_energy_ppl * 100) if isinstance(best_energy_ppl, float) and best_energy_ppl > 0 else 0
        }
    return None

def main():
    print("PC TRANSFORMER - TWO-PHASE HYPERPARAMETER TUNING")
    print("="*60)
    print("PHASE 1: Find stable architecture (minimize Energy)")
    print("PHASE 2: Fine-tune continuous parameters (minimize Perplexity)")
    print("="*60)

    try:
        results = run_tuning_pipeline()
        if results:
            print(f"\n{'='*60}")
            print("TUNING COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Final Results:")
            print(f"- Phase 1 Best Energy: {results['phase1_best_energy']:.4f}")
            if isinstance(results['phase1_best_ppl'], float):
                print(f"- Phase 1 Corresponding PPL: {results['phase1_best_ppl']:.4f}")
            print(f"- Phase 2 Best PPL: {results['phase2_best_ppl']:.4f}")
            if 'improvement_pct' in results and results['improvement_pct'] != 0:
                print(f"- Improvement: {results['improvement_pct']:+.1f}%")
            print(f"\n Parameters saved to: tuning/best_hyperparameters.txt")
        else:
            print("Tuning failed or was interrupted.")
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")
    except Exception as e:
        print(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
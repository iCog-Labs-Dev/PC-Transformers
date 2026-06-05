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

from data_preparation.config import vocab_size
from utils.config_utils import load_best_config

# Silence warnings and Optuna spam
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)

ENERGY_STABILITY_THRESHOLD = 300

# Load best configuration from previous tuning results
BEST_CONFIG = load_best_config()

# FIXED HYPERPARAMETERS FOR PHASE 1 (taken from best config)
FIXED_PARAMS = {
    "n_blocks": BEST_CONFIG.get("n_blocks", 3),
    "num_heads": BEST_CONFIG.get("num_heads", 10),
    "n_embed": BEST_CONFIG.get("n_embed", 160),
    "embed_mult": BEST_CONFIG.get("n_embed", 160) // BEST_CONFIG.get("num_heads", 10),
    "T": BEST_CONFIG.get("T", 20),
    "batch_size": BEST_CONFIG.get("batch_size", 8),
    "block_size": BEST_CONFIG.get("block_size", 208),
    "lr": BEST_CONFIG.get("lr", 3.223786832283688e-05),
    "inference_lr": BEST_CONFIG.get("inference_lr", 0.096),
    "dropout": BEST_CONFIG.get("dropout", 0.46876145412214615),
    "alpha": BEST_CONFIG.get("alpha", 0.5),
    "internal_energy_fn_name": BEST_CONFIG.get("internal_energy_fn_name", "pc_e"),
    "output_energy_fn_name": BEST_CONFIG.get("output_energy_fn_name", "ce"),
    "combined_internal_weight": BEST_CONFIG.get("combined_internal_weight", 0.8779955579743048),
    "combined_output_weight": BEST_CONFIG.get("combined_output_weight", 0.12200444202569516),
    "use_flash_attention": BEST_CONFIG.get("use_flash_attention", False),
    "optimizer_name": BEST_CONFIG.get("optimizer_name", "sgd"),
    "output_optimizer_name": BEST_CONFIG.get("output_optimizer_name", "adam"),
    "optimizer_beta1": BEST_CONFIG.get("optimizer_beta1", 0.9),
    "optimizer_beta2": BEST_CONFIG.get("optimizer_beta2", 0.999),
    "optimizer_eps": BEST_CONFIG.get("optimizer_eps", 1e-8),
    "optimizer_weight_decay": BEST_CONFIG.get("optimizer_weight_decay", 0.01),
    "optimizer_momentum": BEST_CONFIG.get("optimizer_momentum", 0.9),
    "num_epochs": BEST_CONFIG.get("num_epochs", 5),
    "warmup_steps": BEST_CONFIG.get("warmup_steps", 369),
    "clamp_value": BEST_CONFIG.get("clamp_value", 3.0),
    "clip_value": BEST_CONFIG.get("clip_value", 0.01),
}

class IndentedLogger:
    """Custom logger that adds indentation to batch logs for better readability"""
    def __init__(self, trial_num):
        self.trial_num = trial_num
    
    def info(self, msg):
        if msg and "Batch" in msg:
            # Show batch logs with indentation
            print(f"    {msg}")
        elif msg and ("Epoch" in msg or "Training completed" in msg or "Saved checkpoint" in msg):
            # Only show epoch-level logs if needed
            pass

def define_search_space(trial):
    """
    Defines the Phase 1 hyperparameter search space for Optuna trials.
    
    Only clamp_value and clip_value are tuned.
    """
    # Get base values from FIXED_PARAMS
    base_clamp = FIXED_PARAMS.get("clamp_value", 3.0)
    base_clip = FIXED_PARAMS.get("clip_value", 0.01)
    
    return {
        "clamp_value": trial.suggest_float("clamp_value", 0.1, 5, log=True),
        "clip_value": trial.suggest_float("clip_value", 0.01, 0.1, log=True),
    }

def define_search_space_phase2(trial, best_params):
    """
    Defines the hyperparameter search space for Phase 2 (Continuous Fine-Tuning).
    """
    lr = best_params.get("lr", FIXED_PARAMS["lr"])
    inference_lr = best_params.get("inference_lr", FIXED_PARAMS["inference_lr"])
    dropout = best_params.get("dropout", FIXED_PARAMS["dropout"])
    clamp_value = best_params.get("clamp_value", FIXED_PARAMS["clamp_value"])
    clip_value = best_params.get("clip_value", FIXED_PARAMS["clip_value"])
    
    return {
        "lr": trial.suggest_float("lr", max(1e-6, lr * 0.3), min(1e-3, lr * 3.0), log=True),
        "inference_lr": trial.suggest_float("inference_lr", max(0.01, inference_lr * 0.5), min(0.5, inference_lr * 2.0), log=True), 
        "dropout": trial.suggest_float("dropout", max(0.0, dropout - 0.1), min(0.6, dropout + 0.1)),
        "clamp_value": trial.suggest_float("clamp_value", max(0.01, clamp_value * 0.5), min(10.0, clamp_value * 2.0), log=True),
        "clip_value": trial.suggest_float("clip_value", max(0.0001, clip_value * 0.5), min(1.0, clip_value * 2.0), log=True),
    }

def create_model(params):
    """
    Assembles the model, config, and data for a single trial.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Merge fixed parameters with tuned parameters
    merged_params = {**FIXED_PARAMS, **params}
    
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=merged_params["block_size"],
        lr=merged_params["lr"],
        inference_lr=merged_params["inference_lr"],
        peak_learning_rate=merged_params["lr"],
        warmup_steps=merged_params["warmup_steps"],
        n_embed=merged_params["n_embed"],
        dropout=merged_params["dropout"],
        T=merged_params["T"],
        num_heads=merged_params["num_heads"],
        n_blocks=merged_params["n_blocks"],
        batch_size=merged_params["batch_size"],
        num_epochs=merged_params["num_epochs"], 
        internal_energy_fn_name=merged_params["internal_energy_fn_name"],
        output_energy_fn_name=merged_params["output_energy_fn_name"],    
        combined_internal_weight=merged_params["combined_internal_weight"],
        combined_output_weight=merged_params["combined_output_weight"],
        use_flash_attention=merged_params["use_flash_attention"],
        alpha=merged_params["alpha"],
        clamp_value=merged_params["clamp_value"],
        clip_value=merged_params["clip_value"],
        optimizer_name=merged_params["optimizer_name"],
        output_optimizer_name=merged_params["output_optimizer_name"],
        optimizer_beta1=merged_params["optimizer_beta1"],
        optimizer_beta2=merged_params["optimizer_beta2"],
        optimizer_eps=merged_params["optimizer_eps"],
        optimizer_weight_decay=merged_params["optimizer_weight_decay"],
        optimizer_momentum=merged_params["optimizer_momentum"],
    )

    model = PCTransformer(config).to(device)
    model.register_all_lateral_weights() 
    train_loader, valid_loader, _ = get_loaders(
        batch_size=merged_params["batch_size"], 
        block_size=merged_params["block_size"], 
        distributed=False
    )
    
    return model, config, train_loader, valid_loader, device

def run_phase1_trial(trial):
    """
    Phase 1 Trial Execution - Only tuning clamp_value and clip_value
    """
    try:
        # Get only clamp and clip values
        tuned_params = define_search_space(trial)
        
        # Print trial header with clamp and clip values
        print(f"\n[Trial {trial.number:3d}] clamp={tuned_params['clamp_value']:7.4f}  clip={tuned_params['clip_value']:8.6f}")
        print(f"  Training...")

        model, config, train_loader, valid_loader, device = create_model(tuned_params)
        start_time = time.time()

        # Create indented logger for this trial
        indented_logger = IndentedLogger(trial.number)
        
        # Run training with indented logger to show batch outputs
        train_energy, train_ppl, _ = train(model, train_loader, config, global_step=0, device=device, logger=indented_logger)
        
        if torch.isnan(torch.tensor(train_energy)) or train_energy > ENERGY_STABILITY_THRESHOLD:
            print(f"  ✗ Pruned (energy={train_energy:.1f})")
            raise optuna.TrialPruned()

        # Evaluate on validation set
        print(f"  Evaluating...")
        val_energy, val_ppl = evaluate(model, config, valid_loader, max_batches=10, device=device)
        
        total_time = time.time() - start_time
        
        # Store results
        trial.set_user_attr("val_ppl", float(val_ppl))
        trial.set_user_attr("val_energy", float(val_energy))
        trial.set_user_attr("time", total_time)
        
        # Store all parameters
        full_params = {**FIXED_PARAMS, **tuned_params}
        for key, value in full_params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(f"  ✓ Energy={val_energy:8.2f}  PPL={val_ppl:7.2f}  ({total_time:.1f}s)")
        return float(val_energy)

    except Exception as e:
        if not isinstance(e, optuna.TrialPruned):
            print(f"  ✗ Failed: {str(e)[:50]}")
        raise optuna.TrialPruned()
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def run_phase2_trial(trial, best_params):
    """
    Phase 2 Trial Execution - Fine-tuning continuous parameters
    """
    try:
        # Get continuous parameters to tune
        continuous_params = define_search_space_phase2(trial, best_params)
        
        # Merge all parameters
        params = {**FIXED_PARAMS, **continuous_params}
        
        # Print trial header with tuned parameters
        print(f"\n[Trial {trial.number:3d}] lr={params['lr']:.2e}  inf_lr={params['inference_lr']:.4f}  drop={params['dropout']:.4f}  clamp={params['clamp_value']:.4f}  clip={params['clip_value']:.6f}")
        print(f"  Training...")

        model, config, train_loader, valid_loader, device = create_model(params)
        start_time = time.time()

        # Create indented logger for this trial
        indented_logger = IndentedLogger(trial.number)
        
        # Run training with indented logger
        train_energy, train_ppl, _ = train(model, train_loader, config, global_step=0, device=device, logger=indented_logger)
        
        if torch.isnan(torch.tensor(train_energy)) or train_energy > ENERGY_STABILITY_THRESHOLD:
            print(f"  ✗ Pruned")
            raise optuna.TrialPruned()

        # Evaluate on validation set
        print(f"  Evaluating...")
        val_energy, val_ppl = evaluate(model, config, valid_loader, max_batches=10, device=device)
        
        total_time = time.time() - start_time
        
        # Store results
        trial.set_user_attr("val_energy", float(val_energy))
        trial.set_user_attr("time", total_time)
        for key, value in params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(f"  ✓ PPL={val_ppl:7.2f}  ({total_time:.1f}s)")
        return float(val_ppl)

    except Exception as e:
        if not isinstance(e, optuna.TrialPruned):
            print(f"  ✗ Failed: {str(e)[:50]}")
        raise optuna.TrialPruned()
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def run_tuning_pipeline():
    """
    Coordinates the two-phase tuning pipeline
    """
    tuning_path = Path("tuning")
    tuning_path.mkdir(exist_ok=True)
    
    # Using JournalStorage to permanently bypass SQLite errors
    storage_file = str(tuning_path / "optuna_journal.log")
    storage = JournalStorage(JournalFileStorage(storage_file))

    print("\n" + "="*60)
    print("PHASE 1: Optimizing clamp_value and clip_value only")
    print("="*60)
    print(f"Fixed Architecture: n_blocks={FIXED_PARAMS['n_blocks']}, n_heads={FIXED_PARAMS['num_heads']}, n_embed={FIXED_PARAMS['n_embed']}, T={FIXED_PARAMS['T']}")
    print(f"Tuning ranges: clamp=[{FIXED_PARAMS['clamp_value']*0.1:.2f}, {FIXED_PARAMS['clamp_value']*10:.1f}]  clip=[{FIXED_PARAMS['clip_value']*0.1:.5f}, {FIXED_PARAMS['clip_value']*10:.2f}]")
    print("-" * 60)
    
    # Phase 1: Optimize clamp and clip values
    study_energy = optuna.create_study(
        study_name="pc_transformer_phase1_clamp_clip",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )

    print("Starting Phase 1 optimization...")
    study_energy.optimize(run_phase1_trial, n_trials=30, n_jobs=1, show_progress_bar=False)

    if not study_energy.best_trial:
        print("No successful trials in Phase 1")
        return None
        
    best_energy = study_energy.best_value
    best_energy_ppl = study_energy.best_trial.user_attrs.get("val_ppl", "N/A")
    best_params = study_energy.best_trial.params
    
    print(f"\n{'='*60}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Best Energy: {best_energy:.4f}")
    print(f"Best PPL: {best_energy_ppl}")
    print(f"Best clamp_value: {best_params.get('clamp_value', 'N/A'):.6f}")
    print(f"Best clip_value: {best_params.get('clip_value', 'N/A'):.6f}")

    # Phase 2: Fine-tune continuous parameters
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tuning continuous parameters")
    print(f"{'='*60}")
    print("Tuning: lr, inference_lr, dropout, clamp_value, clip_value")
    print("-" * 60)
    
    study_ppl = optuna.create_study(
        study_name="pc_transformer_phase2_continuous",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=5,
            multivariate=True,
        )
    )
    
    print("Starting Phase 2 optimization...")
    
    def phase2_trial_wrapper(trial):
        return run_phase2_trial(trial, best_params)

    study_ppl.optimize(phase2_trial_wrapper, n_trials=50, n_jobs=1, show_progress_bar=False)

    if not study_ppl.best_trial:
        print("No successful trials in Phase 2")
        return None
        
    best_ppl = study_ppl.best_value
    final_params = study_ppl.best_trial.params
    
    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Best PPL achieved: {best_ppl:.4f}")
    
    print(f"\nParameter changes from Phase 1 -> Phase 2:")
    for key in ['lr', 'inference_lr', 'dropout', 'clamp_value', 'clip_value']:
        phase1_val = best_params.get(key, FIXED_PARAMS.get(key))
        phase2_val = final_params.get(key)
        if phase1_val is not None and phase2_val is not None:
            change_pct = ((phase2_val - phase1_val) / phase1_val * 100) if phase1_val != 0 else 0
            print(f"  {key}: {phase1_val:.6f} -> {phase2_val:.6f} ({change_pct:+.1f}%)")
    
    # Save results
    output_file = tuning_path / "best_hyperparameters_optimized.txt"
    with open(output_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("BEST HYPERPARAMETERS (AFTER OPTIMIZATION)\n")
        f.write("="*60 + "\n\n")
        
        f.write("FIXED ARCHITECTURE PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        for key, value in FIXED_PARAMS.items():
            if key not in final_params and key not in ['clamp_value', 'clip_value']:
                f.write(f"{key} = {value}\n")
        
        f.write("\n" + "="*60 + "\n\n")
        
        f.write("PHASE 1 - BEST CLAMP/CLIP VALUES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Energy: {best_energy:.6f}\n")
        f.write(f"Corresponding PPL: {best_energy_ppl}\n")
        f.write("-" * 40 + "\n")
        for key in ['clamp_value', 'clip_value']:
            val = best_params.get(key, 'N/A')
            f.write(f"{key} = {val}\n")
        
        f.write("\n" + "="*60 + "\n\n")
        
        f.write("PHASE 2 - OPTIMIZED CONTINUOUS PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best PPL: {best_ppl:.6f}\n")
        f.write("-" * 40 + "\n")
        for key, value in final_params.items():
            f.write(f"{key} = {value}\n")
        
        # Add recommendation
        f.write("\n" + "="*60 + "\n")
        f.write("RECOMMENDED FINAL CONFIGURATION\n")
        f.write("="*60 + "\n")
        final_recommendation = {**FIXED_PARAMS, **final_params}
        for key, value in final_recommendation.items():
            f.write(f"{key} = {value}\n")
    
    print(f"\n✓ Best hyperparameters saved to: {output_file}")
    
    # Calculate improvement
    improvement = 0
    if isinstance(best_energy_ppl, (int, float)):
        improvement = ((best_energy_ppl - best_ppl) / best_energy_ppl * 100) if best_energy_ppl > 0 else 0
    
    return {
        "phase1_best_energy": best_energy,
        "phase1_best_ppl": best_energy_ppl,
        "phase2_best_ppl": best_ppl,
        "phase1_parameters": best_params,
        "phase2_parameters": final_params,
        "fixed_parameters": FIXED_PARAMS,
        "improvement_pct": improvement
    }

def main():
    print("PC TRANSFORMER - TWO-PHASE HYPERPARAMETER TUNING")
    print("="*60)
    print("PHASE 1: Optimize clamp_value and clip_value only")
    print("PHASE 2: Fine-tune lr, inference_lr, dropout, clamp, clip")
    print("="*60)

    try:
        results = run_tuning_pipeline()
        if results:
            print(f"\n{'='*60}")
            print("TUNING COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Final Results:")
            print(f"- Phase 1 Best Energy: {results['phase1_best_energy']:.4f}")
            if isinstance(results['phase1_best_ppl'], (int, float)):
                print(f"- Phase 1 Best PPL: {results['phase1_best_ppl']:.4f}")
            print(f"- Phase 2 Best PPL: {results['phase2_best_ppl']:.4f}")
            if results['improvement_pct'] != 0:
                print(f"- Improvement: {results['improvement_pct']:+.1f}%")
            print(f"\nBest parameters saved to: tuning/best_hyperparameters_optimized.txt")
        else:
            print("Tuning failed or was interrupted.")
    except KeyboardInterrupt:
        print("\nTuning interrupted by user.")
    except Exception as e:
        print(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
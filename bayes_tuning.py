"""
Bayesian Hyperparameter Tuning
"""
import torch
import gc
import psutil
import os
from predictive_coding.config import GPTConfig
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader, valid_loader, tokenizer, pad_token_id
from training import train
from eval import evaluate
from utils.model_utils import reset_pc_modules, pad_collate_fn
from torch.utils.data import DataLoader, Subset
import logging
import time
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_optimal_data_sizes():
    """Determine optimal data sizes based on available memory"""
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_gb >= 8:
            return 3000, 600
        elif gpu_gb >= 4:
            return 2000, 400 
        else:
            return 1200, 240
    else:
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 16:
            return 1500, 300
        else:
            return 800, 160

def create_subset_loaders(batch_size):
    """Create appropriately sized data loaders"""
    train_size, valid_size = get_optimal_data_sizes()
    max_train = len(train_loader.dataset)
    max_valid = len(valid_loader.dataset)

    train_size = min(train_size, max_train)
    valid_size = min(valid_size, max_valid)
        
    train_indices = torch.randperm(max_train)[:train_size]
    train_subset = Subset(train_loader.dataset, train_indices)
    
    valid_indices = torch.randperm(max_valid)[:valid_size]
    valid_subset = Subset(valid_loader.dataset, valid_indices)
        
    train_subset_loader = DataLoader(train_subset, batch_size=batch_size, 
        shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))
    
    valid_subset_loader = DataLoader(valid_subset, batch_size=batch_size, 
        shuffle=False, collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))
    
    return train_subset_loader, valid_subset_loader

def get_dynamic_batch_size(n_embed, block_size):
    """Calculate optimal batch size based on model size"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory - 1.5 * (1024**3) 
        sequence_memory = block_size * n_embed * 4
        batch_size = max(4, min(24, int(available_memory / (sequence_memory * 3000))))
    else:
        batch_size = max(4, min(12, 8))
    
    return batch_size

def normalize_energy(energy_value, energy_fn_name):
    """
    Normalize energy values to comparable scales across different energy functions.
    Based on empirical testing: MSE~1.86, scaled_MSE~0.09, KLD~9.02
    """
    normalization_factors = {
        'mse': 1.0,         
        'scaled_mse': 20.0, 
        'kld': 0.2   
    }
    factor = normalization_factors.get(energy_fn_name, 1.0)
    return energy_value * factor

def get_adaptive_weight(energy_fn_name):
    """
    Get adaptive weight for combining CE loss and energy based on energy function type.

    Args:
        energy_fn_name (str): Name of the energy function

    Returns:
        float: Weight for CE loss (0-1), where (1-weight) is applied to energy
    """
    adaptive_weights = {
        'kld': 0.4, 
        'mse': 0.6, 
        'scaled_mse': 0.6
    }
    return adaptive_weights.get(energy_fn_name, 0.5)

def update_global_config(config):
    """Update global GPTConfig to match trial config - CRITICAL for shape consistency"""
    GPTConfig.num_heads = config.num_heads
    GPTConfig.n_embed = config.n_embed
    GPTConfig.block_size = config.block_size
    GPTConfig.vocab_size = config.vocab_size
    GPTConfig.dropout = config.dropout
    GPTConfig.local_learning_rate = config.local_learning_rate
    GPTConfig.T = config.T
    GPTConfig.n_blocks = config.n_blocks
    GPTConfig.update_bias = config.update_bias
    GPTConfig.use_lateral = config.use_lateral
    GPTConfig.energy_fn_name = config.energy_fn_name
    
def get_dynamic_model_config(trial, vocab_size):
    """Get model configuration with dynamic parameter combinations"""
    n_embed_candidates = list(range(64, 769, 16))
    n_embed = n_embed_candidates[trial.suggest_int('embed_idx', 0, len(n_embed_candidates) - 1)]

    valid_heads = [h for h in range(4, min(16, n_embed // 12) + 1)
                if n_embed % h == 0 and 12 <= n_embed // h <= 128]

    if not valid_heads:
        valid_heads = [h for h in [4, 6, 8, 12, 16]
                    if h <= n_embed and n_embed % h == 0 and n_embed // h >= 8]
        if not valid_heads:
            if n_embed >= 48 and n_embed % 4 == 0:
                logger.warning(f"Forcing num_heads=4 for n_embed={n_embed} (head_dim={n_embed//4})")
            else:
                logger.warning(f"Skipping n_embed={n_embed} - cannot support minimum 4 heads")
                return None
        
    num_heads = valid_heads[trial.suggest_int('head_idx', 0, len(valid_heads) - 1)]
    block_size_candidates = list(range(64, 513, 16))
    block_size = block_size_candidates[trial.suggest_int('block_idx', 0, len(block_size_candidates)-1)]

    n_blocks = trial.suggest_int('n_blocks', 1, 6)
    T = trial.suggest_int('T', 4, 20, log=True)
    base_lr = trial.suggest_float('base_lr', 1e-5, 1e-3, log=True)
    scaled_lr = base_lr * (n_embed / 256) ** 0.5 * (block_size / 256) ** 0.25

    energy_fn_name = ['kld', 'mse', 'scaled_mse'][trial.suggest_int('energy_idx', 0, 2)]
    update_bias = trial.suggest_int('update_bias_int', 0, 1) == 1
    use_lateral = True
    head_dim = n_embed // num_heads
    
    logger.info(
    f"Params: n_embed={n_embed}, block_size={block_size}, num_heads={num_heads} (head_dim={head_dim}), "
    f"n_blocks={n_blocks}, T={T}, energy_fn={energy_fn_name}, update_bias={update_bias}, use_lateral={use_lateral}, "
    f"base_lr={base_lr:.2e}, scaled_lr={scaled_lr:.2e}, valid_heads={valid_heads}")
    
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        dropout=trial.suggest_float('dropout', 0.05, 0.3),
        local_learning_rate=scaled_lr,
        T=T,
        is_holding_error=True,
        num_heads=num_heads,
        n_blocks=n_blocks,
        num_epochs=1,
        update_bias=update_bias,
        use_lateral=use_lateral,
        energy_fn_name=energy_fn_name
    )

def objective(trial):
    """Bayesian Objective function"""
    start_time = time.time()
    model = None

    try:
        logger.info(f"Trial {trial.number}")
        cleanup_memory()
        vocab_size = tokenizer.vocab_size
        config = get_dynamic_model_config(trial, vocab_size)

        if config is None:
            logger.warning(f"Trial {trial.number} skipped - no valid config with min 4 heads")
            return float("inf")

        update_global_config(config)
        try:
            model = PCTransformer(config)            
        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float("inf")
        
        batch_size = get_dynamic_batch_size(config.n_embed, config.block_size)
        train_loader, valid_loader = create_subset_loaders(batch_size=batch_size)
        logger.info(f"Using batch size: {batch_size}")

        if len(train_loader) == 0:
            logger.error("Train loader is empty!")
            return float("inf")
        if len(valid_loader) == 0:
            logger.error("Valid loader is empty!")
            return float("inf")
        
        try:
            model.train()
            _, _ = train(model, train_loader, tokenizer)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            logger.error(f"Training traceback: {traceback.format_exc()}")
            return float("inf")
        
        reset_pc_modules(model)

        try:
            model.eval()
            max_val_batches = min(10, len(valid_loader))
            avg_energy, val_loss = evaluate(model, valid_loader, tokenizer, max_batches=max_val_batches, compute_metrics=False)

            normalized_energy = normalize_energy(avg_energy, config.energy_fn_name)
            adaptive_weight = get_adaptive_weight(config.energy_fn_name)

            combined_objective = adaptive_weight * val_loss + (1 - adaptive_weight) * normalized_energy

            trial_time = time.time() - start_time
            logger.info(f"Trial {trial.number} completed in {trial_time:.1f}s")
            logger.info(f"  CE Loss: {val_loss:.4f}, Energy: {avg_energy:.4f}, Normalized Energy: {normalized_energy:.4f}")
            logger.info(f"  Weight: {adaptive_weight:.2f}, Combined Objective: {combined_objective:.4f}")

            trial.set_user_attr("config", config.__dict__)
            trial.set_user_attr("ce_loss", val_loss)
            trial.set_user_attr("energy", avg_energy)
            trial.set_user_attr("normalized_energy", normalized_energy)
            trial.set_user_attr("combined_objective", combined_objective)
            return combined_objective
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            import traceback
            logger.error(f"Evaluation traceback: {traceback.format_exc()}")
            return float("inf")
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        import traceback
        logger.error(f"Trial traceback: {traceback.format_exc()}")
        return float("inf")
    finally:
        if model is not None:
            try:
                reset_pc_modules(model)
                del model
            except:
                pass
        cleanup_memory()

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
            interval_steps=1))
    
    logger.info(f"Starting bayesian tuning with {n_trials} trials")
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        

        logger.info("Optimization completed!")
        if study.best_trial:
            trial = study.best_trial
            logger.info(f"Best trial: {trial.number}. Best combined objective: {trial.value:.5f}")


            ce_loss = trial.user_attrs.get("ce_loss", "N/A")
            energy = trial.user_attrs.get("energy", "N/A")
            normalized_energy = trial.user_attrs.get("normalized_energy", "N/A")
            logger.info(f"  CE Loss: {ce_loss:.4f}, Raw Energy: {energy:.4f}, Normalized Energy: {normalized_energy:.4f}")

            logger.info("Best parameters:")
            config_dict = trial.user_attrs.get("config")
            if config_dict:
                adaptive_weight = get_adaptive_weight(config_dict['energy_fn_name'])
                logger.info(f"  Energy function: {config_dict['energy_fn_name']} (CE weight: {adaptive_weight:.2f})")
                logger.info(
                    f"  n_embed={config_dict['n_embed']}, block_size={config_dict['block_size']}, num_heads={config_dict['num_heads']} "
                    f"(head_dim={config_dict['n_embed'] // config_dict['num_heads']}), "
                    f"n_blocks={config_dict['n_blocks']}, T={config_dict['T']}, "
                    f"update_bias={config_dict['update_bias']}, use_lateral={config_dict['use_lateral']}, "
                    f"scaled_lr={config_dict['local_learning_rate']:.2e}")


            results_path = f"{study_name}_results.txt"
            with open(results_path, "w") as f:
                f.write(f"WEIGHTED OPTIMIZATION RESULTS\n")
                f.write(f"=====================================\n\n")
                f.write(f"Best combined objective: {trial.value:.4f}\n")
                f.write(f"  CE Loss: {trial.user_attrs.get('ce_loss', 'N/A'):.4f}\n")
                f.write(f"  Raw Energy: {trial.user_attrs.get('energy', 'N/A'):.4f}\n")
                f.write(f"  Normalized Energy: {trial.user_attrs.get('normalized_energy', 'N/A'):.4f}\n\n")

                config = trial.user_attrs.get("config")
                if config:
                    adaptive_weight = get_adaptive_weight(config['energy_fn_name'])
                    f.write(f"Optimization Strategy:\n")
                    f.write(f"  Energy function: {config['energy_fn_name']}\n")
                    f.write(f"  CE Loss weight: {adaptive_weight:.2f}\n")
                    f.write(f"  Energy weight: {1-adaptive_weight:.2f}\n\n")

                    f.write("Best parameters:\n")
                    f.write(f"  n_embed: {config['n_embed']}\n")
                    f.write(f"  block_size: {config['block_size']}\n")
                    f.write(f"  num_heads: {config['num_heads']}\n")
                    f.write(f"  head_dim: {config['n_embed'] // config['num_heads']}\n")
                    f.write(f"  n_blocks: {config['n_blocks']}\n")
                    f.write(f"  T: {config['T']}\n")
                    f.write(f"  dropout: {config['dropout']}\n")
                    f.write(f"  energy_fn: {config['energy_fn_name']}\n")
                    f.write(f"  update_bias: {config['update_bias']}\n")
                    f.write(f"  use_lateral: {config['use_lateral']}\n")
                    f.write(f"  scaled_lr: {config['local_learning_rate']:.2e}\n")
            
            logger.info(f"Results saved to {results_path}")
        return study
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted")
        return study

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    study = run_tuning(n_trials= 30, study_name="bayesian_tuning")
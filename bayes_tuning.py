"""
Bayesian Hyperparameter Tuning
"""

import optuna
import torch
import gc
import psutil
import os
from predictive_coding.config import GPTConfig
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader, valid_loader
from training import train
from eval import evaluate
from utils.model_utils import load_tokenizer, reset_pc_modules
from torch.utils.data import DataLoader, Subset
import logging
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        return memory_mb, gpu_memory_mb
    return memory_mb, 0

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

def create_subset_loaders(train_size=None, valid_size=None, batch_size=16):
    """Create appropriately sized data loaders"""
    if train_size is None or valid_size is None:
        train_size, valid_size = get_optimal_data_sizes()
    

    max_train = len(train_loader.dataset)
    max_valid = len(valid_loader.dataset)
    train_size = min(train_size, max_train)
    valid_size = min(valid_size, max_valid)
    
    logger.info(f"Using {train_size} training samples and {valid_size} validation samples")
    

    train_indices = torch.randperm(max_train)[:train_size]
    train_subset = Subset(train_loader.dataset, train_indices)
    train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    valid_indices = torch.randperm(max_valid)[:valid_size]
    valid_subset = Subset(valid_loader.dataset, valid_indices)
    valid_subset_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    
    return train_subset_loader, valid_subset_loader

def get_dynamic_batch_size(n_embed, n_blocks, block_size):
    """Calculate optimal batch size based on model size"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory - 1.5 * (1024**3)  # Reserve 1.5GB
        sequence_memory = block_size * n_embed * 4
        estimated_batch_size = max(4, min(24, int(available_memory / (sequence_memory * 3000))))
    else:
        estimated_batch_size = max(4, min(12, 8))
    
    return estimated_batch_size

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
    
    logger.info(f"Updated global config: n_embed={GPTConfig.n_embed}, num_heads={GPTConfig.num_heads}")

def get_safe_model_config(trial, vocab_size):
    """Get model configuration with guaranteed shape compatibility"""
    
    safe_configs = [
        (64, [2, 4]),
        (128, [2, 4, 8]),
        (256, [2, 4, 8]),
        (512, [2, 4, 8]),
    ]


    config_idx = trial.suggest_int('config_idx', 0, len(safe_configs) - 1)
    n_embed, valid_head_options = safe_configs[config_idx]
    

    head_idx = trial.suggest_int('head_idx', 0, len(valid_head_options) - 1)
    num_heads = valid_head_options[head_idx]
    

    block_choices = [64, 128, 256, 512]
    block_idx = trial.suggest_int('block_idx', 0, len(block_choices) - 1)
    block_size = block_choices[block_idx]
    

    n_blocks = trial.suggest_int('n_blocks', 1, 4)
    T = trial.suggest_int('T', 5, 10)
    

    base_lr = trial.suggest_float('base_lr', 1e-5, 1e-3, log=True)
    lr_scale = (n_embed / 256) ** 0.5 * (block_size / 256) ** 0.25
    scaled_lr = base_lr * lr_scale
    

    energy_choices = ['kld', 'mse', 'scaled_mse']
    energy_idx = trial.suggest_int('energy_idx', 0, len(energy_choices) - 1)
    energy_fn_name = energy_choices[energy_idx]
    

    update_bias = trial.suggest_int('update_bias_int', 0, 1) == 1
    use_lateral = True
    
    head_dim = n_embed // num_heads
    
    logger.info(f"Trial {trial.number} config:")
    logger.info(f"  n_embed={n_embed}, block_size={block_size}, num_heads={num_heads} (head_dim={head_dim})")
    logger.info(f"  n_blocks={n_blocks}, T={T}, energy_fn={energy_fn_name}")
    logger.info(f"  update_bias={update_bias}, use_lateral={use_lateral}")
    logger.info(f"  base_lr={base_lr:.2e}, scaled_lr={scaled_lr:.2e}")
    
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        dropout=trial.suggest_float('dropout', 0.05, 0.25),
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
    """Completely fixed objective function"""
    start_time = time.time()
    model = None
    
    try:
        logger.info(f"Starting trial {trial.number}")
        initial_memory, initial_gpu_memory = get_memory_usage()
        logger.info(f"Initial memory: {initial_memory:.1f}MB RAM, {initial_gpu_memory:.1f}MB GPU")
        
        cleanup_memory()
        
        tokenizer = load_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        config = get_safe_model_config(trial, vocab_size)
        

        update_global_config(config)
        

        try:
            model = PCTransformer(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            model_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model created successfully: {model_params:,} parameters")
            
        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float("inf")
        

        optimal_batch_size = get_dynamic_batch_size(config.n_embed, config.n_blocks, config.block_size)
        train_subset_loader, valid_subset_loader = create_subset_loaders(batch_size=optimal_batch_size)
        
        logger.info(f"Using batch size: {optimal_batch_size}")
        

        try:
            model.train()
            avg_energy, _ = train(model, train_subset_loader)
            
            mid_memory, mid_gpu_memory = get_memory_usage()
            logger.info(f"After training: {mid_memory:.1f}MB RAM, {mid_gpu_memory:.1f}MB GPU")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            logger.error(f"Training traceback: {traceback.format_exc()}")
            return float("inf")
        

        try:
            model.eval()
            with torch.no_grad():
                max_val_batches = min(10, len(valid_subset_loader))
                avg_energy_val, val_loss = evaluate(model, valid_subset_loader, max_batches=max_val_batches, compute_metrics=False)
            
            trial_time = time.time() - start_time
            logger.info(f"Trial {trial.number} completed in {trial_time:.1f}s")
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            return val_loss
            
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

def run_tuning(n_trials=20, study_name="bayesian_tuning"):
    """Run completely fixed hyperparameter tuning"""
    
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    
    logger.info(f"Starting bayesian tuning with {n_trials} trials")
    
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info("Optimization completed!")
        if study.best_trial:
            trial = study.best_trial
            logger.info(f"Best loss: {trial.value:.4f}")
            logger.info("Best parameters:")
            for key, value in trial.params.items():
                logger.info(f"  {key}: {value}")
            

            results_path = f"{study_name}_results.txt"
            with open(results_path, "w") as f:
                f.write(f"Best validation loss: {trial.value:.4f}\n\n")
                f.write("Best parameters:\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"Results saved to {results_path}")
        
        return study
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted")
        return study

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    study = run_tuning(n_trials=20, study_name="bayesian_tuning")

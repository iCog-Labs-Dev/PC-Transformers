"""
Bayesian Hyperparameter Tuning
"""

import optuna
import torch
import gc
import psutil
import os
import sys
import contextlib
from io import StringIO
from predictive_coding.config import GPTConfig
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader, valid_loader
from training import train
from eval import evaluate
from utils.model_utils import load_tokenizer, reset_pc_modules, pad_collate_fn
from torch.utils.data import DataLoader, Subset
import logging
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (for hiding diversity gradient warnings)"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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

    try:
        max_train = len(train_loader.dataset)
        max_valid = len(valid_loader.dataset)
        logger.info(f"Original dataset sizes: train={max_train}, valid={max_valid}")
    except Exception as e:
        logger.error(f"Error accessing original data loaders: {e}")
        raise
    
    train_size = min(train_size, max_train)
    valid_size = min(valid_size, max_valid)
    
    logger.info(f"Using {train_size} training samples and {valid_size} validation samples")
    

    train_indices = torch.randperm(max_train)[:train_size]
    train_subset = Subset(train_loader.dataset, train_indices)
    
    valid_indices = torch.randperm(max_valid)[:valid_size]
    valid_subset = Subset(valid_loader.dataset, valid_indices)
    

    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    

    train_subset_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
    )
    
    valid_subset_loader = DataLoader(
        valid_subset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
    )
    
    return train_subset_loader, valid_subset_loader

def get_dynamic_batch_size(n_embed, n_blocks, block_size):
    """Calculate optimal batch size based on model size"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory - 1.5 * (1024**3) 
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

def get_dynamic_model_config(trial, vocab_size):
    """Get model configuration with dynamic parameter combinations like nanoGPT"""
    
    n_embed_candidates = []
    
    for base in range(64, 769, 16):
        n_embed_candidates.append(base)
    
    special_values = [384, 576, 640, 704]
    for val in special_values:
        if val not in n_embed_candidates and 64 <= val <= 768:
            n_embed_candidates.append(val)
    
    n_embed_candidates = sorted(n_embed_candidates)
    
    embed_idx = trial.suggest_int('embed_idx', 0, len(n_embed_candidates) - 1)
    n_embed = n_embed_candidates[embed_idx]
    
    valid_heads = []
    min_heads = 4
    max_heads = min(16, n_embed // 12)
    
    for h in range(min_heads, max_heads + 1):
        if n_embed % h == 0:
            head_dim = n_embed // h
            if 12 <= head_dim <= 128: 
                valid_heads.append(h)
    
    if not valid_heads:
        for h in [4, 6, 8, 12, 16]:
            if h <= n_embed and n_embed % h == 0:
                head_dim = n_embed // h
                if head_dim >= 8:
                    valid_heads.append(h)
        
        if not valid_heads:
            if n_embed >= 48 and n_embed % 4 == 0: 
                logger.warning(f"Forcing num_heads=4 for n_embed={n_embed} (head_dim={n_embed//4})")
            else:
                logger.warning(f"Skipping n_embed={n_embed} - cannot support minimum 4 heads")
                return None 
    
    head_idx = trial.suggest_int('head_idx', 0, len(valid_heads) - 1)
    num_heads = valid_heads[head_idx]
    
    block_candidates = [64, 96, 128, 192, 256, 320, 384, 512]
    block_idx = trial.suggest_int('block_idx', 0, len(block_candidates) - 1)
    block_size = block_candidates[block_idx]
    
    n_blocks = trial.suggest_int('n_blocks', 1, 6)
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
    
    logger.info(f"Trial {trial.number} dynamic config:")
    logger.info(f"  n_embed={n_embed}, block_size={block_size}, num_heads={num_heads} (head_dim={head_dim})")
    logger.info(f"  n_blocks={n_blocks}, T={T}, energy_fn={energy_fn_name}")
    logger.info(f"  update_bias={update_bias}, use_lateral={use_lateral}")
    logger.info(f"  base_lr={base_lr:.2e}, scaled_lr={scaled_lr:.2e}")
    logger.info(f"  Valid heads for n_embed={n_embed}: {valid_heads}")
    
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
        logger.info(f"Starting trial {trial.number}")
        initial_memory, initial_gpu_memory = get_memory_usage()
        logger.info(f"Initial memory: {initial_memory:.1f}MB RAM, {initial_gpu_memory:.1f}MB GPU")
        
        cleanup_memory()
        
        tokenizer = load_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        config = get_dynamic_model_config(trial, vocab_size)

        if config is None:
            logger.warning(f"Trial {trial.number} skipped - no valid config with min 4 heads")
            return float("inf")

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
        logger.info(f"Train loader length: {len(train_subset_loader)}, Valid loader length: {len(valid_subset_loader)}")


        if len(train_subset_loader) == 0:
            logger.error("Train loader is empty!")
            return float("inf")
        if len(valid_subset_loader) == 0:
            logger.error("Valid loader is empty!")
            return float("inf")
        
        try:
            model.train()
            avg_energy, _ = train(model, train_subset_loader, tokenizer)
            
            mid_memory, mid_gpu_memory = get_memory_usage()
            logger.info(f"After training: {mid_memory:.1f}MB RAM, {mid_gpu_memory:.1f}MB GPU")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            logger.error(f"Training traceback: {traceback.format_exc()}")
            return float("inf")
        
        try:
            model.eval()
            logger.info("Starting evaluation")
            
            with torch.no_grad(), suppress_stdout():
                max_val_batches = min(10, len(valid_subset_loader))
                avg_energy_val, val_loss = evaluate(model, valid_subset_loader, tokenizer, max_batches=max_val_batches, compute_metrics=False)
            
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
    
    logger.info(f"Starting bayesian tuning with {n_trials} trials")
    
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Results
        logger.info("Optimization completed!")
        if study.best_trial:
            trial = study.best_trial
            logger.info(f"Best loss: {trial.value:.4f}")
            logger.info("Best parameters:")
            for key, value in trial.params.items():
                logger.info(f"  {key}: {value}")
            
            # Save results
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
    
    study = run_tuning(n_trials=30, study_name="bayesian_tuning")

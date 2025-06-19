#!/usr/bin/env python3
"""
Advanced Bayesian Hyperparameter Tuning with Adaptive Strategies
This version includes:
- Multi-stage optimization (coarse -> fine)
- Adaptive data sizing based on trial performance
- Dynamic early stopping
- Resource-aware parameter selection
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
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveBayesianTuner:
    def __init__(self, study_name="adaptive_pc_transformer_tuning"):
        self.study_name = study_name
        self.tokenizer = load_tokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.trial_history = []
        
        # Adaptive parameters
        self.current_stage = "coarse"  # coarse -> fine -> ultra_fine
        self.stage_thresholds = {"coarse": 15, "fine": 35}
        self.best_loss_so_far = float('inf')
        
    def get_memory_info(self):
        """Get detailed memory information"""
        process = psutil.Process(os.getpid())
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_used_mb = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_total_gb = gpu_props.total_memory / (1024**3)
            gpu_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
            return ram_gb, ram_used_mb, gpu_total_gb, gpu_used_mb
        return ram_gb, ram_used_mb, 0, 0
    
    def get_adaptive_data_sizes(self, trial_number, model_complexity):
        """Adaptively determine data sizes based on trial progress and model complexity"""
        ram_gb, _, gpu_gb, _ = self.get_memory_info()
        
        # Base sizes on available resources
        if gpu_gb >= 8:
            base_train, base_valid = 8000, 1500
        elif gpu_gb >= 4:
            base_train, base_valid = 5000, 1000
        elif gpu_gb >= 2:
            base_train, base_valid = 3000, 600
        else:
            base_train, base_valid = 1500, 300
            
        # Adjust based on optimization stage
        if self.current_stage == "coarse":
            # Use smaller datasets for initial exploration
            train_size = int(base_train * 0.6)
            valid_size = int(base_valid * 0.6)
        elif self.current_stage == "fine":
            # Use larger datasets for refinement
            train_size = int(base_train * 0.8)
            valid_size = int(base_valid * 0.8)
        else:  # ultra_fine
            # Use full allocated datasets for final optimization
            train_size = base_train
            valid_size = base_valid
            
        # Adjust based on model complexity
        complexity_factor = model_complexity / 1000000  # Normalize by 1M parameters
        if complexity_factor > 2:
            train_size = int(train_size * 0.7)
            valid_size = int(valid_size * 0.7)
        elif complexity_factor > 1:
            train_size = int(train_size * 0.85)
            valid_size = int(valid_size * 0.85)
            
        # Ensure we don't exceed dataset limits
        max_train = len(train_loader.dataset)
        max_valid = len(valid_loader.dataset)
        train_size = min(train_size, max_train)
        valid_size = min(valid_size, max_valid)
        
        return train_size, valid_size
    
    def get_stage_config(self, trial, stage):
        """Get configuration based on optimization stage"""
        if stage == "coarse":
            # Coarse exploration - wider ranges, simpler models
            n_embed = trial.suggest_categorical('n_embed', [64, 128, 256])
            block_size = trial.suggest_categorical('block_size', [64, 128, 256])
            n_blocks = trial.suggest_int('n_blocks', 1, 3)
            T = trial.suggest_int('T', 3, 5)
            
        elif stage == "fine":
            # Fine-tuning around promising areas
            n_embed = trial.suggest_categorical('n_embed', [128, 256, 512])
            block_size = trial.suggest_categorical('block_size', [128, 256, 512])
            n_blocks = trial.suggest_int('n_blocks', 2, 4)
            T = trial.suggest_int('T', 4, 7)
            
        else:  # ultra_fine
            # Ultra-fine optimization
            n_embed = trial.suggest_categorical('n_embed', [256, 512, 768])
            block_size = trial.suggest_categorical('block_size', [256, 512, 1024])
            n_blocks = trial.suggest_int('n_blocks', 2, 5)
            T = trial.suggest_int('T', 5, 8)
        
        # Common parameters across stages
        # Fixed head choices to avoid dynamic categorical distribution
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])

        # Ensure head configuration is valid for the chosen embedding size
        max_heads = min(8, n_embed // 32)
        if num_heads > max_heads or n_embed % num_heads != 0:
            # Fall back to largest valid head count
            valid_heads = [h for h in [1, 2, 4, 8] if h <= max_heads and n_embed % h == 0]
            num_heads = max(valid_heads) if valid_heads else 1
        
        # Dynamic learning rate
        base_lr = trial.suggest_float('base_lr', 1e-5, 1e-3, log=True)
        lr_scale = (n_embed / 256) ** 0.5 * (block_size / 256) ** 0.25
        scaled_lr = base_lr * lr_scale
        
        return GPTConfig(
            vocab_size=self.vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            dropout=trial.suggest_float('dropout', 0.05, 0.4),
            local_learning_rate=scaled_lr,
            T=T,
            is_holding_error=True,
            num_heads=num_heads,
            n_blocks=n_blocks,
            num_epochs=1,
            update_bias=trial.suggest_categorical('update_bias', [True, False]),
            use_lateral=trial.suggest_categorical('use_lateral', [True, False]),
            energy_fn_name=trial.suggest_categorical('energy_fn_name', ['kld', 'mse', 'scaled_mse', 'l1'])
        )
    
    def update_stage(self, trial_number):
        """Update optimization stage based on progress"""
        if trial_number >= self.stage_thresholds.get("fine", 35) and self.current_stage != "ultra_fine":
            self.current_stage = "ultra_fine"
            logger.info(f"Switching to ultra-fine optimization stage at trial {trial_number}")
        elif trial_number >= self.stage_thresholds.get("coarse", 15) and self.current_stage == "coarse":
            self.current_stage = "fine"
            logger.info(f"Switching to fine optimization stage at trial {trial_number}")
    
    def objective(self, trial):
        """Adaptive objective function"""
        start_time = time.time()
        model = None
        
        try:
            # Update optimization stage
            self.update_stage(trial.number)
            
            logger.info(f"Starting trial {trial.number} (Stage: {self.current_stage})")
            
            # Clean up before starting
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Get stage-appropriate configuration
            config = self.get_stage_config(trial, self.current_stage)
            
            # Create model
            model = PCTransformer(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            model_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model: {model_params:,} parameters, {config.n_embed}d, {config.n_blocks} blocks, {config.block_size} seq_len")
            
            # Get adaptive data sizes
            train_size, valid_size = self.get_adaptive_data_sizes(trial.number, model_params)
            
            # Dynamic batch size
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = gpu_memory - 2 * (1024**3)  # Reserve 2GB
                sequence_memory = config.block_size * config.n_embed * 4
                batch_size = max(4, min(64, int(available_memory / (sequence_memory * 2000))))
            else:
                batch_size = max(4, min(32, 16))
            
            # Create data loaders
            train_indices = torch.randperm(len(train_loader.dataset))[:train_size]
            train_subset = Subset(train_loader.dataset, train_indices)
            train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            
            valid_indices = torch.randperm(len(valid_loader.dataset))[:valid_size]
            valid_subset = Subset(valid_loader.dataset, valid_indices)
            valid_subset_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"Data: {train_size} train, {valid_size} valid samples, batch_size={batch_size}")
            
            # Training
            model.train()
            avg_energy, _ = train(model, train_subset_loader)
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                max_val_batches = min(30 if self.current_stage == "ultra_fine" else 20, len(valid_subset_loader))
                avg_energy_val, val_loss = evaluate(model, valid_subset_loader, max_batches=max_val_batches, compute_metrics=False)
            
            trial_time = time.time() - start_time
            logger.info(f"Trial {trial.number} completed in {trial_time:.1f}s, loss: {val_loss:.4f}")
            
            # Track best performance
            if val_loss < self.best_loss_so_far:
                self.best_loss_so_far = val_loss
                logger.info(f"New best loss: {val_loss:.4f}")
            
            return val_loss
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return float("inf")
        finally:
            if model is not None:
                try:
                    reset_pc_modules(model)
                    del model
                except:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def run_optimization(self, n_trials=60):
        """Run the adaptive optimization"""
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            storage=f'sqlite:///{self.study_name}.db',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=8,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        logger.info(f"Starting adaptive Bayesian optimization with {n_trials} trials")
        
        try:
            study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
            
            # Save results
            logger.info("Optimization completed!")
            if study.best_trial:
                logger.info(f"Best loss: {study.best_trial.value:.4f}")
                logger.info("Best parameters:")
                for key, value in study.best_trial.params.items():
                    logger.info(f"  {key}: {value}")
            
            return study
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted")
            return study

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    tuner = AdaptiveBayesianTuner("adaptive_pc_transformer_tuning")
    study = tuner.run_optimization(n_trials=60)

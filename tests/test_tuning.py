"""
Tests for the tuning module.

This module tests the Bayesian hyperparameter optimization components.
"""

import torch
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from tuning.config import get_dynamic_model_config, update_global_config, normalize_energy
from tuning.dataloader import get_optimal_data_sizes, get_dynamic_batch_size
from tuning.tuning_logs import initialize_logs, log_trial_to_summary, log_trial_to_detailed_log, write_final_results
from tuning.monitor_tuning import monitor_study
from predictive_coding.config import GPTConfig

# ===================== Config Tests (config.py) =====================

def test_get_dynamic_model_config_basic():
    """
    Test that get_dynamic_model_config creates a valid configuration with basic parameters.
    """
    mock_trial = Mock()
    # Provide all the values that suggest_int and suggest_float will be called with
    mock_trial.suggest_int.side_effect = [128, 0, 128, 2, 8, 200, 0, 1]  # n_embed, head_idx, block_size, n_blocks, T, warmup_steps, energy_idx, update_bias_int
    mock_trial.suggest_float.side_effect = [0.1, 1e-4]  # dropout, base_lr
    
    config = get_dynamic_model_config(mock_trial, vocab_size=100, flash=False)
    
    assert config is not None
    assert isinstance(config, GPTConfig)
    assert config.vocab_size == 100
    assert config.n_embed == 128
    assert config.block_size == 128
    assert config.n_blocks == 2
    assert config.T == 8
    assert config.num_heads == 4  # Should be valid for n_embed=128
    assert config.dropout == 0.1
    assert config.energy_fn_name == "kld"
    assert config.update_bias == True

def test_get_dynamic_model_config_flash_attention():
    """
    Test that get_dynamic_model_config properly handles flash attention flag.
    """
    mock_trial = Mock()
    mock_trial.suggest_int.side_effect = [128, 0, 128, 2, 8, 200, 0, 1]
    mock_trial.suggest_float.side_effect = [0.1, 1e-4]
    
    config = get_dynamic_model_config(mock_trial, vocab_size=100, flash=True)
    
    assert config is not None
    assert config.use_flash_attention == True

def test_update_global_config_dict():
    """
    Test that update_global_config properly updates global config with dictionary input.
    """
    config_dict = {
        'num_heads': 8,
        'n_embed': 256,
        'block_size': 128,
        'dropout': 0.2,
        'energy_fn_name': 'mse'
    }
    
    # Store original values
    original_values = {}
    for key in config_dict.keys():
        if hasattr(GPTConfig, key):
            original_values[key] = getattr(GPTConfig, key)
    
    try:
        update_global_config(config_dict)
        
        # Check that values were updated
        for key, value in config_dict.items():
            if hasattr(GPTConfig, key):
                assert getattr(GPTConfig, key) == value
    finally:
        # Restore original values
        for key, value in original_values.items():
            setattr(GPTConfig, key, value)

def test_update_global_config_object():
    """
    Test that update_global_config properly updates global config with object input.
    """
    config = GPTConfig(vocab_size=100, block_size=64)
    config.num_heads = 4
    config.n_embed = 128
    config.dropout = 0.15
    
    # Store original values
    original_values = {}
    for key in ['num_heads', 'n_embed', 'dropout']:
        if hasattr(GPTConfig, key):
            original_values[key] = getattr(GPTConfig, key)
    
    try:
        update_global_config(config)
        
        # Check that values were updated
        assert getattr(GPTConfig, 'num_heads') == 4
        assert getattr(GPTConfig, 'n_embed') == 128
        assert getattr(GPTConfig, 'dropout') == 0.15
    finally:
        # Restore original values
        for key, value in original_values.items():
            setattr(GPTConfig, key, value)

def test_normalize_energy():
    """
    Test that normalize_energy properly scales energy values for different energy functions.
    """
    # Test MSE normalization
    mse_energy = normalize_energy(1.0, 'mse')
    assert mse_energy == 1.0
    
    # Test scaled MSE normalization
    scaled_mse_energy = normalize_energy(1.0, 'scaled_mse')
    assert scaled_mse_energy == 20.0
    
    # Test KLD normalization
    kld_energy = normalize_energy(1.0, 'kld')
    assert kld_energy == 0.2
    
    # Test unknown energy function
    unknown_energy = normalize_energy(1.0, 'unknown')
    assert unknown_energy == 1.0

# ===================== Dataloader Tests (dataloader.py) =====================

@patch('torch.cuda.is_available')
@patch('torch.cuda.get_device_properties')
def test_get_optimal_data_sizes_gpu_large(mock_get_props, mock_cuda_available):
    """
    Test get_optimal_data_sizes with large GPU memory.
    """
    mock_cuda_available.return_value = True
    mock_props = Mock()
    mock_props.total_memory = 16 * (1024**3)  # 16GB
    mock_get_props.return_value = mock_props
    
    train_size, valid_size = get_optimal_data_sizes()
    
    assert train_size == 20000
    assert valid_size == 5000

@patch('torch.cuda.is_available')
@patch('torch.cuda.get_device_properties')
def test_get_optimal_data_sizes_gpu_medium(mock_get_props, mock_cuda_available):
    """
    Test get_optimal_data_sizes with medium GPU memory.
    """
    mock_cuda_available.return_value = True
    mock_props = Mock()
    mock_props.total_memory = 6 * (1024**3)  # 6GB
    mock_get_props.return_value = mock_props
    
    train_size, valid_size = get_optimal_data_sizes()
    
    assert train_size == 2000
    assert valid_size == 400

@patch('torch.cuda.is_available')
@patch('psutil.virtual_memory')
def test_get_optimal_data_sizes_cpu(mock_virtual_memory, mock_cuda_available):
    """
    Test get_optimal_data_sizes with CPU only.
    """
    mock_cuda_available.return_value = False
    mock_memory = Mock()
    mock_memory.total = 32 * (1024**3)  # 32GB
    mock_virtual_memory.return_value = mock_memory
    
    train_size, valid_size = get_optimal_data_sizes()
    
    assert train_size == 1500
    assert valid_size == 300

@patch('torch.cuda.is_available')
@patch('torch.cuda.get_device_properties')
def test_get_dynamic_batch_size_gpu(mock_get_props, mock_cuda_available):
    """
    Test get_dynamic_batch_size with GPU available.
    """
    mock_cuda_available.return_value = True
    mock_props = Mock()
    mock_props.total_memory = 8 * (1024**3)  # 8GB
    mock_get_props.return_value = mock_props
    
    batch_size = get_dynamic_batch_size(n_embed=256, block_size=128)
    
    assert isinstance(batch_size, int)
    assert 4 <= batch_size <= 24

@patch('torch.cuda.is_available')
def test_get_dynamic_batch_size_cpu(mock_cuda_available):
    """
    Test get_dynamic_batch_size with CPU only.
    """
    mock_cuda_available.return_value = False
    
    batch_size = get_dynamic_batch_size(n_embed=256, block_size=128)
    
    assert isinstance(batch_size, int)
    assert 4 <= batch_size <= 12

# ===================== Tuning Logs Tests (tuning_logs.py) =====================

def test_initialize_logs():
    """
    Test that initialize_logs creates proper log files with correct headers.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        study_name = "test_study"
        summary_path, trials_path = initialize_logs(study_name)
        
        # Check that files were created
        assert os.path.exists(summary_path)
        assert os.path.exists(trials_path)
        
        # Check summary file content
        with open(summary_path, 'r') as f:
            content = f.read()
            assert "BAYESIAN TUNING SUMMARY" in content
            assert "Objective: Minimize combined energy" in content
            assert "Trial Progress:" in content
        
        # Check trials file content
        with open(trials_path, 'r') as f:
            content = f.read()
            assert "DETAILED TRIAL RESULTS" in content
            assert "Objective: Minimize combined energy" in content

def test_log_trial_to_summary():
    """
    Test that log_trial_to_summary writes trial data in correct format.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        study_name = "test_study"
        summary_path, _ = initialize_logs(study_name)
        
        # Create mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.user_attrs = {
            "ce_loss": 0.5,
            "energy": 1.2,
            "normalized_energy": 0.8,
            "combined_energy": 1.3,
            "trial_time": 120.5,
            "config": {"energy_fn_name": "mse"}
        }
        
        log_trial_to_summary(summary_path, mock_trial)
        
        # Check that data was written
        with open(summary_path, 'r') as f:
            content = f.read()
            assert "1" in content  # Trial number
            assert "120.5" in content  # Trial time
            assert "0.5" in content  # CE loss
            assert "1.2" in content  # Energy
            assert "0.8" in content  # Normalized energy
            assert "1.3" in content  # Combined energy
            assert "mse" in content  # Energy function

def test_log_trial_to_detailed_log():
    """
    Test that log_trial_to_detailed_log writes detailed trial information.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        study_name = "test_study"
        _, trials_path = initialize_logs(study_name)
        
        # Create mock trial and config
        mock_trial = Mock()
        mock_trial.number = 1
        
        mock_config = Mock()
        mock_config.energy_fn_name = "mse"
        mock_config.n_embed = 256
        mock_config.block_size = 128
        mock_config.num_heads = 8
        mock_config.n_blocks = 4
        mock_config.T = 10
        mock_config.peak_learning_rate = 1e-4
        mock_config.warmup_steps = 200
        mock_config.dropout = 0.1
        mock_config.update_bias = True
        
        log_trial_to_detailed_log(
            trials_path, mock_trial, mock_config,
            trial_time=120.5, val_loss=0.5, raw_energy=1.2,
            norm_energy=0.8, combined_energy=1.3
        )
        
        # Check that detailed data was written
        with open(trials_path, 'r') as f:
            content = f.read()
            assert "TRIAL 1" in content
            assert "120.5s" in content
            assert "0.5" in content  # val_loss
            assert "1.2" in content  # raw_energy
            assert "0.8" in content  # norm_energy
            assert "1.3" in content  # combined_energy
            assert "mse" in content  # energy_fn_name
            assert "256x128" in content  # n_embed x block_size
            assert "heads=8" in content
            assert "blocks=4" in content
            assert "T=10" in content

def test_write_final_results():
    """
    Test that write_final_results writes final results in correct format.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        results_path = os.path.join(temp_dir, "results.txt")
        
        # Create mock trial
        mock_trial = Mock()
        mock_trial.value = 1.25
        mock_trial.user_attrs = {
            "config": {
                "n_embed": 256,
                "block_size": 128,
                "num_heads": 8,
                "dropout": 0.1
            },
            "ce_loss": 0.5,
            "energy": 1.2,
            "normalized_energy": 0.8,
            "combined_energy": 1.3
        }
        
        write_final_results(results_path, mock_trial)
        
        # Check that results were written
        with open(results_path, 'r') as f:
            content = f.read()
            assert "COMBINED ENERGY OPTIMIZATION RESULTS" in content
            assert "1.2500" in content  # Best combined energy
            assert "0.5000" in content  # CE Loss
            assert "1.2000" in content  # Raw Energy
            assert "0.8000" in content  # Normalized Energy
            assert "1.3000" in content  # Combined Energy
            assert "n_embed: 256" in content
            assert "block_size: 128" in content
            assert "num_heads: 8" in content
            assert "dropout: 0.1" in content

# ===================== Monitor Tests (monitor_tuning.py) =====================

@patch('optuna.load_study')
def test_monitor_study_success(mock_load_study):
    """
    Test that monitor_study successfully loads and displays study information.
    """
    # Create mock study
    mock_study = Mock()
    mock_study.direction = "minimize"
    mock_study.trials = [Mock(), Mock(), Mock()]  # 3 trials
    mock_study.best_trial = Mock()
    mock_study.best_trial.value = 1.25
    mock_study.best_trial.params = {"n_embed": 256, "dropout": 0.1}
    mock_study.trials[-1].state.name = "COMPLETE"
    mock_study.trials[-1].value = 1.3
    mock_study.trials[-1].number = 2
    
    mock_load_study.return_value = mock_study
    
    # Mock Path.exists to return True
    with patch('pathlib.Path.exists', return_value=True):
        monitor_study("test_study")
    
    # Verify study was loaded
    mock_load_study.assert_called_once()

@patch('optuna.load_study')
def test_monitor_study_no_trials(mock_load_study):
    """
    Test that monitor_study handles studies with no trials.
    """
    # Create mock study with no trials
    mock_study = Mock()
    mock_study.direction = "minimize"
    mock_study.trials = []
    mock_study.best_trial = None
    
    mock_load_study.return_value = mock_study
    
    # Mock Path.exists to return True
    with patch('pathlib.Path.exists', return_value=True):
        monitor_study("test_study")
    
    # Verify study was loaded
    mock_load_study.assert_called_once()

@patch('pathlib.Path.exists')
def test_monitor_study_not_found(mock_exists):
    """
    Test that monitor_study handles non-existent studies.
    """
    mock_exists.return_value = False
    
    # Should not raise an exception
    monitor_study("nonexistent_study")

# ===================== Integration Tests =====================

def test_config_integration():
    """
    Test integration between config functions.
    """
    mock_trial = Mock()
    mock_trial.suggest_int.side_effect = [256, 0, 128, 2, 8, 200, 0, 1]
    mock_trial.suggest_float.side_effect = [0.1, 1e-4]
    
    # Test full config creation and update cycle
    config = get_dynamic_model_config(mock_trial, vocab_size=100, flash=False)
    assert config is not None
    
    # Test energy normalization
    raw_energy = 1.0
    normalized = normalize_energy(raw_energy, config.energy_fn_name)
    assert normalized != raw_energy  # Should be normalized
    
    # Test config update
    config_dict = config.__dict__
    update_global_config(config_dict)
    
    # Verify some key attributes were updated
    assert hasattr(GPTConfig, 'n_embed')
    assert hasattr(GPTConfig, 'block_size') 
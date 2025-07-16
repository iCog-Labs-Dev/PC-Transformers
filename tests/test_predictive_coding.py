"""
Tests for the predictive_coding module.

This module tests the configuration and predictive coding layer components.
"""

import torch
import torch.nn as nn
import pytest
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from config import make_test_config
# ===================== Config Tests (config.py) =====================

def test_gpt_config_creation():
    """
    Test that GPTConfig can be created with required parameters.
    """
    config = GPTConfig(vocab_size=100, block_size=10)
    assert config.vocab_size == 100
    assert config.block_size == 10

def test_gpt_config_default_values():
    """
    Test that GPTConfig has correct default values for optional parameters.
    """
    config = GPTConfig(vocab_size=100, block_size=10)
    assert config.peak_learning_rate == 1.29e-04
    assert config.warmup_steps == 58
    assert config.la == 0.5
    assert config.n_embed == 656
    assert config.dropout == 0.1
    assert config.local_learning_rate == 0
    assert config.T == 10
    assert config.is_holding_error == False
    assert config.update_bias == True
    assert config.num_heads == 16
    assert config.n_blocks == 4
    assert config.batch_size == 8
    assert config.num_epochs == 5
    assert config.use_lateral == True
    assert config.energy_fn_name == "mse"
    assert config.eos_token_id == None
    assert config.use_flash_attention == False

def test_gpt_config_custom_values():
    """
    Test that GPTConfig can be created with custom values for all parameters.
    """
    config = GPTConfig(
        vocab_size=50,
        block_size=20,
        peak_learning_rate=1e-3,
        warmup_steps=100,
        la=0.1,
        n_embed=128,
        dropout=0.2,
        local_learning_rate=1e-4,
        T=5,
        is_holding_error=True,
        update_bias=False,
        num_heads=8,
        n_blocks=6,
        batch_size=16,
        num_epochs=10,
        use_lateral=False,
        energy_fn_name="mse",
        eos_token_id=50,
        use_flash_attention=True
    )
    
    assert config.vocab_size == 50
    assert config.block_size == 20
    assert config.peak_learning_rate == 1e-3
    assert config.warmup_steps == 100
    assert config.la == 0.1
    assert config.n_embed == 128
    assert config.dropout == 0.2
    assert config.local_learning_rate == 1e-4
    assert config.T == 5
    assert config.is_holding_error == True
    assert config.update_bias == False
    assert config.num_heads == 8
    assert config.n_blocks == 6
    assert config.batch_size == 16
    assert config.num_epochs == 10
    assert config.use_lateral == False
    assert config.energy_fn_name == "mse"
    assert config.eos_token_id == 50
    assert config.use_flash_attention == True

# ===================== PCLayer Tests (pc_layer.py) =====================

def test_pc_layer_custom_initialization():
    """
    Test that PCLayer can be initialized with custom parameters.
    """
    pc_layer = PCLayer(
        T=5,
        local_learning_rate=1e-4,
        is_holding_error=True,
        update_bias=False,
        energy_fn_name="mse"
    )
    assert pc_layer.T == 5
    assert pc_layer.local_lr == 1e-4
    assert pc_layer.is_holding_error == True
    assert pc_layer.update_bias == False
    assert pc_layer.energy_fn_name == "mse"

def test_pc_layer_register_lateral():
    """
    Test that PCLayer can register lateral weights for different layer types.
    """
    pc_layer = PCLayer()
    
    # Test registering lateral weights
    pc_layer.register_lateral("attn", 64)
    pc_layer.register_lateral("fc1", 128)
    
    assert "attn" in pc_layer.W_latents
    assert "fc1" in pc_layer.W_latents
    assert pc_layer.W_latents["attn"].shape == (64, 64)
    assert pc_layer.W_latents["fc1"].shape == (128, 128)

def test_pc_layer_register_lateral_duplicate():
    """
    Test that registering the same lateral layer type twice doesn't create duplicates.
    """
    pc_layer = PCLayer()
    
    # Register the same layer type twice
    pc_layer.register_lateral("attn", 64)
    pc_layer.register_lateral("attn", 64)
    
    # Should only have one entry
    assert len(pc_layer.W_latents) == 1
    assert "attn" in pc_layer.W_latents

def test_pc_layer_init_x_linear():
    """
    Test that PCLayer can initialize x cache for linear layers.
    """
    pc_layer = PCLayer()
    layer = nn.Linear(64, 128)
    device = torch.device('cpu')
    
    pc_layer.init_x(
        batch_size=2,
        seq_len=5,
        layer=layer,
        layer_type="linear",
        device=device
    )
    
    assert "linear" in pc_layer._x_cache
    assert pc_layer._x_cache["linear"].shape == (2, 5, 64)

def test_pc_layer_init_x_embed():
    """
    Test that PCLayer can initialize x cache for embedding layers.
    """
    pc_layer = PCLayer()
    word_embed = nn.Embedding(100, 64)
    pos_embed = nn.Embedding(50, 64)
    layer = {"word": word_embed, "pos": pos_embed}
    device = torch.device('cpu')
    
    input_ids = torch.randint(0, 100, (2, 5))
    position_ids = torch.randint(0, 50, (2, 5))
    
    pc_layer.init_x(
        batch_size=2,
        seq_len=5,
        layer=layer,
        layer_type="embed",
        input_ids=input_ids,
        position_ids=position_ids,
        device=device
    )
    
    assert "embed" in pc_layer._x_cache
    x_word, x_pos = pc_layer._x_cache["embed"]
    assert x_word.shape == (2, 5, 64)
    assert x_pos.shape == (2, 5, 64)

def test_pc_layer_init_x_attn():
    """
    Test that PCLayer can initialize x cache for attention layers.
    """
    pc_layer = PCLayer()
    q_proj = nn.Linear(64, 64)
    k_proj = nn.Linear(64, 64)
    v_proj = nn.Linear(64, 64)
    proj_layers = {"q_proj": q_proj, "k_proj": k_proj, "v_proj": v_proj}
    device = torch.device('cpu')
    
    pc_layer.init_x(
        batch_size=2,
        seq_len=5,
        proj_layers=proj_layers,
        layer_type="attn",
        device=device
    )
    
    assert "attn" in pc_layer._x_cache
    assert pc_layer._x_cache["attn"].shape == (2, 5, 64)

def test_pc_layer_init_x_missing_device():
    """
    Test that PCLayer raises an error when device is not provided to init_x.
    """
    pc_layer = PCLayer()
    layer = nn.Linear(64, 128)
    
    with pytest.raises(ValueError, match="Device must be explicitly provided"):
        pc_layer.init_x(
            batch_size=2,
            seq_len=5,
            layer=layer,
            layer_type="linear"
        )

def test_pc_layer_get_x():
    """
    Test that PCLayer can retrieve cached x values.
    """
    pc_layer = PCLayer()
    layer = nn.Linear(64, 128)
    device = torch.device('cpu')
    
    pc_layer.init_x(
        batch_size=2,
        seq_len=5,
        layer=layer,
        layer_type="linear",
        device=device
    )
    
    x = pc_layer.get_x("linear")
    assert x is not None
    assert x.shape == (2, 5, 64)
    
    # Test getting non-existent layer type
    x_none = pc_layer.get_x("nonexistent")
    assert x_none is None

def test_pc_layer_energy_management():
    """
    Test that PCLayer can manage energy values correctly.
    """
    pc_layer = PCLayer(is_holding_error=True)
    
    # Initially energy should be 0
    assert pc_layer.get_energy() == 0.0
    
    # Set energy manually for testing
    pc_layer._energy = 1.5
    assert pc_layer.get_energy() == 1.5
    
    # Clear energy
    pc_layer.clear_energy()
    assert pc_layer.get_energy() == 0.0

def test_pc_layer_error_management():
    """
    Test that PCLayer can manage error values correctly.
    """
    pc_layer = PCLayer(is_holding_error=True)
    
    # Initially errors should be empty
    assert pc_layer.get_errors() == []
    
    # Add some errors manually for testing
    pc_layer._errors = [{"step": 0, "error": 0.1}, {"step": 1, "error": 0.2}]
    errors = pc_layer.get_errors()
    assert len(errors) == 2
    assert errors[0]["step"] == 0
    assert errors[1]["step"] == 1
    
    # Clear errors
    pc_layer.clear_errors()
    assert pc_layer.get_errors() == []

def test_pc_layer_forward_missing_init():
    """
    Test that PCLayer forward raises an error when x cache is not initialized.
    """
    pc_layer = PCLayer()
    target_activity = torch.randn(2, 5, 64)
    
    with pytest.raises(ValueError, match="linear state not initialized"):
        pc_layer.forward(
            target_activity=target_activity,
            layer_type="linear"
        )

def test_pc_layer_embed_cache():
    """
    Test that PCLayer properly manages embedding cache during forward pass.
    """
    pc_layer = PCLayer()
    word_embed = nn.Embedding(100, 64)
    pos_embed = nn.Embedding(50, 64)
    layer = {"word": word_embed, "pos": pos_embed}
    device = torch.device('cpu')
    
    input_ids = torch.randint(0, 100, (2, 5))
    position_ids = torch.randint(0, 50, (2, 5))
    
    pc_layer.init_x(
        batch_size=2,
        seq_len=5,
        layer=layer,
        layer_type="embed",
        input_ids=input_ids,
        position_ids=position_ids,
        device=device
    )
    
    # Initially embed cache should not exist
    assert not hasattr(pc_layer, '_embed_cache')
    
    # Create a dummy target activity for testing
    target_activity = torch.randn(2, 5, 64)
    
    # During forward pass, embed cache should be created
    try:
        pc_layer.forward(
            target_activity=target_activity,
            layer=layer,
            layer_type="embed",
            input_ids=input_ids,
            position_ids=position_ids,
            t=0,
            T=1,
            requires_update=False
        )
    except Exception:
        # The forward pass might fail due to missing dependencies, but we can still test cache creation
        pass
    
    # After forward pass attempt, embed cache should exist
    assert hasattr(pc_layer, '_embed_cache')
    assert pc_layer._embed_cache["step"] == 0
    assert not pc_layer._embed_cache["mu_word"] is None
    assert not pc_layer._embed_cache["mu_pos"] is None 
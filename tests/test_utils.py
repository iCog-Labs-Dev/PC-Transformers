from config import make_test_config
from model_architecture.pc_t_model import PCTransformer
import torch

def test_device_utils():
    """
    Test that device utilities can be imported and used.
    """
    from utils.device_utils import setup_device, cleanup_memory
    
    # Test device setup
    local_rank, device, ddp = setup_device()
    assert isinstance(local_rank, int)
    assert isinstance(device, torch.device)
    assert isinstance(ddp, bool)
    
    # Test memory cleanup (should not raise an error)
    cleanup_memory()

def test_model_utils():
    """
    Test that model utilities can be imported and used.
    """
    from utils.model_utils import load_model, reset_pc_modules
    
    # Test model loading (with a dummy path, should fail gracefully)
    config = make_test_config(n_blocks=1)
    model = PCTransformer(config)
    
    # Test reset_pc_modules (should not raise an error)
    reset_pc_modules(model)
    
    # Check that model still has expected components
    assert hasattr(model, 'embedding')
    assert hasattr(model, 'blocks')
    assert hasattr(model, 'output')

def test_pc_utils():
    """
    Test that predictive coding utilities can be imported and used.
    """
    from utils.pc_utils import ids_to_one_hot
    
    # Test one-hot encoding
    vocab_size = 20
    batch_size = 2
    seq_len = 5
    ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    one_hot = ids_to_one_hot(ids, vocab_size)
    assert one_hot.shape == (batch_size, seq_len, vocab_size)
    assert one_hot.dtype == torch.float32

def test_attention_utils():
    """
    Test that attention utilities can be imported and used.
    """
    from utils.attention_utils import apply_standard_attention
    
    batch_size = 2
    seq_len = 5
    num_heads = 2
    head_dim = 4
    
    # Create dummy Q, K, V tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Test attention computation
    output = apply_standard_attention(q, k, v)
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)

# ===================== Integration Tests =====================

def test_pc_transformer_with_utils():
    """
    Test that PCTransformer works with utility functions.
    """
    from utils.pc_utils import ids_to_one_hot
    from utils.device_utils import setup_device
    from utils.model_utils import reset_pc_modules
    
    config = make_test_config(n_blocks=1, T=2)
    model = PCTransformer(config)
    local_rank, device, ddp = setup_device()
    
    # Create test data
    batch_size = 2
    seq_len = 5
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Convert to one-hot for testing
    target_one_hot = ids_to_one_hot(target_ids, vocab_size)
    
    # Test reset_pc_modules
    reset_pc_modules(model)
    
    assert target_one_hot.shape == (batch_size, seq_len, vocab_size)
    assert input_ids.shape == (batch_size, seq_len)
    assert target_ids.shape == (batch_size, seq_len)

def test_pc_transformer_different_configs():
    """
    Test PCTransformer with different configuration parameters.
    """
    configs = [
        make_test_config(n_blocks=1, T=2, num_heads=2),
        make_test_config(n_blocks=2, T=3, num_heads=4),
        make_test_config(n_blocks=1, T=1, num_heads=2, n_embed=16),
    ]
    
    for config in configs:
        model = PCTransformer(config)
        assert len(model.blocks) == config.n_blocks
        assert model.config.T == config.T
        assert model.config.num_heads == config.num_heads
        assert model.config.n_embed == config.n_embed 
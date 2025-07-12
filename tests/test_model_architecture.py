from config import make_test_config
# ===================== Embedding Layer (embedding.py) =====================
import torch
from model_architecture.embedding import Embedding_Layer

def test_embedding_layer_output_shape():
    """
    Test that the Embedding_Layer produces outputs of the correct shape.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len)
    embedding_layer = Embedding_Layer(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    out = embedding_layer.word_embeddings(input_ids) + embedding_layer.position_embeddings(position_ids)
    out = embedding_layer.LayerNorm(out)
    out = embedding_layer.dropout(out)
    assert out.shape == (batch_size, seq_len, config.n_embed)

def test_embedding_layer_layernorm_changes_output():
    """
    Test that LayerNorm changes the output of the Embedding_Layer.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len, dropout=0.0)
    embedding_layer = Embedding_Layer(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    out = embedding_layer.word_embeddings(input_ids) + embedding_layer.position_embeddings(position_ids)
    out_norm = embedding_layer.LayerNorm(out)
    assert not torch.allclose(out, out_norm), "LayerNorm should change the output."

def test_embedding_layer_dropout_changes_output():
    """
    Test that Dropout changes the output of the Embedding_Layer in training and eval modes.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len, dropout=0.5)
    embedding_layer = Embedding_Layer(config)
    embedding_layer.train()
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    out = embedding_layer.word_embeddings(input_ids) + embedding_layer.position_embeddings(position_ids)
    out = embedding_layer.LayerNorm(out)
    out1 = embedding_layer.dropout(out)
    out2 = embedding_layer.dropout(out)
    assert not torch.allclose(out1, out2), "Dropout should change the output in training mode."
    embedding_layer.eval()
    out3 = embedding_layer.dropout(out)
    out4 = embedding_layer.dropout(out)
    assert torch.allclose(out3, out4), "Dropout should not change the output in eval mode."

# ===================== Attention Module (attention.py) =====================
import math
from model_architecture.attention import Attention

def test_attention_output_shape():
    """
    Test that the Attention module produces outputs of the correct shape.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len)
    attn = Attention(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    q = attn.q(x)
    k = attn.k(x)
    v = attn.v(x)
    q = q.view(batch_size, seq_len, config.num_heads, config.n_embed // config.num_heads).transpose(1, 2)
    k = k.view(batch_size, seq_len, config.num_heads, config.n_embed // config.num_heads).transpose(1, 2)
    v = v.view(batch_size, seq_len, config.num_heads, config.n_embed // config.num_heads).transpose(1, 2)
    from utils.attention_utils import apply_standard_attention
    attn_out = apply_standard_attention(q, k, v)
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, config.n_embed)
    assert attn_out.shape == (batch_size, seq_len, config.n_embed)

def test_attention_various_head_counts():
    """
    Test that the Attention module can handle different head counts and produce correct outputs.
    """
    batch_size = 2
    seq_len = 5
    for n_embed, num_heads in [(8, 2), (12, 3), (16, 4)]:
        config = make_test_config(n_embed=n_embed, num_heads=num_heads, block_size=seq_len)
        attn = Attention(config)
        x = torch.randn(batch_size, seq_len, n_embed)
        q = attn.q(x)
        k = attn.k(x)
        v = attn.v(x)
        q = q.view(batch_size, seq_len, num_heads, n_embed // num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, n_embed // num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, n_embed // num_heads).transpose(1, 2)
        from utils.attention_utils import apply_standard_attention
        attn_out = apply_standard_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embed)
        assert attn_out.shape == (batch_size, seq_len, n_embed)

def test_attention_mask_application():
    """
    Test that the Attention module correctly applies a mask to the output.
    """
    batch_size = 1
    seq_len = 4
    config = make_test_config(block_size=seq_len)
    attn = Attention(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    q = attn.q(x)
    k = attn.k(x)
    v = attn.v(x)
    q = q.view(batch_size, seq_len, config.num_heads, config.n_embed // config.num_heads).transpose(1, 2)
    k = k.view(batch_size, seq_len, config.num_heads, config.n_embed // config.num_heads).transpose(1, 2)
    v = v.view(batch_size, seq_len, config.num_heads, config.n_embed // config.num_heads).transpose(1, 2)
    from utils.attention_utils import apply_standard_attention
    mask = torch.ones(batch_size, config.num_heads, seq_len, seq_len)
    mask[:, :, :, -1] = 0
    attn_out_masked = apply_standard_attention(q, k, v, mask=mask)
    attn_out_unmasked = apply_standard_attention(q, k, v, mask=None)
    assert not torch.allclose(attn_out_masked[..., -1, :], attn_out_unmasked[..., -1, :]), "Mask should affect the output for masked positions."

# ===================== MLP Block (mlp.py) =====================
import torch.nn as nn
from model_architecture.mlp import MLP

def test_mlp_output_shape():
    """
    Test that the MLP module produces outputs of the correct shape.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len)
    mlp = MLP(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    out1 = mlp.fc1(x)
    out2 = mlp.fc2(out1)
    assert out2.shape == (batch_size, seq_len, config.n_embed)

def test_mlp_various_input_sizes():
    """
    Test that the MLP module can handle different input sizes.
    """
    config = make_test_config()
    mlp = MLP(config)
    for batch_size, seq_len in [(1, 3), (4, 2), (2, 7)]:
        x = torch.randn(batch_size, seq_len, config.n_embed)
        out1 = mlp.fc1(x)
        out2 = mlp.fc2(out1)
        assert out2.shape == (batch_size, seq_len, config.n_embed)

# ===================== Output Layer (output.py) =====================
from model_architecture.output import OutputLayer

def test_output_layer_logits_shape():
    """
    Test that the OutputLayer produces outputs of the correct shape.
    """
    batch_size = 2
    seq_len = 5
    vocab_size = 20
    config = make_test_config(block_size=seq_len, vocab_size=vocab_size)
    output_layer = OutputLayer(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    logits = output_layer.output(x)
    assert logits.shape == (batch_size, seq_len, vocab_size) 

# ===================== Transformer Block (transformer_block.py) =====================
from model_architecture.transformer_block import TransformerBlock

def test_transformer_block_components():
    """
    Test that the TransformerBlock has all required submodules (ln1, attn, ln2, mlp)
    and that its layer norms produce outputs of the correct shape.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len)
    block = TransformerBlock(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    
    # Test that components exist and have correct shapes
    assert hasattr(block, 'ln1')
    assert hasattr(block, 'attn')
    assert hasattr(block, 'ln2')
    assert hasattr(block, 'mlp')
    
    # Test layer norms output shape
    ln1_out = block.ln1(x)
    ln2_out = block.ln2(x)
    assert ln1_out.shape == (batch_size, seq_len, config.n_embed)
    assert ln2_out.shape == (batch_size, seq_len, config.n_embed)

def test_transformer_block_layer_norms_different():
    """
    Test that the two LayerNorms in TransformerBlock produce outputs that are different from the input.
    Note: Since both LayerNorms are initialized identically, their outputs may be similar, but should not be identical to the input.
    """
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len)
    block = TransformerBlock(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    
    # Test that layer norms change the output
    ln1_out = block.ln1(x)
    ln2_out = block.ln2(x)
    assert not torch.allclose(x, ln1_out), "LayerNorm1 should change the output"
    assert not torch.allclose(x, ln2_out), "LayerNorm2 should change the output"
    # Check output shapes
    assert ln1_out.shape == (batch_size, seq_len, config.n_embed)
    assert ln2_out.shape == (batch_size, seq_len, config.n_embed)

# ===================== PC Transformer Model (pc_t_model.py) =====================
from model_architecture.pc_t_model import PCTransformer

def test_pc_transformer_initialization():
    """
    Test that the PCTransformer model initializes with the correct number of blocks and required components.
    """
    config = make_test_config(n_blocks=2)
    model = PCTransformer(config)
    assert hasattr(model, 'embedding')
    assert hasattr(model, 'blocks')
    assert hasattr(model, 'output')
    assert len(model.blocks) == config.n_blocks

def test_pc_transformer_components():
    """
    Test that the PCTransformer and its blocks have all required submodules.
    """
    batch_size = 2
    seq_len = 5
    vocab_size = 20
    config = make_test_config(block_size=seq_len, vocab_size=vocab_size, n_blocks=1)
    model = PCTransformer(config)
    # Test that model components exist and have correct shapes
    assert hasattr(model, 'embedding')
    assert hasattr(model, 'blocks')
    assert hasattr(model, 'output')
    assert len(model.blocks) == config.n_blocks
    # Test that blocks have expected components
    for block in model.blocks:
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'attn')
        assert hasattr(block, 'ln2')
        assert hasattr(block, 'mlp')

def test_pc_transformer_input_validation():
    """
    Test that input and target IDs for PCTransformer have the correct shape and dimensions.
    """
    batch_size = 2
    seq_len = 5
    vocab_size = 20
    config = make_test_config(block_size=seq_len, vocab_size=vocab_size, n_blocks=1)
    model = PCTransformer(config)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Test that input validation works
    assert input_ids.shape == (batch_size, seq_len)
    assert target_ids.shape == (batch_size, seq_len)
    assert input_ids.ndim == 2, "Expected input_ids shape [B, S]"

def test_pc_transformer_config_attributes():
    """
    Test that the PCTransformer model exposes its configuration attributes correctly.
    """
    config = make_test_config(n_blocks=2)
    model = PCTransformer(config)
    assert hasattr(model, 'config')
    assert model.config.n_blocks == config.n_blocks
    assert model.config.vocab_size == config.vocab_size
    assert model.config.n_embed == config.n_embed 

def test_pc_transformer_train_mode():
    """
    Test that PCTransformer can be set to training mode and maintains its components.
    """
    config = make_test_config(n_blocks=1, T=2)
    model = PCTransformer(config)
    model.train()
    
    assert model.training == True
    assert hasattr(model, 'embedding')
    assert hasattr(model, 'blocks')
    assert hasattr(model, 'output')

def test_pc_transformer_eval_mode():
    """
    Test that PCTransformer can be set to evaluation mode and maintains its components.
    """
    config = make_test_config(n_blocks=1, T=2)
    model = PCTransformer(config)
    model.eval()
    
    assert model.training == False
    assert hasattr(model, 'embedding')
    assert hasattr(model, 'blocks')
    assert hasattr(model, 'output')

def test_pc_transformer_register_lateral_weights():
    """
    Test that the register_all_lateral_weights method can be called without errors.
    """
    config = make_test_config(n_blocks=1, T=2)
    model = PCTransformer(config)
    
    # This should not raise an error
    model.register_all_lateral_weights()
    
    # Check that lateral weights are registered
    for block in model.blocks:
        assert hasattr(block.attn.pc_qkv, 'W_latents')
        assert hasattr(block.attn.pc_output, 'W_latents')
        assert hasattr(block.mlp.pc_layer1, 'W_latents')
        assert hasattr(block.mlp.pc_layer2, 'W_latents')
    
    assert hasattr(model.output.pc_layer, 'W_latents')


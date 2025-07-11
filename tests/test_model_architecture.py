# ===================== Test Config Helper =====================
def make_test_config(**overrides):
    base = dict(
        vocab_size=20,
        block_size=5,
        n_embed=8,
        dropout=0.1,
        local_learning_rate=1e-3,
        T=2,
        is_holding_error=True,
        num_heads=2,
        n_blocks=1,
        num_epochs=1,
        update_bias=False,
        energy_fn_name="scaled_mse",
        eos_token_id=19
    )
    base.update(overrides)
    from predictive_coding.config import GPTConfig
    return GPTConfig(**base)

# ===================== Embedding Layer (embedding.py) =====================
import torch
from model_architecture.embedding import Embedding_Layer

def test_embedding_layer_output_shape():
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
    batch_size = 2
    seq_len = 5
    config = make_test_config(block_size=seq_len)
    mlp = MLP(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    out1 = mlp.fc1(x)
    out2 = mlp.fc2(out1)
    assert out2.shape == (batch_size, seq_len, config.n_embed)

def test_mlp_various_input_sizes():
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
    batch_size = 2
    seq_len = 5
    vocab_size = 20
    config = make_test_config(block_size=seq_len, vocab_size=vocab_size)
    output_layer = OutputLayer(config)
    x = torch.randn(batch_size, seq_len, config.n_embed)
    logits = output_layer.output(x)
    assert logits.shape == (batch_size, seq_len, vocab_size) 
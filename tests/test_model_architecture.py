import torch
from model_architecture.embedding import Embedding_Layer
from predictive_coding.config import GPTConfig

def test_embedding_layer_output_shape():
    """
    Test that Embedding_Layer returns the correct output shape for given input and position IDs.
    The output should have shape (batch_size, seq_len, n_embed).
    """
    batch_size = 2
    seq_len = 5
    n_embed = 8
    config = GPTConfig(
        vocab_size=20,
        block_size=seq_len,
        n_embed=n_embed,
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
    embedding_layer = Embedding_Layer(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    # Forward pass: word + position embeddings, layer norm, dropout, pc_layer
    out = embedding_layer.word_embeddings(input_ids) + embedding_layer.position_embeddings(position_ids)
    out = embedding_layer.LayerNorm(out)
    out = embedding_layer.dropout(out)
    # Optionally pass through pc_layer if needed
    assert out.shape == (batch_size, seq_len, n_embed) 
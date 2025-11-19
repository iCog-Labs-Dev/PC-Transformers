import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Embedding_Layer(nn.Module):
    """
    Embedding layer with word and positional embeddings, layer normalization, dropout, and a predictive coding layer.
    """
    def __init__(self, config):
        super(Embedding_Layer, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.rms_norm = nn.RMSNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.pc_layer= PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,                    
        )
import torch 
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Embedding_Layer(nn.Module):
    def __init__(self, config):
        super(Embedding_Layer, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.LayerNorm = nn.LayerNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.pc_layer= PCLayer(T=config.T,
                               local_learning_rate=config.local_learning_rate,
                               is_holding_error= config.is_holding_error,
                               update_bias = config.update_bias,
                               energy_fn_name=config.energy_fn_name,
                               
                               )

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
                               )

    
    def evaluate(self, input_ids, position_ids=None):
        word_embed = self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                word_embed.size(1), device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)
        pos_embed=self.position_embeddings(position_ids)
        embeddings = word_embed + pos_embed
        embeddings = self.LayerNorm(embeddings)

        return embeddings

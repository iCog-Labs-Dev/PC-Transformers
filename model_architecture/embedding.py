import torch 
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Embedding_Layer(nn.Module):
    """
    Embedding layer with word and positional embeddings, layer normalization, dropout, and a predictive coding layer.
    """
    def __init__(self, config):
        """
        Initialize the Embedding_Layer.

        Args:
            config: Configuration object with vocab_size, n_embed, block_size, dropout, T, etc.
        """
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

    
    def evaluate(self, input_ids, position_ids=None):
        """
        Compute embeddings for input token and position IDs (inference mode).

        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T) with token IDs.
            position_ids (torch.Tensor, optional): Tensor of shape (B, T) with position IDs. If None, generated automatically.
        Returns:
            torch.Tensor: Embedded input of shape (B, T, n_embed).
        """
        word_embed = self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(word_embed.size(1)).unsqueeze(0).expand_as(input_ids)
        pos_embed=self.position_embeddings(position_ids)
        embeddings = word_embed + pos_embed
        embeddings = self.LayerNorm(embeddings)

        return embeddings

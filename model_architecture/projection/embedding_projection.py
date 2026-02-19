import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Embedding_LayerProjection(nn.Module):
    """
    Embedding layer with word and positional embeddings, layer normalization, dropout, and a predictive coding layer.
    """
    def __init__(self, config):
        super(Embedding_LayerProjection, self).__init__()
        self.word_embeddings_projection = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings_projection = nn.Embedding(config.block_size, config.n_embed)
        self.rms_norm = nn.RMSNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.pc_layer_projection= PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,                    
            optimizer_name=config.optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_sign_value=config.optimizer_sign_value,
            optimizer_weight_bound=config.optimizer_weight_bound,
        )
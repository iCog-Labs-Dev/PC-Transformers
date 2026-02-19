import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayerProjection(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_projection = nn.Linear(config.n_embed, config.vocab_size)
        
        self.pc_layer_projection = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.output_energy_fn_name,
            optimizer_name=config.optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_sign_value=config.optimizer_sign_value,
            optimizer_weight_bound=config.optimizer_weight_bound,
        )

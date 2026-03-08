import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayer(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        
        self.pc_layer = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.output_energy_fn_name,
            use_memory_init=getattr(config, "use_memory_init", True),
            memory_slots=getattr(config, "memory_slots", 128),
            memory_delta=getattr(config, "memory_delta", 8.0),
            memory_update_qk=getattr(config, "memory_update_qk", True),
        )

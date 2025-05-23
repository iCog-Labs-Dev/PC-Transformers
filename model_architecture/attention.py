import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Attention(nn.Module):
    def __init__(self, config):
         super().__init__()

         self.config = config
         self.num_heads = config.num_heads
         self.n_embed = config.n_embed
         self.head_dim = config.n_embed // config.num_heads
         self.dropout = nn.Dropout(config.dropout)

         self.q = nn.Linear(config.n_embed, config.n_embed)
         self.k = nn.Linear(config.n_embed, config.n_embed)
         self.v = nn.Linear(config.n_embed, config.n_embed)
         self.output = nn.Linear(config.n_embed, config.n_embed)
        
         self.pc_qkv=PCLayer(T= config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias
        )

         self.pc_output = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias
        )

    def forward(self, fc1_x):
        self.pc_output(target_activity = fc1_x, layer = self.output, layer_type = "output_attn")
        att_score = self.pc_output.get_x("output_attn")

        self.pc_qkv(target_activity = att_score, proj_layers={
            "q_proj": self.q,
            "k_proj": self.k,
            "v_proj": self.v
        }, layer_type = "attn")

        x_qkv = self.pc_qkv.get_x("attn")

        return x_qkv

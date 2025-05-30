import torch.nn as nn
import torch
import math
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
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )

         self.pc_output = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
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
    
    def evaluate(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Reshape for multi-head: [B, T, H, D/H] → [B, H, T, D/H]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score: [B, H, T, T]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  
        scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_probs = nn.Softmax(dim=-1)(scores)
        attention_probs = self.dropout(attention_probs)

        # Context vector: [B, H, T, D/H]
        context = torch.matmul(attention_probs, V)

        #Concatenate heads: [B, T, H, D/H] → [B, T, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embed)
        output = self.output(context)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from predictive_coding.pc_layer import PCLayer

import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
         super().__init__()

         #assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by number of heads"
         self.config = config

         self.n_head = config.n_head
         self.n_embd = config.n_embd
         self.head_dim = config.n_embd // config.n_head
         self.dropout = config.dropout

         self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
         self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
         self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
         self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layer
         self.attn_dropout = nn.Dropout(config.dropout)
         self.pc_qkv=PCLayer(T= config.T,local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error

        )

         self.pc_output = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error
        )

    def forward(self, fc1_x):
        self.pc_output.clear_energy()
        self.pc_output.clear_errors()

        attn_out=self.pc_output(target_activity = fc1_x, layer = self.output, kind = "attn_out")
        att_score = self.pc_output.get_x("attn_out")

        self.pc_qkv.clear_energy()
        self.pc_qkv.clear_errors()

        score=self.pc_qkv(target_activity = att_score, layers= {'q': self.query, 'k': self.key, 'v':self.value, 'num_heads': self.num_heads}, kind = "attn")
        x_qkv = self.pc_qkv.get_x("attn")

        return x_qkv
                             
         
         
         
         #self.resid_dropout = nn.Dropout(config.dropout)
         #self.register_buffer(
            #"causal_mask",
            #torch.tril(torch.ones(config.block_size, config.block_size))
             #       .view(1, 1, config.block_size, config.block_size))
             
    #def forward(self, x): 
       # B, T, C = x.size()
        
        #q = self.q(x)
       #q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        #k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        #v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        
        #attn_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        #attn_score = attn_score.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('inf'))
        #attn_weights = F.softmax(attn_score, dim=-1)
        #attn_weights = self.attn_dropout(attn_weights)
        #attn_out = attn_weights @ v

        # Merge heads back: [B, T, C]
        #attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        # Final projection
        #out = self.o(attn_out)
        #out = self.resid_dropout(out)
        #return out  

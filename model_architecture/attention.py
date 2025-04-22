import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
         super().__init__()

         assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by number of heads"

         self.n_head = config.n_head
         self.n_embd = config.n_embd
         self.head_dim = config.n_embd // config.n_head
         self.dropout = config.dropout

         self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
         self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
         self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
         self.o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layer
         self.attn_dropout = nn.Dropout(config.dropout)
         self.resid_dropout = nn.Dropout(config.dropout)

         # Flash attention
         self.flash = hasattr(F, 'scaled_dot_product_attention')

         if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for auto-regressive attention
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )



    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Reshae for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        
        if self.flash:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True)
        else:
            attn_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_score = attn_score.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('inf'))
            attn_weights = F.softmax(attn_score, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out = attn_weights @ v

        # Merge heads back: [B, T, C]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        # Final projection
        out = self.o(attn_out)
        out = self.resid_dropout(out)
        return out  

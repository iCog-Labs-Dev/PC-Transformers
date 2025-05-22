import torch 
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer


class Embedding_Layer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_pos_embed, pad_token_id, T=5, lr=0.01):
        super(Embedding_Layer, self).__init__()
        self.word_embeddings=nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.postion_embeddings=nn.Embedding(max_pos_embed, embedding_dim)
        self.LayerNorm= nn.LayerNorm(embedding_dim, elementwise_affine=True)
        self.dropout= nn.Dropout(0.1)
        
        self.register_buffer("position_ids", torch.arange(max_pos_embed).expand(1, -1))
       # self.pc_layer = PCLayer(T=T, lr=lr)

    def forward(self, input_ids):
        input_shape= input_ids.size()
        seq_len= input_shape[1]
        
        input_embeds = self.word_embeddings(input_ids)
        
        position_ids= self.position_ids[:, :seq_len]
        position_embeddings=self.position_embeddings(position_ids)
        
        
        embeddings = input_embeds + position_embeddings
        embeddings= self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
       # if self.training:
          #  embeddings=self.PCLayer(embeddings)
        return embeddings
        

import torch 
import torch.nn as nn
from typing import Optional
from predictive_coding.pc_layer import PCLayer


class Embedding_Layer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_pos_embed, pad_token_id, T=5, lr=0.01):
        super(Embedding_Layer, self).__init__()
        self.config = config

        self.word_embeddings=nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.postion_embeddings=nn.Embedding(max_pos_embed, embedding_dim)
        self.LayerNorm= nn.LayerNorm(embedding_dim, elementwise_affine=True)
        self.dropout= nn.Dropout(0.1)
        
        self.pc_layer= PCLayer(T=config.T,
                               local_learning_rate=config.local_learning_rate,
                               energy_fn= config.energy.fn,
                               x_init=config.x_init,
                               is_holding_error= config.is_holding_error,
                               LayerNorm_instance=self.LayerNorm,
                               dropout_instances=self.dropout
                               )
        
        #self.register_buffer("position_ids", torch.arange(max_pos_embed).expand(1, -1))
       # self.pc_layer = PCLayer(T=T, lr=lr)

    def forward(self, input_ids, x_qkv, position_ids:Optional[torch.tensor]=None)-> torch.Tensor:
        
        self.pc_layer.clear_energy()
        self.pclayer.clear_errors()
        
        if position_ids is None:
           position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
           position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings= self.pc_layer(x_qkv, input_ids, position_ids, layers={'w': self.word_embeddings, 'p': self.postion_embeddings}, kind="embed")
        
        return embeddings
        
        
        
        
        
        
        #input_shape= input_ids.size()
        #seq_len= input_shape[1]
        
        #input_embeds = self.word_embeddings(input_ids)
        
        #position_ids= self.position_ids[:, :seq_len]
        #position_embeddings=self.position_embeddings(position_ids)
        
        
        #embeddings = input_embeds + position_embeddings
        #embeddings= self.LayerNorm(embeddings)
        #embeddings=self.dropout(embeddings)
       # if self.training:
          #  embeddings=self.PCLayer(embeddings)
        #return embeddings
        

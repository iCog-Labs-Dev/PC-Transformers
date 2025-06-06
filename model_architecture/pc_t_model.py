import torch.nn as nn
import torch
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from .transformer_utils import ids_to_one_hot
from .output import OutputLayer
from predictive_coding.pc_utils import x_init


class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)

    def forward(self, target_ids, input_ids):
        batch_size, seq_len = input_ids.shape
        target_logits = ids_to_one_hot(target_ids, self.output.config.vocab_size)
        position_ids = (torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len))

        
        self.embedding.pc_layer.init_x(
            batch_size=batch_size,
            seq_len=seq_len,
            layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
            layer_type="embed",
            input_ids=input_ids,
            position_ids=position_ids
        )


        for block in self.blocks:

            block.attn.pc_qkv.init_x(
                batch_size=batch_size,
                seq_len=seq_len,
                proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                layer_type="attn"
            )

            block.attn.pc_output.init_x(
                batch_size=batch_size,
                seq_len=seq_len,
                layer=block.attn.output,
                layer_type="linear"
            )

            block.mlp.pc_layer1.init_x(
                batch_size=batch_size,
                seq_len=seq_len,
                layer=block.mlp.fc1,
                layer_type="fc1"
            )
            
            block.mlp.pc_layer2.init_x(
                batch_size=batch_size,
                seq_len=seq_len,
                layer=block.mlp.fc2,
                layer_type="linear"
            )

          
        self.output.pc_layer.init_x(
              batch_size=batch_size,
              seq_len=seq_len,
              layer=self.output.output,
              layer_type="linear"
          )

        for t in range(self.config.T):
            futures = []
            
            futures.append(torch.jit.fork(
                self.output.pc_layer.forward,
                target_activity=target_logits,
                layer=self.output.output,
                layer_type="linear",
                t=t,
                T=self.config.T,
            ))

            for block_idx, block in enumerate(self.blocks):
                futures.append(torch.jit.fork(
                    block.attn.pc_qkv.forward,
                    target_activity=block.attn.pc_output.get_x("linear"),
                    proj_layers={
                        "q_proj": block.attn.q,
                        "k_proj": block.attn.k,
                        "v_proj": block.attn.v,
                    },
                    layer_type="attn",
                    t=t,
                    T=self.config.T,
                ))
                futures.append(torch.jit.fork(
                    block.attn.pc_output.forward,
                    target_activity=block.mlp.pc_layer1.get_x("fc1"),
                    layer=block.attn.output,
                    layer_type="linear",
                    t=t,
                    T=self.config.T,
                ))
                futures.append(torch.jit.fork(
                    block.mlp.pc_layer1.forward,
                    target_activity=block.mlp.pc_layer2.get_x("linear"),
                    layer=block.mlp.fc1,
                    layer_type="fc1",
                    t=t,
                    T=self.config.T,
                ))
                target = (
                    self.blocks[block_idx + 1].attn.pc_qkv.get_x("attn")
                    if block_idx < len(self.blocks) - 1
                    else self.output.pc_layer.get_x("linear")
                )
                futures.append(torch.jit.fork(
                    block.mlp.pc_layer2.forward,
                    target_activity=target,
                    layer=block.mlp.fc2,
                    layer_type="linear",
                    t=t,
                    T=self.config.T,
                ))

            futures.append(torch.jit.fork(
                self.embedding.pc_layer.forward,
                target_activity=self.blocks[0].attn.pc_qkv.get_x("attn"),
                layer={
                    "word": self.embedding.word_embeddings,
                    "pos": self.embedding.position_embeddings,
                },
                layer_type="embed",
                input_ids=input_ids,
                position_ids=position_ids,
                t=t,
                T=self.config.T,
            ))
            

            for future in futures:
                torch.jit.wait(future)

        # Compute logits
        output_x = self.output.pc_layer.get_x("linear")
        logits = output_x @ self.output.output.weight.T + self.output.output.bias
        return logits

    def evaluate(self, input_ids):
        x = self.embedding.evaluate(input_ids, position_ids=None)
        for block in self.blocks:
            x = block.evaluate(x)

        return self.output.evaluate(x)

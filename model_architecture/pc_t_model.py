# import torch
# import torch.nn as nn
# from .embedding import Embedding_Layer
# from .transformer_block import TransformerBlock
# from utils.pc_utils import ids_to_one_hot
# from .output import OutputLayer


# class PCTransformer(nn.Module):
#     """
#     Top-down Predictive Coding Transformer model.

#     This model integrates predictive coding principles into a transformer architecture.
#     It consists of an embedding layer, multiple transformer blocks, and an output layer,
#     each equipped with predictive coding layers for iterative inference and local learning.
#     """

#     def __init__(self, config):
#         """
#         Initialize the PCTransformer model.

#         Args:
#             config: Configuration object containing model hyperparameters (e.g., n_blocks, vocab_size, T, etc.).
#         """
#         super().__init__()
#         self.config = config
#         self.embedding = Embedding_Layer(config)
#         self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
#         self.output = OutputLayer(config)

#     def register_all_lateral_weights(self):
#         """
#         Register lateral weights for all predictive coding layers in the model.
#         This enables lateral (recurrent) connections for local learning in each layer.
#         """
#         for block in self.blocks:
#             block.attn.pc_qkv.register_lateral("attn", block.attn.q.in_features)
#             block.attn.pc_output.register_lateral("linear", block.attn.output.in_features)
#             block.mlp.pc_layer1.register_lateral("fc1", block.mlp.fc1.in_features)
#             block.mlp.pc_layer2.register_lateral("linear", block.mlp.fc2.in_features)
#         self.output.pc_layer.register_lateral("linear", self.output.output.in_features)

#         for module in self.modules():
#             if hasattr(module, 'W_latents'):
#                 for key in module.W_latents:
#                     if module.W_latents[key] is not None:
#                         module.W_latents[key] = module.W_latents[key].to(next(self.parameters()).device)

#     def forward(self, target_ids, input_ids):
#         """
#         Forward pass of the PCTransformer model.

#         Args:
#             target_ids (torch.Tensor): Target token IDs of shape (B, T).
#             input_ids (torch.Tensor): Input token IDs of shape (B, T).

#         Returns:
#             logits (torch.Tensor): Tensor of shape (B, T, vocab_size), the model's output logits for each token position.
#         """
#         for module in self.modules():
#             if hasattr(module, "clear_energy"):
#                 module.clear_energy()
            
#             if hasattr(module, "clear_errors"):
#                 module.clear_errors()

#         B, S = input_ids.shape
#         device = input_ids.device
#         vocab_size = self.output.config.vocab_size
        
#         # Clip input_ids and target_ids to valid range before using them
#         if input_ids.max() >= vocab_size:
#             input_ids = torch.clamp(input_ids, max=vocab_size-1)
        
#         if target_ids.max() >= vocab_size:
#             target_ids = torch.clamp(target_ids, max=vocab_size-1)
        
#         target_logits = ids_to_one_hot(target_ids, vocab_size).to(device)
#         position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

#         self.embedding.pc_layer.init_x(
#             batch_size=B,
#             seq_len= S,
#             layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
#             layer_type="embed",
#             input_ids=input_ids,
#             position_ids=position_ids,
#             device=device
#         )

#         for block in self.blocks:
#             block.attn.pc_qkv.init_x(
#                 batch_size=B,
#                 seq_len=S,
#                 proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
#                 layer_type="attn",
#                 device=device
#             )
#             block.attn.pc_output.init_x(
#                 batch_size=B,
#                 seq_len=S,
#                 layer=block.attn.output,
#                 layer_type="linear",
#                 device=device
#             )
#             block.mlp.pc_layer1.init_x(
#                 batch_size=B,
#                 seq_len=S,
#                 layer=block.mlp.fc1,
#                 layer_type="fc1",
#                 device=device
#             )
#             block.mlp.pc_layer2.init_x(
#                 batch_size=B,
#                 seq_len=S,
#                 layer=block.mlp.fc2,
#                 layer_type="linear",
#                 device=device
#             )
#         self.output.pc_layer.init_x(
#             batch_size=B,
#             seq_len=S,
#             layer=self.output.output,
#             layer_type="linear",
#             device=device
#         )

#         for t in range(self.config.T):
#             futures = []
#             futures.append(torch.jit.fork(
#                 self.output.pc_layer.forward,
#                 target_activity=target_logits,
#                 layer=self.output.output,
#                 layer_type="linear",
#                 t=t,
#                 T=self.config.T,
#                 requires_update=self.training
#             ))
            
#             for idx in range(len(self.blocks) - 1, -1, -1):
#                 block = self.blocks[idx]
#                 next_target = (
#                     self.blocks[idx + 1].attn.pc_qkv.get_x("attn")
#                     if idx < len(self.blocks) - 1
#                     else self.output.pc_layer.get_x("linear")
#                 )
                
#                 layer_norm2 = block.ln2(next_target)
                
#                 futures.append(torch.jit.fork(
#                     block.mlp.pc_layer2.forward,
#                     target_activity=layer_norm2,
#                     layer=block.mlp.fc2,
#                     layer_type="linear",
#                     t=t,
#                     T=self.config.T,
#                     requires_update=self.training
#                 ))

#                 futures.append(torch.jit.fork(
#                     block.mlp.pc_layer1.forward,
#                     target_activity=block.mlp.pc_layer2.get_x("linear"),
#                     layer=block.mlp.fc1,
#                     layer_type="fc1",
#                     t=t,
#                     T=self.config.T,
#                     requires_update=self.training
#                 ))
                
#                 layer_norm1 = block.ln1(block.mlp.pc_layer1.get_x("fc1"))
                    
#                 futures.append(torch.jit.fork(
#                     block.attn.pc_output.forward,
#                     target_activity=layer_norm1,
#                     layer=block.attn.output,
#                     layer_type="linear",
#                     t=t,
#                     T=self.config.T,
#                     requires_update=self.training
#                 ))

#                 futures.append(torch.jit.fork(
#                     block.attn.pc_qkv.forward,
#                     target_activity=block.attn.pc_output.get_x("linear"),
#                     proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
#                     layer_type="attn",
#                     t=t,
#                     T=self.config.T,
#                     requires_update=self.training,
#                     flash= getattr(self.config, 'use_flash_attention', False)
#                 ))


#             futures.append(torch.jit.fork(
#                 self.embedding.pc_layer.forward,
#                 target_activity=self.blocks[0].attn.pc_qkv.get_x("attn"),
#                 layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
#                 layer_type="embed",
#                 input_ids=input_ids,
#                 position_ids=position_ids,
#                 t=t,
#                 T=self.config.T,
#                 requires_update=self.training
#             ))

#             # Wait for all concurrent inference steps to complete
#             for future in futures:
#                 try:
#                     torch.jit.wait(future)
#                 except Exception as e:
#                     print(f"Error in parallel inference step: {e}")

#         output_x = self.output.pc_layer.get_x("linear")
#         logits = output_x @ self.output.output.weight.T + self.output.output.bias
#         return logits
import torch
import torch.nn as nn
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from utils.pc_utils import ids_to_one_hot
from .output import OutputLayer

class PCTransformer(nn.Module):
    """
    Top-down Predictive Coding Transformer model.

    This model integrates predictive coding principles into a transformer architecture.
    It consists of an embedding layer, multiple transformer blocks, and an output layer,
    each equipped with predictive coding layers for iterative inference and local learning.
    """

    def __init__(self, config):
        """
        Initialize the PCTransformer model.

        Args:
            config: Configuration object containing model hyperparameters (e.g., n_blocks, vocab_size, T, etc.).
        """
        super().__init__()
        self.config = config
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)

    def register_all_lateral_weights(self):
        """
        Register lateral weights for all predictive coding layers in the model.
        This enables lateral (recurrent) connections for local learning in each layer.
        """
        for block in self.blocks:
            block.attn.pc_qkv.register_lateral("attn", block.attn.q.in_features)
            block.attn.pc_output.register_lateral("linear", block.attn.output.in_features)
            block.mlp.pc_layer1.register_lateral("fc1", block.mlp.fc1.in_features)
            block.mlp.pc_layer2.register_lateral("linear", block.mlp.fc2.in_features)
        self.output.pc_layer.register_lateral("linear", self.output.output.in_features)

        for module in self.modules():
            if hasattr(module, 'W_latents'):
                for key in module.W_latents:
                    if module.W_latents[key] is not None:
                        module.W_latents[key] = module.W_latents[key].to(next(self.parameters()).device)

    def forward(self, target_ids, input_ids):
        """
        Forward pass of the PCTransformer model using CUDA streams for GPU parallelism.

        Args:
            target_ids (torch.Tensor): Target token IDs of shape (B, T).
            input_ids (torch.Tensor): Input token IDs of shape (B, T).

        Returns:
            logits (torch.Tensor): Tensor of shape (B, T, vocab_size), the model's output logits for each token position.
        """
        for module in self.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            
            if hasattr(module, "clear_errors"):
                module.clear_errors()

        B, S = input_ids.shape
        device = input_ids.device
        vocab_size = self.output.config.vocab_size
        
        # Clip input_ids and target_ids to valid range before using them
        if input_ids.max() >= vocab_size:
            input_ids = torch.clamp(input_ids, max=vocab_size-1)
        
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)
        
        target_logits = ids_to_one_hot(target_ids, vocab_size).to(device)
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        self.embedding.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
            layer_type="embed",
            input_ids=input_ids,
            position_ids=position_ids,
            device=device
        )

        for block in self.blocks:
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                layer_type="attn",
                device=device
            )
            block.attn.pc_output.init_x(
                batch_size=B,
                seq_len=S,
                layer=block.attn.output,
                layer_type="linear",
                device=device
            )
            block.mlp.pc_layer1.init_x(
                batch_size=B,
                seq_len=S,
                layer=block.mlp.fc1,
                layer_type="fc1",
                device=device
            )
            block.mlp.pc_layer2.init_x(
                batch_size=B,
                seq_len=S,
                layer=block.mlp.fc2,
                layer_type="linear",
                device=device
            )
        self.output.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer=self.output.output,
            layer_type="linear",
            device=device
        )

        # Create CUDA streams for parallel execution
        streams = [torch.cuda.Stream(device=device) for _ in range(len(self.blocks) * 4 + 2)]

        for t in range(self.config.T):
            stream_idx = 0
            # Execute output layer in a separate stream
            with torch.cuda.stream(streams[stream_idx]):
                self.output.pc_layer.forward(
                    target_activity=target_logits,
                    layer=self.output.output,
                    layer_type="linear",
                    t=t,
                    T=self.config.T,
                    requires_update=self.training
                )
            stream_idx += 1

            # Iterate through blocks in reverse order for parallel execution
            for idx in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[idx]
                next_target = (
                    self.blocks[idx + 1].attn.pc_qkv.get_x("attn")
                    if idx < len(self.blocks) - 1
                    else self.output.pc_layer.get_x("linear")
                )
                
                layer_norm2 = block.ln2(next_target)
                
                # Execute MLP layer 2 in a stream
                with torch.cuda.stream(streams[stream_idx]):
                    block.mlp.pc_layer2.forward(
                        target_activity=layer_norm2,
                        layer=block.mlp.fc2,
                        layer_type="linear",
                        t=t,
                        T=self.config.T,
                        requires_update=self.training
                    )
                stream_idx += 1

                # Execute MLP layer 1 in a stream
                with torch.cuda.stream(streams[stream_idx]):
                    block.mlp.pc_layer1.forward(
                        target_activity=block.mlp.pc_layer2.get_x("linear"),
                        layer=block.mlp.fc1,
                        layer_type="fc1",
                        t=t,
                        T=self.config.T,
                        requires_update=self.training
                    )
                stream_idx += 1
                
                layer_norm1 = block.ln1(block.mlp.pc_layer1.get_x("fc1"))
                    
                # Execute attention output in a stream
                with torch.cuda.stream(streams[stream_idx]):
                    block.attn.pc_output.forward(
                        target_activity=layer_norm1,
                        layer=block.attn.output,
                        layer_type="linear",
                        t=t,
                        T=self.config.T,
                        requires_update=self.training
                    )
                stream_idx += 1

                # Execute attention QKV in a stream
                with torch.cuda.stream(streams[stream_idx]):
                    block.attn.pc_qkv.forward(
                        target_activity=block.attn.pc_output.get_x("linear"),
                        proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                        layer_type="attn",
                        t=t,
                        T=self.config.T,
                        requires_update=self.training,
                        flash=getattr(self.config, 'use_flash_attention', False)
                    )
                stream_idx += 1

            # Execute embedding layer in a stream
            with torch.cuda.stream(streams[stream_idx]):
                self.embedding.pc_layer.forward(
                    target_activity=self.blocks[0].attn.pc_qkv.get_x("attn"),
                    layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
                    layer_type="embed",
                    input_ids=input_ids,
                    position_ids=position_ids,
                    t=t,
                    T=self.config.T,
                    requires_update=self.training
                )
            stream_idx += 1

            # Synchronize all streams to ensure completion
            for stream in streams[:stream_idx]:
                stream.synchronize()

        output_x = self.output.pc_layer.get_x("linear")
        logits = output_x @ self.output.output.weight.T + self.output.output.bias
        return logits
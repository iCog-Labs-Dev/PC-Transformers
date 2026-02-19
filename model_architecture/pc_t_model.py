import torch
import torch.nn as nn
from .projection.embedding_projection import Embedding_LayerProjection
from .projection.transformer_block_projection import TransformerBlockProjection
from .projection.output_projecton import OutputLayerProjection

from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from utils.pc_utils import ids_to_one_hot
from .output import OutputLayer
from utils.device_utils import create_streams_or_futures, execute_parallel, synchronize_execution

class PCTransformer(nn.Module):
    """
    Top-down Predictive Coding Transformer model.

    This model integrates predictive coding principles into a transformer architecture.
    It consists of an embedding layer, multiple transformer blocks, and an output layer,
    each equipped with predictive coding layers for iterative inference and local learning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)
        self.embedding_projection = Embedding_LayerProjection(config)
        self.blocks_projection = nn.ModuleList([TransformerBlockProjection(config) for _ in range(config.n_blocks)])
        self.output_projection = OutputLayerProjection(config)

    def register_all_lateral_weights(self):
        """
        Register lateral weights for all predictive coding layers in the model.
        This enables lateral connections for local learning in each layer.
        """
        for block in self.blocks:
            block.attn.pc_qkv.register_lateral("attn", block.attn.q.in_features)
            block.attn.pc_output.register_lateral("linear_attn", block.attn.output.in_features)
            block.mlp.pc_layer1.register_lateral("fc1", block.mlp.fc1.in_features)
            block.mlp.pc_layer2.register_lateral("fc2", block.mlp.fc2.in_features)
        self.output.pc_layer.register_lateral("linear", self.output.output.in_features)
        for block_projection in self.blocks_projection:
            block_projection.attn.pc_qkv_projection.register_lateral("projection_attn", block_projection.attn.q_projection.in_features)
            block_projection.attn.pc_output_projection.register_lateral("projection_linear_attn", block_projection.attn.output_projection.in_features)
            block_projection.mlp.pc_layer1_projection.register_lateral("projection_fc1", block_projection.mlp.fc1_projection.in_features)
            block_projection.mlp.pc_layer2_projection.register_lateral("projection_fc2", block_projection.mlp.fc2_projection.in_features)
        self.output_projection.pc_layer_projection.register_lateral("projection_linear_output", self.output_projection.output_projection.in_features)

        for module in self.modules():
            if hasattr(module, 'W_latents'):
                for key in module.W_latents:
                    if module.W_latents[key] is not None:
                        module.W_latents[key] = module.W_latents[key].to(next(self.parameters()).device)

    def forward(self, target_ids, input_ids, use_kv_cache=False):
        """
        Forward pass of the PCTransformer model, using device-specific parallelism (CUDA streams or torch.jit.fork).

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

        # Initialize all predictive coding layers
        
        
        self.embedding_projection.pc_layer_projection.init_q(
            batch_size=B,
            seq_len=S,
            layer_type="projection_embed",
            device = device,
            layer={"word": self.embedding_projection.word_embeddings_projection, "pos": self.embedding_projection.position_embeddings_projection},
            proj_layers=None,
            input_ids=input_ids,
            position_ids=position_ids,
        )


        for block_projection in self.blocks_projection:
            
            block_projection.attn.pc_qkv_projection.init_q(
                batch_size=B,
                seq_len=S,
                layer_type="projection_attn",
                device=device,
                layer=None,
                proj_layers={
                    "q_proj": block_projection.attn.q_projection,
                    "k_proj": block_projection.attn.k_projection,
                    "v_proj": block_projection.attn.v_projection
                },
                input_ids=None,
                position_ids=None,
            )
        
            block_projection.attn.pc_output_projection.init_q(
                batch_size=B,
                seq_len=S,
                layer_type="projection_linear_attn",
                device=device,
                layer=block_projection.attn.output_projection,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )
        
            block_projection.mlp.pc_layer1_projection.init_q(
                batch_size=B,
                seq_len=S,
                layer_type="projection_fc1",
                device=device,
                layer=block_projection.mlp.fc1_projection,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )
        
            block_projection.mlp.pc_layer2_projection.init_q(
                batch_size=B,
                seq_len=S,
                layer_type="projection_fc2",
                device=device,
                layer=block_projection.mlp.fc2_projection,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )

       
        self.output_projection.pc_layer_projection.init_q(
            batch_size=B,
            seq_len=S,
            layer_type="projection_linear_output",
            device=device,
            layer=self.output_projection.output_projection,
            proj_layers= None, 
            input_ids = None,
            position_ids = None,
        )


        self.embedding.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer_type="embed",
            device = device,
            layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
            proj_layers=None,
            input_ids=input_ids,
            position_ids=position_ids,
        )
        for block in self.blocks:
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="attn",
                device = device,
                layer = None,
                proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                input_ids = None,
                position_ids = None,
            )
            block.attn.pc_output.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="linear_attn",
                device=device,
                layer=block.attn.output,
                proj_layers= None, 
                input_ids = None,
                position_ids = None,
            )
            block.mlp.pc_layer1.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="fc1",
                device=device,
                layer=block.mlp.fc1,
                proj_layers= None, 
                input_ids = None,
                position_ids = None,
            )
            block.mlp.pc_layer2.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="fc2",
                device=device,
                layer=block.mlp.fc2,
                proj_layers= None, 
                input_ids = None,
                position_ids = None,
            )
        self.output.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer_type="linear_output",
            device=device,
            layer=self.output.output,
            proj_layers= None, 
            input_ids = None,
            position_ids = None,
        )
        # one forward path for projection


        # Initialize streams or futures for parallel execution
        # Initialize streams or futures for parallel execution
        use_cuda, streams_or_futures = create_streams_or_futures(
            device, 
            2 * (len(self.blocks) * 4 + 2)
        )

        # TODO CORRECT PROJECTION

        execute_parallel(
            use_cuda,
            streams_or_futures,
            self.embedding_projection.pc_layer_projection.forward,
            target_activity=self.blocks_projection[0].attn.pc_qkv_projection.get_q("projection_attn"),
            layer_type="projection_embed",
            t=0,
            T=1,
            requires_update=False,
            td_err=None,
            layer={
                "word": self.embedding_projection.word_embeddings_projection,
                "pos": self.embedding_projection.position_embeddings_projection
            },
            layer_norm=self.blocks_projection[0].ln2,
            proj_layers=None,
            input_ids=input_ids,
            position_ids=position_ids,
            flash=False
        )


        # Iterate through blocks_projection in reverse order
        for idx in range(len(self.blocks_projection) - 1, -1, -1):

            block_projection = self.blocks_projection[idx]

            if idx == 0:
                td_embed = self.embedding_projection.pc_layer_projection.get_td_err_projection("projection_embed")
            else:
                td_embed = self.blocks_projection[idx - 1].mlp.pc_layer2_projection.get_td_err_projection("projection_fc2")


            execute_parallel(
                use_cuda,
                streams_or_futures,
                block_projection.attn.pc_qkv_projection.forward,
                target_activity=block_projection.attn.pc_output_projection.get_q("projection_linear_attn"),
                layer_type="projection_attn",
                t=0,
                T=1,
                requires_update=False,
                td_err=td_embed,
                layer=None,
                layer_norm=block_projection.ln2,
                proj_layers={
                    "q_proj": block_projection.attn.q_projection,
                    "k_proj": block_projection.attn.k_projection,
                    "v_proj": block_projection.attn.v_projection
                },
                input_ids=None,
                position_ids=None,
                flash=getattr(self.config, "use_flash_attention", False),
                use_cache=use_kv_cache,
                kv_cache=block_projection.attn.kv_cache_projection if use_kv_cache else None,
            )


            # Update projection KV cache
            if hasattr(block_projection.attn.pc_qkv_projection, "_last_kv_cache_projection"):
                block_projection.attn.kv_cache_projection = block_projection.attn.pc_qkv_projection._last_kv_cache_projection


            td_attn_qkv = (
                block_projection.attn.pc_qkv_projection.get_td_err_projection("projection_attn")
            )


            execute_parallel(
                use_cuda,
                streams_or_futures,
                block_projection.attn.pc_output_projection.forward,
                target_activity=block_projection.mlp.pc_layer1_projection.get_q("projection_fc1"),
                layer_type="projection_linear_attn",
                t=0,
                T=1,
                requires_update=False,
                td_err=td_attn_qkv,
                layer=block_projection.attn.output_projection,
                layer_norm=block_projection.ln1,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
                flash=False
            )


            td_attn_op = block_projection.attn.pc_output_projection.get_td_err_projection("projection_linear_attn")


            execute_parallel(
                use_cuda,
                streams_or_futures,
                block_projection.mlp.pc_layer1_projection.forward,
                target_activity=block_projection.mlp.pc_layer2_projection.get_q("projection_fc2"),
                layer_type="projection_fc1",
                t=0,
                T=1,
                requires_update=False,
                td_err=td_attn_op,
                layer=block_projection.mlp.fc1_projection,
                layer_norm=block_projection.ln1,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
                flash=False
            )


            next_target = (
                self.blocks_projection[idx + 1]
                .attn.pc_qkv_projection
                .get_q("projection_attn")
                if idx < len(self.blocks_projection) - 1
                else self.output_projection.pc_layer_projection.get_q("projection_linear_output")
            )


            layer_norm2 = (
                block_projection.ln2
                if idx < len(self.blocks_projection) - 1
                else None
            )


            td_mlp1 = block_projection.mlp.pc_layer1_projection.get_td_err_projection("projection_fc1")


            execute_parallel(
                use_cuda,
                streams_or_futures,
                block_projection.mlp.pc_layer2_projection.forward,
                target_activity=next_target,
                layer_type="projection_fc2",
                t=0,
                T=1,
                requires_update=False,
                td_err=td_mlp1,
                layer=block_projection.mlp.fc2_projection,
                layer_norm=layer_norm2,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
                flash=False
            )


        td_mlp2 = self.blocks_projection[-1].mlp.pc_layer2_projection.get_td_err_projection("projection_fc2")


        execute_parallel(
            use_cuda,
            streams_or_futures,
            self.output_projection.pc_layer_projection.forward,
            target_activity=target_logits,
            layer_type="projection_linear_output",
            t=0,
            T=1,
            requires_update=False,
            td_err=td_mlp2,
            layer=self.output_projection.output_projection,
            layer_norm=None,
            proj_layers=None,
            input_ids=None,
            position_ids=None,
            flash=False
        )

        synchronize_execution(use_cuda, streams_or_futures)

        for idx, block in enumerate(self.blocks):
            block.attn.pc_qkv.init_q(
                projection_layer_type="projection_attn",
                q=self.blocks_projection[idx].attn.pc_qkv_projection.get_q("projection_attn"),
                device=device,
            )
            block.attn.pc_output.init_q(
                projection_layer_type="projection_linear_attn",
                q=self.blocks_projection[idx].attn.pc_output_projection.get_q("projection_linear_attn"),
                device=device,
            )
            block.mlp.pc_layer1.init_q(
                projection_layer_type="projection_fc1",
                q=self.blocks_projection[idx].mlp.pc_layer1_projection.get_q("projection_fc1"),
                device=device,
            )
            block.mlp.pc_layer2.init_q(
                projection_layer_type="projection_fc2",
                q=self.blocks_projection[idx].mlp.pc_layer2_projection.get_q("projection_fc2"),
                device=device,
            )

        self.output.pc_layer.init_q(
            projection_layer_type="projection_linear_output",
            q=self.output_projection.pc_layer_projection.get_q("projection_linear_output"),
            device=device,
        )


        # finished projection 
        
        
        for t in range(self.config.T):

            do_update = (t == self.config.T - 1)


            execute_parallel(
                use_cuda,
                streams_or_futures,
                self.embedding.pc_layer.forward,
                target_activity=self.blocks[0].attn.pc_qkv.get_x("attn"),
                layer_type="embed",
                t=t,
                T=self.config.T,
                requires_update=do_update,
                td_err = None,
                layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
                layer_norm=self.blocks[0].ln2,
                proj_layers=None,
                input_ids=input_ids,
                position_ids=position_ids,
                flash=False
            )
            # Execute output layer
            

            # Iterate through blocks in reverse order for parallel execution
            for idx in range(len(self.blocks) - 1, -1, -1):
                
                block = self.blocks[idx]

                if idx == 0:
                   td_embed = self.embedding.pc_layer.get_td_err("embed") if t > 0 else None
                else:
                   td_embed = self.blocks[idx - 1].mlp.pc_layer2.get_td_err("fc2") if t > 0 else None
                
                
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_qkv.forward,
                    target_activity=block.attn.pc_output.get_x("linear_attn"),
                    layer_type="attn",
                    t=t,
                    T=self.config.T,
                    requires_update=do_update,
                    td_err= td_embed,
                    layer = None,
                    layer_norm=block.ln2,
                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                    input_ids=None,
                    position_ids=None,
                    flash=getattr(self.config, 'use_flash_attention', False),
                    use_cache=use_kv_cache,  
                    kv_cache=block.attn.kv_cache if use_kv_cache else None, 
                )

                # Update cache after last iteration
                if use_kv_cache and t == self.config.T - 1:
                    block.attn.kv_cache = block.attn.pc_qkv._last_kv_cache

                td_attn_qkv = block.attn.pc_qkv.get_td_err("attn") if t > 0 else None

                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_output.forward,
                    target_activity=block.mlp.pc_layer1.get_x("fc1"),
                    layer_type="linear_attn",
                    t=t,
                    T=self.config.T,
                    requires_update=do_update,
                    td_err= td_attn_qkv,
                    layer=block.attn.output, 
                    layer_norm=block.ln1,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False

                )
                td_attn_op = block.attn.pc_output.get_td_err("linear_attn") if t > 0 else None

                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.mlp.pc_layer1.forward,
                    target_activity=block.mlp.pc_layer2.get_x("fc2"),
                    layer_type="fc1",
                    t=t,
                    T=self.config.T,
                    requires_update=do_update,
                    td_err=td_attn_op,
                    layer=block.mlp.fc1,
                    layer_norm=block.ln1,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False

                )


                next_target = (
                    self.blocks[idx + 1].attn.pc_qkv.get_x("attn")
                    if idx < len(self.blocks) - 1
                    else self.output.pc_layer.get_x("linear_output")
                )
                
                layer_norm2 = (block.ln2
                   if idx < len(self.blocks) - 1
                    else None)
                td_mlp1 = block.mlp.pc_layer1.get_td_err("fc1") if t > 0 else None

                # Execute MLP layer 2
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.mlp.pc_layer2.forward,
                    target_activity=next_target,
                    layer_type="fc2",
                    t=t,
                    T=self.config.T,
                    requires_update=do_update,
                    td_err= td_mlp1,
                    layer=block.mlp.fc2,
                    layer_norm=layer_norm2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False

                )
                
                
                
    
                
            
    
            
            td_mlp2 = self.blocks[-1].mlp.pc_layer2.get_td_err("fc2") if t > 0 else None
            execute_parallel(
                use_cuda,
                streams_or_futures,
                self.output.pc_layer.forward,
                target_activity=target_logits,
                layer_type="linear_output",
                t=t,
                T=self.config.T,
                requires_update=do_update,
                td_err= td_mlp2,
                layer=self.output.output,
                layer_norm=None,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
                flash=False

            )
            # Execute embedding layer
            

            # Synchronize all parallel tasks
            synchronize_execution(use_cuda, streams_or_futures)
        logits = self.output.pc_layer.get_mu("linear_output")
        return logits
    
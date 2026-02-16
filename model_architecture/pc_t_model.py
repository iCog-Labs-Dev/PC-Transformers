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
        super().__init__()
        self.config = config
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)

    def register_all_lateral_weights(self):
        """
        Register lateral weights for all predictive coding layers in the model.
        This enables lateral connections for local learning in each layer.
        """
        for block in self.blocks:
            for lt in ["x_query", "x_key", "x_value"]:
                block.attn.pc_qkv.register_lateral(lt, block.attn.q.in_features)
            block.attn.pc_output.register_lateral("linear", block.attn.output.in_features)
            block.mlp.pc_layer1.register_lateral("fc1", block.mlp.fc1.in_features)
            block.mlp.pc_layer2.register_lateral("linear", block.mlp.fc2.in_features)
        self.output.pc_layer.register_lateral("linear", self.output.output.in_features)

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
                layer_type="x_query",
                device=device,
                layer=block.attn.q,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="x_key",
                device=device,
                layer=block.attn.k,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="x_value",
                device=device,
                layer=block.attn.v,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="x_score",
                device=device,
                layer=block.attn.score,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
            )
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="x_A",
                device=device,
                layer=block.attn.A,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
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

        for t in range(self.config.T):
            # Execute output layer
            td_mlp2 = self.blocks[-1].mlp.pc_layer2.get_td_err("fc2") if t > 0 else None
            self.output.pc_layer.forward(
                target_activity=target_logits,
                layer_type="linear_output",
                t=t,
                T=self.config.T,
                requires_update=True,
                td_err=td_mlp2,
                layer=self.output.output,
                layer_norm=None,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
                flash=False,
            )

            # Iterate through blocks in reverse order (sequential to preserve dependencies)
            for idx in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[idx]
                next_target = (
                    self.blocks[idx + 1].attn.pc_qkv.get_x("x_query")
                    if idx < len(self.blocks) - 1
                    else self.output.pc_layer.get_x("linear_output")
                )
                
                layer_norm2 = (block.ln2
                   if idx < len(self.blocks) - 1
                    else None)
                td_mlp1 = block.mlp.pc_layer1.get_td_err("fc1") if t > 0 else None

                # Execute MLP layer 2
                block.mlp.pc_layer2.forward(
                    target_activity=next_target,
                    layer_type="fc2",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_mlp1,
                    layer=block.mlp.fc2,
                    layer_norm=layer_norm2,
                    proj_layers={"residual": block.attn.pc_output.get_x("linear_attn")},
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )

                td_attn_op = block.attn.pc_output.get_td_err("linear_attn") if t > 0 else None

                # Execute MLP layer 1
                block.mlp.pc_layer1.forward(
                    target_activity=block.mlp.pc_layer2.get_x("fc2"),
                    layer_type="fc1",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_attn_op,
                    layer=block.mlp.fc1,
                    layer_norm=block.ln1,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )
                
                if idx == 0:
                   td_embed = self.embedding.pc_layer.get_td_err("embed") if t > 0 else None
                else:
                   td_embed = self.blocks[idx - 1].mlp.pc_layer2.get_td_err("fc2") if t > 0 else None

                # linear_attn td_err comes from {x_A, x_value} (previous iteration)
                td_x_A = block.attn.pc_qkv.get_td_err("x_A") if t > 0 else None
                td_x_value = block.attn.pc_qkv.get_td_err("x_value") if t > 0 else None
                td_linear_attn = None
                if td_x_A is not None and td_x_value is not None:
                    td_linear_attn = 0.5 * (td_x_A + td_x_value)
                elif td_x_A is not None:
                    td_linear_attn = td_x_A
                elif td_x_value is not None:
                    td_linear_attn = td_x_value

                # Execute attention output (linear_attn)
                # linear_attn is computed as A @ V using cached mu from pc_qkv.
                mu_A = block.attn.pc_qkv.get_mu("x_A")
                if mu_A is None:
                    mu_A = block.attn.pc_qkv.get_x("x_A")

                if use_kv_cache and getattr(block.attn.pc_qkv, "_last_kv_cache", None) is not None and block.attn.pc_qkv._last_kv_cache[1] is not None:
                    mu_V = block.attn.pc_qkv._last_kv_cache[1]
                else:
                    mu_V = block.attn.pc_qkv.get_mu("x_value")
                    if mu_V is None:
                        mu_V = block.attn.pc_qkv.get_x("x_value")

                block.attn.pc_output.forward(
                    target_activity=block.mlp.pc_layer1.get_x("fc1"),
                    layer_type="linear_attn",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_linear_attn,
                    layer=block.attn.output,
                    layer_norm=block.ln1,
                    proj_layers={
                        "mu_A": mu_A,
                        "mu_V": mu_V,
                        "num_heads": self.config.num_heads,
                        "residual": block.attn.pc_qkv.get_x("x_query"),
                    },
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )

                # x_value target linear_attn, td_error from embed
                block.attn.pc_qkv.forward(
                    target_activity=block.attn.pc_output.get_x("linear_attn"),
                    layer_type="x_value",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_embed,
                    layer=block.attn.v,
                    layer_norm=block.ln2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                    use_cache=use_kv_cache,
                    kv_cache=block.attn.kv_cache if use_kv_cache else None,
                )

                # x_query target x_score, td_error from embed
                block.attn.pc_qkv.forward(
                    target_activity=block.attn.pc_qkv.get_x("x_query"),
                    layer_type="x_query",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_embed,
                    layer=block.attn.q,
                    layer_norm=block.ln2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )

                # x_key target x_score, td_error from embed
                block.attn.pc_qkv.forward(
                    target_activity=block.attn.pc_qkv.get_x("x_key"),
                    layer_type="x_key",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_embed,
                    layer=block.attn.k,
                    layer_norm=block.ln2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                    use_cache=use_kv_cache,
                    kv_cache=block.attn.kv_cache if use_kv_cache else None,
                )

                # x_score target x_A, td_error from {x_query, x_key} (previous iteration)
                td_x_query = block.attn.pc_qkv.get_td_err("x_query") if t > 0 else None
                td_x_key = block.attn.pc_qkv.get_td_err("x_key") if t > 0 else None
                td_x_score = None
                if td_x_query is not None and td_x_key is not None:
                    td_x_score = 0.5 * (td_x_query + td_x_key)
                elif td_x_query is not None:
                    td_x_score = td_x_query
                elif td_x_key is not None:
                    td_x_score = td_x_key

                block.attn.pc_qkv.forward(
                    target_activity=block.attn.pc_qkv.get_x("x_A"),
                    layer_type="x_score",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_x_score,
                    layer=block.attn.score,
                    layer_norm=block.ln2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                    use_cache=use_kv_cache,
                    kv_cache=block.attn.kv_cache if use_kv_cache else None,
                )

                # x_A target linear_attn, td_error from x_score (previous iteration)
                td_from_x_score = block.attn.pc_qkv.get_td_err("x_score") if t > 0 else None
                block.attn.pc_qkv.forward(
                    target_activity=block.attn.pc_qkv.get_x("x_A"),
                    layer_type="x_A",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=td_from_x_score,
                    layer=block.attn.A,
                    layer_norm=block.ln2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )

                # Update cache after last iteration
                if use_kv_cache and t == self.config.T - 1:
                    block.attn.kv_cache = block.attn.pc_qkv._last_kv_cache
    
            # Execute embedding layer (provides td_err used by x_query/x_key/x_value)
            self.embedding.pc_layer.forward(
                target_activity=self.blocks[0].attn.pc_qkv.get_x("x_query"),
                layer_type="embed",
                t=t,
                T=self.config.T,
                requires_update=True,
                td_err=None,
                layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
                layer_norm=self.blocks[0].ln2,
                proj_layers=None,
                input_ids=input_ids,
                position_ids=position_ids,
                flash=False,
            )
        logits = self.output.pc_layer.get_mu("linear_output")
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class ProjectionAttention(nn.Module):
    """Feedforward attention block for projection network (no iterative inference)."""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.num_heads
        
        self.q_proj = nn.Linear(config.n_embed, config.n_embed) #wq
        self.k_proj = nn.Linear(config.n_embed, config.n_embed) #wk
        self.v_proj = nn.Linear(config.n_embed, config.n_embed) #wv
        self.output_proj = nn.Linear(config.n_embed, config.n_embed) #wattn
        self.dropout = nn.Dropout(config.projection_dropout if hasattr(config, 'projection_dropout') else 0.0)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  #Q = Wq*X # (B, H, S, head_dim)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) #
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1) 
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, H, S, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        return self.dropout(self.output_proj(attn_output))

class ProjectionMLP(nn.Module):
    """Feedforward MLP block for projection network."""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.projection_dropout if hasattr(config, 'projection_dropout') else 0.0)
        self.gelu = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.dropout(self.fc2(x))

class ProjectionBlock(nn.Module):
    """Single transformer block for projection network."""
    def __init__(self, config, block_id: int):
        super().__init__()
        self.block_id = block_id
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = ProjectionAttention(config)
        self.mlp = ProjectionMLP(config)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual + layer norm
        x = x + self.attn(self.ln1(x), mask)
        # MLP with residual + layer norm
        x = x + self.mlp(self.ln2(x))
        return x

class ProjectionNetwork(nn.Module):
    """
    Feedforward projection network that mirrors PCTransformer architecture.
    Used to generate warm-start initializations for predictive coding latents.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        
        # Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([ProjectionBlock(config, i) for i in range(config.n_blocks)])
        
        # Output projection
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.output_proj = nn.Linear(config.n_embed, config.vocab_size)
        
        # Cache for intermediate latents (to initialize PC model)
        self._latent_cache: Dict[str, torch.Tensor] = {}
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None, 
                return_latents: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_ids: (B, S) token IDs
            position_ids: (B, S) position IDs (optional, auto-generated if None)
            return_latents: if True, store intermediate activations for PC initialization
            
        Returns:
            logits: (B, S, vocab_size) output predictions
            latents: dict of intermediate activations {layer_name: tensor}
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        
        # Embedding
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.emb_dropout(x)
        
        if return_latents:
            self._latent_cache["embed"] = x.detach().clone()
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_latents:
                # Store post-attention and post-mlp states separately if needed
                self._latent_cache[f"block{i}_attn"] = x.detach().clone()
                self._latent_cache[f"block{i}_mlp"] = x.detach().clone()
        
        # Output
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        if return_latents:
            self._latent_cache["output"] = x.detach().clone()
            
        return logits, self._latent_cache if return_latents else {}
    def _copy_norm_weights(self, proj_norm: nn.Module, pc_norm: nn.Module):
        """Safely copy normalization layer weights, handling LayerNorm/RMSNorm."""
        proj_norm.weight.data = pc_norm.weight.data.clone()
        if hasattr(pc_norm, 'bias') and pc_norm.bias is not None:
            proj_norm.bias.data = pc_norm.bias.data.clone()
    
    def _copy_linear_weights(self, proj_lin: nn.Linear, pc_lin: nn.Linear):
        """Safely copy linear layer weights, handling bias/no-bias."""
        proj_lin.weight.data = pc_lin.weight.data.clone()
        if hasattr(pc_lin, 'bias') and pc_lin.bias is not None:
            proj_lin.bias.data = pc_lin.bias.data.clone()
    def copy_weights_from(self, pc_model: nn.Module):
        """
        Copy weights from the main PCTransformer to this projection network.
        Robustly handles attribute name variations (e.g., ln vs ln_f, output vs head).
        """
        import torch.nn as nn

        # --- 1. Copy Embeddings ---
        if hasattr(pc_model.embedding, 'word_embeddings'):
            self.word_embeddings.weight.data = pc_model.embedding.word_embeddings.weight.data.clone()
        if hasattr(pc_model.embedding, 'position_embeddings'):
            self.position_embeddings.weight.data = pc_model.embedding.position_embeddings.weight.data.clone()
        
        # --- 2. Copy Transformer Blocks ---
        for proj_block, pc_block in zip(self.blocks, pc_model.blocks):
            # Helper to safely copy norms (LayerNorm or RMSNorm)
            def safe_copy_norm(proj_norm, pc_norm):
                proj_norm.weight.data = pc_norm.weight.data.clone()
                if hasattr(pc_norm, 'bias') and pc_norm.bias is not None:
                    proj_norm.bias.data = pc_norm.bias.data.clone()

            # Helper to safely copy linear layers
            def safe_copy_linear(proj_lin, pc_lin):
                proj_lin.weight.data = pc_lin.weight.data.clone()
                if hasattr(pc_lin, 'bias') and pc_lin.bias is not None:
                    proj_lin.bias.data = pc_lin.bias.data.clone()

            # Copy Norms (Search for common names)
            for attr in ['ln1', 'ln_1', 'norm1', 'ln_0']:
                if hasattr(pc_block, attr) and hasattr(proj_block, attr):
                    safe_copy_norm(getattr(proj_block, attr), getattr(pc_block, attr))
                    break
            for attr in ['ln2', 'ln_2', 'norm2', 'ln_1']:
                if hasattr(pc_block, attr) and hasattr(proj_block, attr):
                    safe_copy_norm(getattr(proj_block, attr), getattr(pc_block, attr))
                    break
            
            # Copy Attention Projections
            attn_map = [('q_proj', ['q', 'q_proj', 'W_q']), 
                        ('k_proj', ['k', 'k_proj', 'W_k']), 
                        ('v_proj', ['v', 'v_proj', 'W_v']), 
                        ('output_proj', ['output', 'out_proj', 'o_proj', 'W_o'])]
            
            for proj_attr, pc_attrs in attn_map:
                if hasattr(proj_block.attn, proj_attr):
                    for pc_attr in pc_attrs:
                        if hasattr(pc_block.attn, pc_attr):
                            pc_layer = getattr(pc_block.attn, pc_attr)
                            if isinstance(pc_layer, nn.Linear):
                                safe_copy_linear(getattr(proj_block.attn, proj_attr), pc_layer)
                                break
            
            # Copy MLP Layers
            mlp_map = [('fc1', ['fc1', 'mlp1', 'c_fc', 'W1']), 
                       ('fc2', ['fc2', 'mlp2', 'c_proj', 'W2'])]
            
            for proj_attr, pc_attrs in mlp_map:
                if hasattr(proj_block.mlp, proj_attr):
                    for pc_attr in pc_attrs:
                        if hasattr(pc_block.mlp, pc_attr):
                            pc_layer = getattr(pc_block.mlp, pc_attr)
                            if isinstance(pc_layer, nn.Linear):
                                safe_copy_linear(getattr(proj_block.mlp, proj_attr), pc_layer)
                                break

        # --- 3. Copy Output Layer (The part causing your error) ---
        # Search for Normalization Layer
        pc_out_norm = None
        for attr in ['ln_f', 'ln', 'norm', 'final_norm', 'layer_norm']:
            if hasattr(pc_model.output, attr):
                candidate = getattr(pc_model.output, attr)
                if isinstance(candidate, (nn.LayerNorm, nn.RMSNorm)) or hasattr(candidate, 'weight'):
                    pc_out_norm = candidate
                    break
        
        if pc_out_norm and hasattr(self, 'ln_f'):
            safe_copy_norm(self.ln_f, pc_out_norm)
        
        # Search for Output Projection Layer
        pc_out_proj = None
        for attr in ['output', 'proj', 'head', 'lm_head', 'classifier', 'decoder']:
            if hasattr(pc_model.output, attr):
                candidate = getattr(pc_model.output, attr)
                if isinstance(candidate, nn.Linear):
                    pc_out_proj = candidate
                    break
        
        if pc_out_proj and hasattr(self, 'output_proj'):
            safe_copy_linear(self.output_proj, pc_out_proj)
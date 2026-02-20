import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    x_init,
    step_embed,
    step_linear,
    step_attn,
    finalize_step,
)
from predictive_coding.lateral_connc import LateralConnections

class PCLayer(nn.Module):
    """
    Predictive Coding Layer wrapper that manages iterative inference state and
    delegates computation to helper functions (step_embed, step_attn, step_linear).
    """
    def __init__(
        self,
        T: int,
        lr: float,
        update_bias: bool,
        energy_fn_name: str,
        num_heads: Optional[int] = None,
        n_embed: Optional[int] = None,
    ):
        super().__init__()
        self.T = T
        self.local_lr = lr
        self.update_bias = update_bias
        self.clamp_value = 3.0
        self.energy_fn_name = energy_fn_name 
        self.num_heads = num_heads
        self.n_embed = n_embed
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy = 0.0
        self._errors = []
    
    def register_lateral(self, layer_type: str, size: int):
        """Create and register lateral connections for layer_type."""
        if layer_type not in self.lateral_connections:
            self.lateral_connections[layer_type] = LateralConnections(size, self.local_lr)
            self.add_module(f"lateral_{layer_type}", self.lateral_connections[layer_type])

    def _reset_step_state(self) -> None:
        """Reset step-local accumulators, kept for future extension."""
        return
    
    def _get_cached_state(self, layer_type: str):
        return self._x_cache.get(layer_type, None)
    
    def forward(
        self,
        target_activity: torch.Tensor,
        layer_type: str,
        t: int,
        T: int,
        requires_update: bool,
        td_err:  Optional[torch.Tensor] = None,
        layer: Optional[nn.Module] = None,
        layer_norm: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        flash: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # ADD THIS
        use_cache: bool = False, 
    ):
        """Perform one predictive coding inference step."""
        self._reset_step_state()
        x = self._get_cached_state(layer_type)

        if layer_type == "embed":
            mu, mu_word, mu_pos, bu_err = step_embed(
                t,
                T,
                target_activity,
                layer,
                layer_type,
                input_ids,
                position_ids,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                requires_update,
                layer_norm=layer_norm,
            )            
            # store for later retrieval
            self._x_cache["embed"] = (mu_word, mu_pos)
            self._mu_cache["embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache["embed"] = bu_err.detach().clone()

            # compute energy
            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._errors.extend(step_errors)
            return mu_word, mu_pos
        
        elif layer_type == "attn":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err, new_kv_cache = step_attn(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                proj_layers,
                layer_type,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                self.update_bias,
                requires_update,
                self.num_heads,
                self.n_embed,
                td_err=td_err, 
                layer_norm=layer_norm,
                flash=flash, 
                kv_cache=kv_cache,  
                use_cache=use_cache,
            )
            # Store cache for retrieval
            if use_cache:
                self._last_kv_cache = new_kv_cache
        
        else:
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err = step_linear(
                t,
                T,
                target_activity,
                x,
                layer, 
                lateral_conn,  
                layer_type,
                self.local_lr, 
                self.clamp_value, 
                self.energy_fn_name, 
                self.update_bias, 
                requires_update,
                td_err=td_err, 
                layer_norm=layer_norm
            )
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None: 
         self._error_cache[layer_type] = bu_err.detach().clone()   
        
        error = target_activity - mu
        energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._energy += energy
        self._errors.extend(step_errors)

        # update x cache
        self._x_cache[layer_type] = x
        return x, mu

    
    
    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        projection_latents: Optional[Dict[str, torch.Tensor]] = None,
        use_projection_init: bool = False,
        block_index: Optional[int] = None,
    ):
        """Initialize cached activity `x` for the layer type."""
        # >>> Use projection init if requested and available <<<
        if use_projection_init and projection_latents is not None:
            # For embed layer, skip projection init (fall back to original)
            if layer_type != "embed":
                self.init_x_from_projection(
                    projection_latents, 
                    layer_type, 
                    device, 
                    block_index,
                    layer=layer,  # NEW
                    proj_layers=proj_layers  # NEW
                )
                # Check if projection init succeeded (cache not set to None)
                if self._x_cache.get(layer_type) is not None:
                    return  # Success, early return
        
        # >>> ORIGINAL INIT CODE (for embed layer or projection fallback) <<<
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None, "Embedding layer requires input_ids and position_ids"
            vocab_size = layer["word"].weight.size(0)
            if input_ids.max() >= vocab_size:
                input_ids = torch.clamp(input_ids, max=vocab_size-1)
            
            max_pos = layer["pos"].weight.size(0)
            if position_ids.max() >= max_pos:
                position_ids = torch.clamp(position_ids, max=max_pos-1)
            
            x_word = layer["word"].weight[input_ids] 
            x_pos = layer["pos"].weight[position_ids] 
            self._x_cache["embed"] = (x_word, x_pos)
            
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            self._x_cache["attn"] = x_init(batch_size, seq_len, H_out, device)
            
            self.register_lateral(layer_type, H_in)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            
            self.register_lateral(layer_type, input_dim)  
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
    def init_x_from_projection(
        self,
        projection_latents: Dict[str, torch.Tensor],
        layer_type: str,
        device: torch.device,
        block_index: Optional[int] = None,
        layer: Optional[nn.Module] = None,  # NEW: to get correct input_dim
        proj_layers: Optional[dict] = None,  # NEW: for attention layers
    ):
        """
        Initialize cached activity `x` using latents from projection network.
        Ensures correct feature dimensions for each layer type.
        """
        # Get batch_size and seq_len from embed latent
        embed_latent = projection_latents.get("embed")
        if embed_latent is not None:
            B, S = embed_latent.shape[0], embed_latent.shape[1]
        else:
            B, S = 1, 1  # fallback
        
        if layer_type == "embed":
            # >>> FIX: Fall back to original embed init (projection doesn't have separate word/pos) <<<
            # Projection stores combined embedding, but PC embed expects tuple (word, pos)
            # We'll use the original init logic for embed layer
            self._x_cache["embed"] = None  # Signal to use original init
            return
            
        elif layer_type == "attn":
            key = f"block{block_index}_attn" if block_index is not None else "block0_attn"
            attn_latent = projection_latents.get(key)
            
            if attn_latent is not None:
                # Get correct output dimension from proj_layers
                if proj_layers is not None and "v_proj" in proj_layers:
                    H_out = proj_layers["v_proj"].weight.shape[0]
                    # Ensure latent has correct shape
                    if attn_latent.shape[2] == H_out:
                        self._x_cache["attn"] = attn_latent.to(device)
                    else:
                        # Shape mismatch - use random init
                        self._x_cache["attn"] = x_init(B, S, H_out, device)
                else:
                    self._x_cache["attn"] = x_init(B, S, self.n_embed or 768, device)
            else:
                H_out = self.n_embed or 768
                self._x_cache["attn"] = x_init(B, S, H_out, device)
            
            # Register lateral connections
            if proj_layers is not None and "q_proj" in proj_layers:
                H_in = proj_layers["q_proj"].weight.shape[1]
                self.register_lateral(layer_type, H_in)
                if layer_type in self.lateral_connections:
                    self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)
                
        elif layer_type in ["fc1", "fc2", "linear_attn", "linear_output"]:
            # >>> FIX: Get correct input dimension from the actual layer <<<
            if layer is not None and hasattr(layer, 'weight'):
                input_dim = layer.weight.shape[1]  # Correct input dimension
            else:
                input_dim = self.n_embed or 768  # Fallback
            
            # Map layer_type to projection cache key
            key_map = {
                "fc1": f"block{block_index}_mlp" if block_index is not None else "block0_mlp",
                "fc2": f"block{block_index}_mlp" if block_index is not None else "block0_mlp",
                "linear_attn": f"block{block_index}_attn" if block_index is not None else "block0_attn",
                "linear_output": "output"
            }
            latent = projection_latents.get(key_map.get(layer_type, layer_type))
            
            if latent is not None:
                # Check if latent has correct feature dimension
                if latent.shape[2] == input_dim:
                    self._x_cache[layer_type] = latent.to(device)
                else:
                    # Shape mismatch - use random init with correct dimension
                    self._x_cache[layer_type] = x_init(B, S, input_dim, device)
            else:
                # No latent available - use random init
                self._x_cache[layer_type] = x_init(B, S, input_dim, device)
            
            # Register lateral connections
            self.register_lateral(layer_type, input_dim)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)          
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached activity tensor for a given layer type."""
        return self._x_cache.get(layer_type, None)
    
    def get_mu(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached mu (prediction) tensor for a given layer type."""
        return self._mu_cache.get(layer_type, None)
    
    def get_td_err(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached top-down error tensor for a given layer type."""
        return self._error_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        """Get the accumulated energy for the layer."""
        return float(self._energy)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        
    def get_errors(self) -> list:
        """Get the list of error values accumulated during inference."""
        return self._errors

    def clear_errors(self):
        """Clear the stored errors for the layer."""
        self._errors = []
        
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)

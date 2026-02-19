import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    # x_init is removed from here since we no longer use it
    step_embed,
    step_linear,
    step_attn,
    finalize_step,
)
from utils.optim.optim_utils import PCOptimizer
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
        optimizer_name: str = "adam",
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
        optimizer_sign_value: float = -1.0,
        optimizer_weight_bound: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.local_lr = lr
        self.update_bias = update_bias
        self.clamp_value = 3.0
        self.energy_fn_name = energy_fn_name 
        self.num_heads = num_heads
        self.n_embed = n_embed

        self.optimizer = PCOptimizer(
            opt_name=optimizer_name,
            beta1=optimizer_beta1,
            beta2=optimizer_beta2,
            eps=optimizer_eps,
            sign_value=optimizer_sign_value,
            weight_bound=optimizer_weight_bound,
        )
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy = 0.0
        self._errors = []
        self._q_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache_projection: Dict[str, torch.Tensor] = {}
        self._error_cache_projection: Dict[str, torch.Tensor] = {}

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
        
    def _get_cached_state_projection(self, layer_type: str):
        return self._q_cache.get(layer_type, None)

    def _get_lateral_connection(self, layer_type: str, device: torch.device):
        lateral_conn = self.lateral_connections.get(layer_type, None)
        if lateral_conn is None:
            return None

        lateral_device = lateral_conn.W_lateral.device
        if lateral_device != device:
            lateral_conn = lateral_conn.to(device)
            self.lateral_connections[layer_type] = lateral_conn
        return lateral_conn
        
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        use_cache: bool = False, 
    ):
        """Perform one predictive coding inference step."""
        self._reset_step_state()
        x = self._get_cached_state(layer_type)
        q = self._get_cached_state_projection(layer_type)
        
        if layer_type == "embed":
            mu, mu_word, mu_pos, bu_err = step_embed(
                t, T, target_activity, layer, layer_type, input_ids, position_ids,
                self.local_lr, self.clamp_value, self.energy_fn_name, requires_update,
                layer_norm=layer_norm, optimizer=self.optimizer,
            )            
            self._x_cache["embed"] = (mu_word, mu_pos)
            self._mu_cache["embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache["embed"] = bu_err.detach().clone()

            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._errors.extend(step_errors)
        
            return mu_word, mu_pos
            
        elif layer_type == "projection_embed":
            mu, mu_word, mu_pos, bu_err = step_embed(
                t, T, target_activity, layer, layer_type, input_ids, position_ids,
                self.local_lr, self.clamp_value, self.energy_fn_name, requires_update,
                layer_norm=layer_norm, optimizer=self.optimizer,
            )            
            self._q_cache["projection_embed"] = (mu_word, mu_pos)
            self._mu_cache_projection["projection_embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache_projection["projection_embed"] = bu_err.detach().clone()

            return mu_word, mu_pos
        
        elif layer_type == "attn":
            lateral_conn = self._get_lateral_connection(layer_type, target_activity.device)
            x, mu, bu_err, new_kv_cache = step_attn(
                t, T, target_activity, x, lateral_conn, proj_layers, layer_type,
                self.local_lr, self.clamp_value, self.energy_fn_name, self.update_bias,
                requires_update, self.num_heads, self.n_embed, td_err=td_err, 
                layer_norm=layer_norm, flash=flash, kv_cache=kv_cache,  
                use_cache=use_cache, optimizer=self.optimizer,
            )
            if use_cache:
                self._last_kv_cache = new_kv_cache
                
        elif layer_type == "projection_attn":
            lateral_conn = self._get_lateral_connection(layer_type, target_activity.device)
            q, mu, bu_err, new_kv_cache = step_attn(
                t, T, target_activity, q, lateral_conn, proj_layers, layer_type,
                self.local_lr, self.clamp_value, self.energy_fn_name, self.update_bias,
                requires_update, self.num_heads, self.n_embed, td_err=td_err, 
                layer_norm=layer_norm, flash=flash, kv_cache=kv_cache,  
                use_cache=use_cache, optimizer=self.optimizer,
            )
            if use_cache:
                self._last_kv_cache_projection = new_kv_cache
            self._mu_cache_projection[layer_type] = mu.detach().clone()  
            if bu_err is not None:
                self._error_cache_projection[layer_type] = bu_err.detach().clone()
            self._q_cache[layer_type] = q
            return q, mu
         
        elif layer_type in ["projection_linear_attn", "projection_fc1", "projection_fc2", "projection_linear_output"]:
            lateral_conn = self._get_lateral_connection(layer_type, target_activity.device)
            q, mu, bu_err = step_linear(
                t, T, target_activity, q, layer, lateral_conn, layer_type,
                self.local_lr, self.clamp_value, self.energy_fn_name, self.update_bias, 
                requires_update, td_err=td_err, layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
            self._mu_cache_projection[layer_type] = mu.detach().clone()  
            if bu_err is not None:
                self._error_cache_projection[layer_type] = bu_err.detach().clone()
            self._q_cache[layer_type] = q
            return q, mu
            
        else:
            lateral_conn = self._get_lateral_connection(layer_type, target_activity.device)
            x, mu, bu_err = step_linear(
                t, T, target_activity, x, layer, lateral_conn, layer_type,
                self.local_lr, self.clamp_value, self.energy_fn_name, self.update_bias, 
                requires_update, td_err=td_err, layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
            
        # cache and stats for standard layers
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None:
            self._error_cache[layer_type] = bu_err.detach().clone()
        
        error = target_activity - mu
        energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._energy += energy
        self._errors.extend(step_errors)

        self._x_cache[layer_type] = x
        return x, mu

    def init_q(
        self,
        projection_layer_type: Optional[str] = None,
        q: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Two modes:
        1) Projection initialization mode (used for projection layers):
           init_q(batch_size=..., seq_len=..., layer_type="projection_*", ...)
           -> initializes self._q_cache for projection inference.

        2) Projection-to-standard injection mode (used after projection pass):
           init_q(projection_layer_type="projection_*", q=<tensor>, device=<device>)
           -> maps projected q into the corresponding self._x_cache target layer.
        """
        if batch_size is not None and seq_len is not None and layer_type is not None:
            if device is None:
                raise ValueError("device is required for init_q projection initialization mode")

            if layer_type == "projection_embed":
                assert input_ids is not None and position_ids is not None, "Projection embedding layer requires input_ids and position_ids"
                assert layer is not None and isinstance(layer, dict), "projection_embed requires layer={'word':..., 'pos':...}"

                vocab_size = layer["word"].weight.size(0)
                if input_ids.max() >= vocab_size:
                    input_ids = torch.clamp(input_ids, max=vocab_size - 1)

                max_pos = layer["pos"].weight.size(0)
                if position_ids.max() >= max_pos:
                    position_ids = torch.clamp(position_ids, max=max_pos - 1)

                q_word = layer["word"].weight[input_ids]
                q_pos = layer["pos"].weight[position_ids]
                self._q_cache["projection_embed"] = (q_word, q_pos)
                return

            if layer_type in ["projection_attn", "projection_linear_attn", "projection_fc1", "projection_fc2", "projection_linear_output"]:
                if proj_layers is not None and "q_proj" in proj_layers and proj_layers["q_proj"] is not None:
                    feature_dim = proj_layers["q_proj"].out_features
                elif layer is not None and hasattr(layer, "in_features"):
                    feature_dim = layer.in_features
                else:
                    raise ValueError(f"Cannot infer feature dimension for projection layer_type: {layer_type}")

                self._q_cache[layer_type] = torch.zeros(batch_size, seq_len, feature_dim, device=device)
                return

            raise ValueError(f"Unsupported projection layer_type in init_q: {layer_type}")

        if projection_layer_type is None or q is None or device is None:
            raise ValueError("init_q requires either projection initialization args or (projection_layer_type, q, device)")

        layer_mapping = {
            "projection_attn": "attn",
            "projection_linear_attn": "linear_attn",
            "projection_fc1": "fc1",
            "projection_fc2": "fc2",
            "projection_linear_output": "linear_output"
        }

        target_layer = layer_mapping.get(projection_layer_type)
        if target_layer is None:
            raise ValueError(f"No target mapping defined for projection layer: {projection_layer_type}")

        self._x_cache[target_layer] = q.detach().clone()

        input_dim = q.shape[-1]
        self.register_lateral(target_layer, input_dim)
        if target_layer in self.lateral_connections:
            self.lateral_connections[target_layer] = self.lateral_connections[target_layer].to(device)

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
    ):
        """
        Initialize cached activity `x` for the layer type.
        - embed: stores (x_word, x_pos) from embedding weights
        - other layers: Random initialization has been removed. Call `init_q` instead.
        """
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
        elif layer_type == "projection_embed":
            assert input_ids is not None and position_ids is not None, "Projection embedding layer requires input_ids and position_ids"
            vocab_size = layer["word"].weight.size(0)
            if input_ids.max() >= vocab_size:
                input_ids = torch.clamp(input_ids, max=vocab_size - 1)

            max_pos = layer["pos"].weight.size(0)
            if position_ids.max() >= max_pos:
                position_ids = torch.clamp(position_ids, max=max_pos - 1)

            q_word = layer["word"].weight[input_ids]
            q_pos = layer["pos"].weight[position_ids]
            self._q_cache["projection_embed"] = (q_word, q_pos)
        elif layer_type in ["projection_attn", "projection_linear_attn", "projection_fc1", "projection_fc2", "projection_linear_output"]:
            if proj_layers is not None and "q_proj" in proj_layers and proj_layers["q_proj"] is not None:
                feature_dim = proj_layers["q_proj"].out_features
            elif layer is not None and hasattr(layer, "in_features"):
                feature_dim = layer.in_features
            else:
                raise ValueError(f"Cannot infer feature dimension for projection layer_type: {layer_type}")

            self._q_cache[layer_type] = torch.zeros(batch_size, seq_len, feature_dim, device=device)
        elif layer_type in ["attn", "linear_attn", "fc1", "fc2", "linear_output"]:
            if proj_layers is not None and "q_proj" in proj_layers and proj_layers["q_proj"] is not None:
                feature_dim = proj_layers["q_proj"].in_features
            elif layer is not None and hasattr(layer, "in_features"):
                feature_dim = layer.in_features
            else:
                raise ValueError(f"Cannot infer feature dimension for layer_type: {layer_type}")

            self._x_cache[layer_type] = torch.zeros(batch_size, seq_len, feature_dim, device=device)
        else:
            raise ValueError(f"Unsupported layer_type in init_x: {layer_type}")

    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._x_cache.get(layer_type, None)
    def get_q(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._q_cache.get(layer_type, None)
        
    
    def get_mu(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._mu_cache.get(layer_type, None)
    
    def get_td_err(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._error_cache.get(layer_type, None)
    def get_td_err_projection(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._error_cache_projection.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        return float(self._energy)

    def clear_energy(self):
        self._energy = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        self._error_cache.clear()
        self._q_cache.clear()
        self._mu_cache_projection.clear()
        self._error_cache_projection.clear()
        if hasattr(self, "_last_kv_cache"):
            self._last_kv_cache = None
        if hasattr(self, "_last_kv_cache_projection"):
            self._last_kv_cache_projection = None
        
    def get_errors(self) -> list:
        return self._errors

    def clear_errors(self):
        self._errors = []
        self._error_cache.clear()
        self._error_cache_projection.clear()
        
    def set_learning_rate(self, lr: float):
        self.local_lr = float(lr)
        for lateral in self.lateral_connections.values():
            lateral.set_learning_rate(lr)
        
    def get_learning_rate(self) -> float:
        return float(self.local_lr)
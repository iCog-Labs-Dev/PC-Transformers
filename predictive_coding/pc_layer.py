import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    x_init,
    step_embed,
    step_linear,
    step_x_query,
    step_x_key,
    step_x_value,
    step_x_score,
    step_x_A,
    finalize_step,
)
from utils.optim.optim_utils import PCOptimizer
from predictive_coding.lateral_connc import LateralConnections

class PCLayer(nn.Module):
    """
    Predictive Coding Layer wrapper that manages iterative inference state and
    delegates computation to helper functions (step_embed, decomposed attention steps, step_linear).
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
        self._last_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
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
                optimizer=self.optimizer,
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

        elif layer_type in {"x_query", "x_key", "x_value"}:
            lateral_conn = self.lateral_connections.get(layer_type, None)
            if layer is None:
                raise ValueError(f"layer must be provided for layer_type={layer_type}")

            if layer_type == "x_query":
                x, mu, bu_err = step_x_query(
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
                    layer_norm=layer_norm,
                    optimizer=self.optimizer,
                )
            elif layer_type == "x_key":
                x, mu, bu_err = step_x_key(
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
                    layer_norm=layer_norm,
                    optimizer=self.optimizer,
                )
            else:
                x, mu, bu_err = step_x_value(
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
                    layer_norm=layer_norm,
                    optimizer=self.optimizer,
                )

            # KV cache support: treat mu of x_key/x_value as K/V
            if use_cache and layer_type in {"x_key", "x_value"}:
                cached_k, cached_v = kv_cache if kv_cache is not None else (None, None)
                # If we already updated partial cache this step, prefer it
                if self._last_kv_cache is not None:
                    cached_k = self._last_kv_cache[0] if self._last_kv_cache[0] is not None else cached_k
                    cached_v = self._last_kv_cache[1] if self._last_kv_cache[1] is not None else cached_v

                if layer_type == "x_key":
                    new_k = mu.detach()
                    k_total = torch.cat([cached_k, new_k], dim=1) if cached_k is not None else new_k
                    self._last_kv_cache = (k_total, cached_v)
                else:
                    new_v = mu.detach()
                    v_total = torch.cat([cached_v, new_v], dim=1) if cached_v is not None else new_v
                    self._last_kv_cache = (cached_k, v_total)

        elif layer_type == "x_score":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            if layer is None:
                raise ValueError("x_score requires a layer (weights/bias)")
            x, mu, bu_err = step_x_score(
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
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )

        elif layer_type == "x_A":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x_score = self._x_cache.get("x_score", None)
            if layer is None:
                raise ValueError("x_A requires a layer (weights/bias)")
            if x_score is None:
                raise ValueError("x_A requires cached x from x_score")
            x, mu, bu_err = step_x_A(
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
                x_score=x_score,
                td_err=td_err,
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
        
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
                layer_norm=layer_norm,
                optimizer=self.optimizer,
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
    ):
        """
        Initialize cached activity `x` for the layer type.
        - embed: stores (x_word, x_pos) from embedding weights
        - x_query/x_key/x_value/x_score/x_A: random init shaped (B, S, n_embed)
        - linear/others: random init sized to layer input dimension
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

        elif layer_type in {"x_query", "x_key", "x_value", "x_score", "x_A"}:
            if layer is not None:
                input_dim = layer.weight.shape[1]
            elif self.n_embed is not None:
                input_dim = int(self.n_embed)
            elif proj_layers is not None and "q_proj" in proj_layers:
                input_dim = proj_layers["q_proj"].weight.shape[1]
            else:
                raise ValueError("Attention sub-layer init requires layer or n_embed")

            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            self.register_lateral(layer_type, input_dim)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            
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
        for lateral in self.lateral_connections.values():
            lateral.set_learning_rate(lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)

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
    step_linear_attn,
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
        self._energy_by_layer_type: Dict[str, float] = {}
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
            self._energy_by_layer_type[layer_type] = self._energy_by_layer_type.get(layer_type, 0.0) + float(energy)
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
            if layer is None:
                raise ValueError("x_score requires a non-None layer")
            # x_score is derived from cached mu_Q and mu_K (and optional KV cache).
            mu_q = self._mu_cache.get("x_query", None)
            mu_k = self._mu_cache.get("x_key", None)
            mu_v = self._mu_cache.get("x_value", None)
            if mu_q is None or mu_k is None or mu_v is None:
                raise ValueError("x_score requires cached mu for x_query/x_key/x_value")

            # Prefer the in-step assembled cache if present.
            kv_for_score = None
            if use_cache:
                if self._last_kv_cache is not None and (self._last_kv_cache[0] is not None or self._last_kv_cache[1] is not None):
                    kv_for_score = self._last_kv_cache
                else:
                    kv_for_score = kv_cache

            x, mu, bu_err, new_kv_cache = step_x_score(
                t,
                T,
                target_activity,
                x,
                layer=layer,
                lateral_conn=None,
                layer_type=layer_type,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                energy_fn_name=self.energy_fn_name,
                update_bias=self.update_bias,
                requires_update=requires_update,
                mu_q=mu_q,
                mu_k=mu_k,
                mu_v=mu_v,
                num_heads=int(self.num_heads) if self.num_heads is not None else 1,
                use_cache=use_cache,
                kv_cache=kv_for_score,
                td_err=td_err,
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )

            if use_cache and new_kv_cache is not None:
                self._last_kv_cache = new_kv_cache

        elif layer_type == "x_A":
            if layer is None:
                raise ValueError("x_A requires a non-None layer")
            x_score = self._x_cache.get("x_score", None)
            if x_score is None:
                raise ValueError("x_A requires cached x from x_score")
            x, mu, bu_err = step_x_A(
                t=t,
                T=T,
                target=target_activity,
                x=x,
                layer=layer,
                lateral_conn=None,
                layer_type=layer_type,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                energy_fn_name=self.energy_fn_name,
                update_bias=self.update_bias,
                requires_update=requires_update,
                x_score=x_score,
                td_err=td_err,
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
        
        else:
            if layer_type == "linear_attn":
                # Compute attention output = A @ V using mu from upstream layer (pc_qkv).
                if proj_layers is None:
                    raise ValueError("linear_attn requires proj_layers with mu_A and mu_V")
                mu_A = proj_layers.get("mu_A", None)
                mu_V = proj_layers.get("mu_V", None)
                num_heads = int(proj_layers.get("num_heads", self.num_heads or 1))
                if mu_A is None or mu_V is None:
                    raise ValueError("proj_layers must include mu_A and mu_V for linear_attn")

                x, mu, bu_err = step_linear_attn(
                    t=t,
                    T=T,
                    target=target_activity,
                    x=x,
                    mu_A=mu_A,
                    mu_V=mu_V,
                    num_heads=num_heads,
                    local_lr=self.local_lr,
                    clamp_value=self.clamp_value,
                    energy_fn_name=self.energy_fn_name,
                    td_err=td_err,
                    residual=proj_layers.get("residual", None),
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
                    residual=(proj_layers.get("residual", None) if (proj_layers is not None and layer_type == "fc2") else None),
                    optimizer=self.optimizer,
                )
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None:
            self._error_cache[layer_type] = bu_err.detach().clone()
        
        if layer_type in {"x_score", "x_A"} and bu_err is not None:
            # bu_err is computed against the internally resized target used by the step.
            error = bu_err
            target_for_energy = mu + bu_err
            energy, step_errors = finalize_step(mu, target_for_energy, error, t, layer_type, self.energy_fn_name)
        else:
            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._energy += energy
        self._energy_by_layer_type[layer_type] = self._energy_by_layer_type.get(layer_type, 0.0) + float(energy)
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

        elif layer_type in {"x_query", "x_key", "x_value"}:
            # Standard latent tensors shaped (B, S, n_embed)
            if layer is not None:
                input_dim = layer.weight.shape[1]
            elif self.n_embed is not None:
                input_dim = int(self.n_embed)
            else:
                raise ValueError("Attention sub-layer init requires layer or n_embed")

            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            self.register_lateral(layer_type, input_dim)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)

        elif layer_type == "x_score":
            # Attention scores tensor shaped (B, nh, S, S)
            if self.num_heads is None:
                raise ValueError("x_score init requires num_heads")
            nh = int(self.num_heads)
            self._x_cache[layer_type] = torch.zeros(batch_size, nh, seq_len, seq_len, device=device)

        elif layer_type == "x_A":
            # Attention logits tensor shaped (B, nh, S, S), init so softmax is causal-uniform.
            if self.num_heads is None:
                raise ValueError("x_A init requires num_heads")
            nh = int(self.num_heads)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            logits = torch.zeros(seq_len, seq_len, device=device)
            logits = logits.masked_fill(~causal_mask, -1e4)
            self._x_cache[layer_type] = logits.unsqueeze(0).unsqueeze(0).expand(batch_size, nh, seq_len, seq_len).contiguous()
        
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

    def get_energy_breakdown(self) -> Dict[str, float]:
        """Get accumulated energy per latent state (keyed by layer_type)."""
        return dict(self._energy_by_layer_type)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._energy_by_layer_type.clear()
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

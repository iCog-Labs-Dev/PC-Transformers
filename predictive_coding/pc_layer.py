import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
import torch.nn.functional as F

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
        max_steps: int,
        lr: float,
        update_bias: bool,
        energy_fn_name: str,
        num_heads: Optional[int] = None,
        n_embed: Optional[int] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
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
        self._last_step_energy = 0.0
        self._errors = []
        self._converged = False
        self._step_energies = []
        self._plateau_count = 0
        self._deferred_update: Optional[Dict[str, Any]] = None
        self._final_update_applied = False
        self.layer_type: Optional[str] = None
        self._last_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def check_convergence(self, t: int, min_steps: int, threshold: float, healthy_threshold: float = 0.0) -> bool:
        if self._converged:
            return True

        if (t + 1) < min_steps:
            return False
            
        if len(self._step_energies) >= 1:
            E_t = self._step_energies[-1]
            
            # 1. Absolute "Healthy" check
            if healthy_threshold > 0 and E_t < healthy_threshold:
                self._converged = True
                return True

            # 2. Relative "Plateau" check
            if len(self._step_energies) >= 2:
                E_prev = self._step_energies[-2]
                relative_change = abs(E_t - E_prev) / (abs(E_prev) + 1e-8)
                
                if relative_change < threshold:
                    self._plateau_count += 1
                else:
                    self._plateau_count = 0
                    
                if self._plateau_count >= 2:
                    self._converged = True
                
        return self._converged

    def apply_deferred_update(self) -> bool:
        """Apply the cached deferred update exactly once for the layer's final step."""
        if self._final_update_applied or self._deferred_update is None:
            return False

        update = self._deferred_update
        kind = update["kind"]

        with torch.no_grad():
            if kind == "embed":
                word_layer = update["layer"]["word"]
                pos_layer = update["layer"]["pos"]
                flat_input_ids = update["input_ids"].reshape(-1)
                flat_position_ids = update["position_ids"].reshape(-1)
                flat_update = update["error"].reshape(-1, update["error"].size(-1))

                delta = self.local_lr * flat_update
                delta = torch.clamp(delta, -0.01, 0.01)

                word_layer.weight.data.index_add_(0, flat_input_ids, delta)
                pos_layer.weight.data.index_add_(0, flat_position_ids, delta)

            elif kind == "linear":
                lateral_conn = update["lateral_conn"]
                if lateral_conn is not None and update["lateral_x"] is not None:
                    lateral_conn.update_weights(update["lateral_x"])

                layer = update["layer"]
                bu_err = update["bu_err"]
                x_input = update["x_input"]

                delta_W = self.local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x_input)
                delta_W = torch.clamp(delta_W, -0.01, 0.01)
                layer.weight.data.add_(delta_W)

                if layer.bias is not None and update["update_bias"]:
                    delta_b = self.local_lr * bu_err.mean(dim=(0, 1))
                    delta_b = torch.clamp(delta_b, -0.01, 0.01)
                    layer.bias.data.add_(delta_b)

            elif kind == "attn":
                lateral_conn = update["lateral_conn"]
                if lateral_conn is not None and update["lateral_x"] is not None:
                    lateral_conn.update_weights(update["lateral_x"])

                proj_layers = update["proj_layers"]
                q_proj = proj_layers["q_proj"]
                k_proj = proj_layers["k_proj"]
                v_proj = proj_layers["v_proj"]
                q = update["q"]
                k = update["k"]
                v = update["v"]
                x_norm = update["x_norm"]

                B = q.size(0)
                S = q.size(2)
                num_heads = q.size(1)
                head_dim = q.size(-1)

                for h in range(num_heads):
                    q_slice = q[:, h, :, :]
                    k_slice = k[:, h, :, :]
                    v_slice = v[:, h, :, :]

                    dW_q_h = torch.einsum("bsd,bse->de", q_slice, x_norm) / (B * S)
                    dW_k_h = torch.einsum("bsd,bse->de", k_slice, x_norm) / (B * S)
                    dW_v_h = torch.einsum("bsd,bse->de", v_slice, x_norm) / (B * S)

                    start = h * head_dim
                    end = (h + 1) * head_dim

                    q_proj.weight.data[start:end, :] += torch.clamp(
                        self.local_lr * dW_q_h, -self.clamp_value, self.clamp_value
                    )
                    k_proj.weight.data[start:end, :] += torch.clamp(
                        self.local_lr * dW_k_h, -self.clamp_value, self.clamp_value
                    )
                    v_proj.weight.data[start:end, :] += torch.clamp(
                        self.local_lr * dW_v_h, -self.clamp_value, self.clamp_value
                    )

                    if update["update_bias"]:
                        if q_proj.bias is not None:
                            delta_b_q = q_slice.mean(dim=(0, 1)) / (B * S)
                            q_proj.bias.data[start:end] += torch.clamp(
                                self.local_lr * delta_b_q, -self.clamp_value, self.clamp_value
                            )
                        if k_proj.bias is not None:
                            delta_b_k = k_slice.mean(dim=(0, 1)) / (B * S)
                            k_proj.bias.data[start:end] += torch.clamp(
                                self.local_lr * delta_b_k, -self.clamp_value, self.clamp_value
                            )
                        if v_proj.bias is not None:
                            delta_b_v = v_slice.mean(dim=(0, 1)) / (B * S)
                            v_proj.bias.data[start:end] += torch.clamp(
                                self.local_lr * delta_b_v, -self.clamp_value, self.clamp_value
                            )

        self._final_update_applied = True
        self._deferred_update = None
        return True
    
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False, 
    ):
        """Perform one predictive coding inference step."""
        self.layer_type = layer_type

        if getattr(self, "_converged", False):
            if layer_type == "embed":
                mu = self._mu_cache["embed"]
                error = target_activity - mu
                self._error_cache["embed"] = error.detach().clone()
                return self._x_cache["embed"]
            else:
                mu = self._mu_cache[layer_type]
                if layer_type == "linear_output":
                    bu_err = target_activity - F.softmax(mu, dim=-1) 
                else:    
                    bu_err = target_activity - mu
                self._error_cache[layer_type] = bu_err.detach().clone()
                return self._x_cache[layer_type], mu

        self._reset_step_state()
        x = self._get_cached_state(layer_type)

        if layer_type == "embed":
            mu, mu_word, mu_pos, bu_err, update_state = step_embed(
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
            if update_state is not None:
                self._deferred_update = update_state

            # compute energy
            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
            self._last_step_energy = energy
            self._energy += energy
            self._step_energies.append(energy)
            self._errors.extend(step_errors)
            return mu_word, mu_pos
        
        elif layer_type == "attn":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err, new_kv_cache, update_state = step_attn(
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
            if update_state is not None:
                self._deferred_update = update_state
        
        else:
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err, update_state = step_linear(
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
            if update_state is not None:
                self._deferred_update = update_state
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None: 
            self._error_cache[layer_type] = bu_err.detach().clone()
        
        error = target_activity - mu
        energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._last_step_energy = energy
        self._energy += energy
        self._step_energies.append(energy)
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
        - attn: creates random initialization shaped (B, S, H_out)
        - linear/others: random init sized to layer input dimension
        """
        self.layer_type = layer_type

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
        """Get the cumulative energy accumulated across executed steps."""
        return float(self._energy)

    def get_last_step_energy(self) -> Optional[float]:
        """Get the most recent per-step energy for the layer."""
        return float(self._last_step_energy)

    def get_step_count(self) -> int:
        """Get the number of actual inference steps executed by the layer."""
        return len(self._step_energies)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._last_step_energy = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        self._error_cache.clear()
        self._converged = False
        self._step_energies.clear()
        self._plateau_count = 0
        self._deferred_update = None
        self._final_update_applied = False
        self._last_kv_cache = None
        
    def get_errors(self) -> list:
        """Get the list of error values accumulated during inference."""
        return self._errors

    def clear_errors(self):
        """Clear the stored errors for the layer."""
        self._errors = []
        
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)
        for lateral_conn in self.lateral_connections.values():
            lateral_conn.set_learning_rate(lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)

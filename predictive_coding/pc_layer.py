import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    step_q_embed,
    step_q_attn,
    step_q_linear,
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
        optimizer_name: str = "sgd",
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
        self._q_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy = 0.0
        self._errors = []

    def _q_key_for_x_layer(self, layer_type: str) -> str:
        if layer_type == "embed":
            return "q_embed"
        return f"q_{layer_type}"
    
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
                optimizer=self.optimizer,
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
        - attn: creates random initialization shaped (B, S, H_out)
        - linear/others: random init sized to layer input dimension
        """
        if layer_type == "embed":
            if "q_embed" not in self._q_cache:
                raise RuntimeError("q_embed is missing. Call init_q('q_embed', ...) before init_x('embed', ...)")
            self._x_cache["embed"] = self._q_cache["q_embed"]
            return
            
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            if "q_attn" not in self._q_cache:
                raise RuntimeError("q_attn is missing. Call init_q('q_attn', ...) before init_x('attn', ...)")
            self._x_cache["attn"] = self._q_cache["q_attn"].detach().to(device)
            
            self.register_lateral(layer_type, H_in)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            q_key = self._q_key_for_x_layer(layer_type)
            if q_key not in self._q_cache:
                raise RuntimeError(f"{q_key} is missing. Call init_q('{q_key}', ...) before init_x('{layer_type}', ...)")
            self._x_cache[layer_type] = self._q_cache[q_key].detach().to(device)
            
            input_dim = layer.weight.shape[1]
            self.register_lateral(layer_type, input_dim)  
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 

    def init_q(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        source_tensor: Optional[torch.Tensor] = None,
        target_activity: Optional[torch.Tensor] = None,
        td_err: Optional[torch.Tensor] = None,
        layer_norm: Optional[nn.Module] = None,
        flash: bool = False,
        requires_update: bool = False,
    ):
        """Initialize projection latent q-cache for each layer type (optional parameter updates)."""
        _ = batch_size
        _ = seq_len

        if layer_type == "q_embed":
            assert input_ids is not None and position_ids is not None, "q_embed requires input_ids and position_ids"
            assert isinstance(layer, dict), "q_embed requires embedding layer dict"
            q_embed, q_word, q_pos = step_q_embed(layer=layer, input_ids=input_ids, position_ids=position_ids)
            self._q_cache["q_embed"] = (q_word.detach().to(device), q_pos.detach().to(device))
            if requires_update:
                target = target_activity if target_activity is not None else q_embed.detach()
                error = target - q_embed
                if td_err is not None:
                    error = error - td_err
                with torch.no_grad():
                    flat_input_ids = input_ids.reshape(-1)
                    flat_update = error.reshape(-1, error.size(-1))
                    flat_position_ids = position_ids.reshape(-1)

                    word_layer: nn.Embedding = layer["word"]
                    pos_layer: nn.Embedding = layer["pos"]

                    update_word = torch.zeros_like(word_layer.weight)
                    update_pos = torch.zeros_like(pos_layer.weight)
                    update_word.index_add_(0, flat_input_ids, flat_update)
                    update_pos.index_add_(0, flat_position_ids, flat_update)

                    self.optimizer.step_param(word_layer.weight, update_word, self.local_lr, clamp_value=0.01)
                    self.optimizer.step_param(pos_layer.weight, update_pos, self.local_lr, clamp_value=0.01)

                q_embed, q_word, q_pos = step_q_embed(layer=layer, input_ids=input_ids, position_ids=position_ids)
                self._q_cache["q_embed"] = (q_word.detach().to(device), q_pos.detach().to(device))
            return q_embed.detach().to(device)

        if layer_type == "q_attn":
            assert source_tensor is not None, "q_attn requires source_tensor"
            assert proj_layers is not None, "q_attn requires proj_layers"
            x_norm = layer_norm(source_tensor) if layer_norm is not None else source_tensor
            q_attn, q_out = step_q_attn(
                x=source_tensor,
                proj_layers=proj_layers,
                num_heads=self.num_heads,
                n_embed=self.n_embed,
                layer_norm=layer_norm,
                flash=flash,
            )
            self._q_cache["q_attn"] = q_attn.detach().to(device)
            if requires_update:
                target = target_activity if target_activity is not None else source_tensor
                bu_err = target - q_out
                if td_err is not None:
                    bu_err = bu_err - td_err
                B, S = bu_err.shape[:2]
                scale = max(B * S, 1)

                q_proj = proj_layers["q_proj"]
                k_proj = proj_layers["k_proj"]
                v_proj = proj_layers["v_proj"]

                update_q = torch.einsum("bsv,bse->ve", bu_err, x_norm.detach()) / scale
                update_k = torch.einsum("bsv,bse->ve", bu_err, x_norm.detach()) / scale
                update_v = torch.einsum("bsv,bse->ve", bu_err, x_norm.detach()) / scale

                self.optimizer.step_param(q_proj.weight, update_q, self.local_lr, clamp_value=0.01)
                self.optimizer.step_param(k_proj.weight, update_k, self.local_lr, clamp_value=0.01)
                self.optimizer.step_param(v_proj.weight, update_v, self.local_lr, clamp_value=0.01)

                if self.update_bias:
                    update_b_q = bu_err.mean(dim=(0, 1)) if q_proj.bias is not None else None
                    update_b_k = bu_err.mean(dim=(0, 1)) if k_proj.bias is not None else None
                    update_b_v = bu_err.mean(dim=(0, 1)) if v_proj.bias is not None else None

                    if update_b_q is not None:
                        self.optimizer.step_param(q_proj.bias, update_b_q, self.local_lr, clamp_value=0.01)
                    if update_b_k is not None:
                        self.optimizer.step_param(k_proj.bias, update_b_k, self.local_lr, clamp_value=0.01)
                    if update_b_v is not None:
                        self.optimizer.step_param(v_proj.bias, update_b_v, self.local_lr, clamp_value=0.01)

                q_attn, q_out = step_q_attn(
                    x=source_tensor,
                    proj_layers=proj_layers,
                    num_heads=self.num_heads,
                    n_embed=self.n_embed,
                    layer_norm=layer_norm,
                    flash=flash,
                )
                self._q_cache["q_attn"] = q_attn.detach().to(device)
            return q_out.detach().to(device)

        assert source_tensor is not None, f"{layer_type} requires source_tensor"
        assert layer is not None, f"{layer_type} requires layer"
        base_layer_type = layer_type.replace("q_", "")
        q_latent, q_out = step_q_linear(
            x=source_tensor,
            layer=layer,
            layer_type=base_layer_type,
            layer_norm=layer_norm,
        )
        self._q_cache[layer_type] = q_latent.detach().to(device)
        if requires_update:
            if layer_norm is not None and base_layer_type == "fc1":
                x_input = layer_norm(source_tensor)
            elif base_layer_type == "fc2":
                x_input = F.gelu(source_tensor)
            else:
                x_input = source_tensor

            target = target_activity if target_activity is not None else source_tensor
            bu_err = target - q_out
            if td_err is not None:
                bu_err = bu_err - td_err

            B, S = bu_err.shape[:2]
            scale = max(B * S, 1)
            update_w = torch.einsum("bsv,bsh->vh", bu_err, x_input.detach()) / scale
            self.optimizer.step_param(layer.weight, update_w, self.local_lr, clamp_value=0.01)

            if self.update_bias and layer.bias is not None:
                update_b = bu_err.mean(dim=(0, 1))
                self.optimizer.step_param(layer.bias, update_b, self.local_lr, clamp_value=0.01)

            q_latent, q_out = step_q_linear(
                x=source_tensor,
                layer=layer,
                layer_type=base_layer_type,
                layer_norm=layer_norm,
            )
            self._q_cache[layer_type] = q_latent.detach().to(device)
        return q_out.detach().to(device)
    
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached activity tensor for a given layer type."""
        return self._x_cache.get(layer_type, None)

    def get_q(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached projection latent tensor for a given q layer type."""
        return self._q_cache.get(layer_type, None)
    
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
        self._q_cache.clear()
        
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

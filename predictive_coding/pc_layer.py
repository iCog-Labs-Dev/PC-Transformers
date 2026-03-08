import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    step_embed,
    step_linear,
    step_attn,
    finalize_step,
)
from predictive_coding.lateral_connc import LateralConnections
from predictive_coding.hopfield_memory import HopfieldMemory

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
        use_memory_init: bool = True,
        memory_slots: int = 128,
        memory_delta: float = 8.0,
        memory_update_qk: bool = True,
    ):
        super().__init__()
        self.T = T
        self.local_lr = lr
        self.update_bias = update_bias
        self.clamp_value = 3.0
        self.energy_fn_name = energy_fn_name 
        self.num_heads = num_heads
        self.n_embed = n_embed
        self.use_memory_init = use_memory_init
        self.memory_slots = int(memory_slots)
        self.memory_delta = float(memory_delta)
        self.memory_update_qk = bool(memory_update_qk)
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        self.memories: Dict[str, HopfieldMemory] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._obs_cache: Dict[str, torch.Tensor] = {}
        self._energy = 0.0
        self._memory_loss = 0.0
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

    def register_memory(self, layer_type: str, obs_dim: int, hidden_dim: int, device: torch.device):
        if layer_type not in self.memories:
            memory = HopfieldMemory(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                memory_slots=self.memory_slots,
                delta=self.memory_delta,
            )
            self.memories[layer_type] = memory
            self.add_module(f"memory_{layer_type}", memory)
        self.memories[layer_type] = self.memories[layer_type].to(device)
    
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
        observation: Optional[torch.Tensor] = None,
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

        # Update associative memory only after final inference step.
        if (
            self.use_memory_init
            and requires_update
            and t == T - 1
            and layer_type in self.memories
            and layer_type in self._obs_cache
        ):
            lh = self.memories[layer_type].local_update(
                observation=self._obs_cache[layer_type],
                converged_state=x.detach(),
                lr=self.local_lr,
                clamp_value=self.clamp_value,
                update_qk=self.memory_update_qk,
            )
            self._memory_loss += float(lh)

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
        observation: Optional[torch.Tensor] = None,
    ):
        """
        Initialize cached activity `x` for the layer type.
        - embed: stores (x_word, x_pos) from embedding weights
        - attn: creates random initialization shaped (B, S, H_out)
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
            
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            if self.use_memory_init and observation is not None:
                self.register_memory("attn", observation.size(-1), H_out, device)
                if not self.memories["attn"]._bootstrapped:
                    # In decoder-only settings, Ifw is typically unavailable; use current state as warm-start target.
                    warm_start = torch.zeros(batch_size, seq_len, H_out, device=device)
                    self.memories["attn"].bootstrap_value_matrix(observation, warm_start)
                retrieved, _, _, _ = self.memories["attn"].retrieve(observation)
                self._x_cache["attn"] = retrieved.detach().clone()
                self._obs_cache["attn"] = observation.detach().clone()
            else:
                self._x_cache["attn"] = torch.zeros(batch_size, seq_len, H_out, device=device)
            
            self.register_lateral(layer_type, H_in)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            if self.use_memory_init and observation is not None:
                self.register_memory(layer_type, observation.size(-1), input_dim, device)
                if not self.memories[layer_type]._bootstrapped:
                    # If Ifw is unavailable, initialize V to match a neutral baseline on first minibatch.
                    warm_start = torch.zeros(batch_size, seq_len, input_dim, device=device)
                    self.memories[layer_type].bootstrap_value_matrix(observation, warm_start)
                retrieved, _, _, _ = self.memories[layer_type].retrieve(observation)
                self._x_cache[layer_type] = retrieved.detach().clone()
                self._obs_cache[layer_type] = observation.detach().clone()
            else:
                self._x_cache[layer_type] = torch.zeros(batch_size, seq_len, input_dim, device=device)
            
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

    def get_memory_loss(self) -> float:
        """Get accumulated memory reconstruction loss LH for this layer."""
        return float(self._memory_loss)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._memory_loss = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        self._obs_cache.clear()
        
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

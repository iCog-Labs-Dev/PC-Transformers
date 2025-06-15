import torch
import torch.nn as nn
from typing import Optional
from utils.pc_utils import x_init, step_embed, step_linear, step_attn, finalize_step


class PCLayer(nn.Module):
    def __init__(
        self,
        T: int = 1,
        local_learning_rate: float = 1e-3,
        is_holding_error: bool = False,
        update_bias: bool = True,
        energy_fn_name: str = "scaled_mse",
    ):
        super().__init__()
        self.T = T
        self.local_lr = local_learning_rate
        self.is_holding_error = is_holding_error
        self.update_bias = update_bias
        self.clamp_value = 1.0
        self.W_latents = nn.ParameterDict()
        self.use_lateral = True
        self._x_cache = {}
        self.energy_fn_name = energy_fn_name 
        self._energy = None
        self._errors = []

    def register_lateral(self, layer_type: str, size: int):
        if layer_type not in self.W_latents:
            W = torch.empty(size, size)
            nn.init.xavier_uniform_(W)
            self.W_latents[layer_type] = nn.Parameter(W)

    def forward(
        self,
        target_activity: torch.Tensor,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        layer_type: str = "fc1",
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        t=0,
        T=1,
        requires_update: bool = True,
    ):
        B, S, _ = target_activity.shape    
        x = None
        self._energy = 0.0
        self._errors = []

        if layer_type == "embed":
            if "embed" not in self._x_cache:
                raise ValueError("Embedding state not initialized. Call init_x first.")
            x_word, x_pos = self._x_cache["embed"]
        else:
            if layer_type not in self._x_cache:
                raise ValueError(f"{layer_type} state not initialized. Call init_x first.")
            x = self._x_cache[layer_type]

        if layer_type == "embed":
            mu = step_embed(t, T, target_activity, layer, layer_type, input_ids, position_ids,
                            self.local_lr, self.clamp_value, self.energy_fn_name, self.is_holding_error,
                            requires_update)
        elif layer_type == "attn":
            x, mu = step_attn(t, T, target_activity, x, self.W_latents, proj_layers, layer_type,
                              self.local_lr, self.clamp_value, self.use_lateral, self.is_holding_error,
                              self.energy_fn_name, self.update_bias, requires_update)
        else:
            x, mu = step_linear(t, T, target_activity, x, layer, self.W_latents, layer_type,
                               self.local_lr, self.clamp_value, self.use_lateral, self.is_holding_error,
                               self.energy_fn_name, self.update_bias, requires_update)

        if self.is_holding_error:
            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type,
                                                self.energy_fn_name, self.is_holding_error)
            self._energy += energy
            self._errors.extend(step_errors)

        if layer_type == "embed":
            # Cache updated x for next step inference
            self._x_cache["embed"] = (x_word, x_pos)
            return x_word, x_pos
        else:
            self._x_cache[layer_type] = x
            return x

    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        layer_type: str = "linear",
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Initialize the layer's state variables and store them in x_cache."""
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None, "Embedding layer requires input_ids and position_ids"
            x_word = layer["word"].weight[input_ids] 
            x_pos = layer["pos"].weight[position_ids] 
            self._x_cache["embed"] = (x_word, x_pos)
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            self._x_cache["attn"] = x_init(batch_size, seq_len, H_out)
            
            if self.use_lateral:
                self.register_lateral(layer_type, H_in)
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim)
            H_in = layer.weight.shape[1]

            if self.use_lateral:
                self.register_lateral(layer_type, H_in)

    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._x_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        return self._energy

    def clear_energy(self):
        self._energy = 0.0
        self._x_cache.clear()

    def get_errors(self) -> list:
        return self._errors

    def clear_errors(self):
        self._errors = []

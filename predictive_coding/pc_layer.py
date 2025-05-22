import torch
import torch.nn as nn
from typing import Optional
from .pc_utils import (x_init, step_embed, step_linear, step_attn)

class PCLayer(nn.Module):
    def __init__(
        self,
        T: int = 10,
        local_learning_rate: float = 1e-3,
        is_holding_error: bool = False,
        update_bias: bool = True
    ):
        super().__init__()
        self.T = T
        self.local_lr = local_learning_rate
        self.is_holding_error = is_holding_error
        self.update_bias = update_bias
        self.clamp_value = 1.0
        self._x_cache = {}
        self._W_cache = {}

    def forward(
        self,
        target_activity: torch.Tensor,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        layer_type: str = "fc1"
    ):
        B, S, H_out = target_activity.shape
        x, W, bias = None, None, None

        if layer_type == "embed":
            W_word, W_pos = layer["word"].weight.shape[0], layer["pos"].weight.shape[0]
            x_word = x_init(B, S, W_word)
            x_pos = x_init(1, S, W_pos)

        elif layer_type == "attn":
            x = x_init(B, S, H_out)
        else:
            x = x_init((B, S, layer.weight.shape[1]))
        
        for t in range(self.T):
            if layer_type == "embed":
                x_word, x_pos = step_embed(t, target_activity, x_word, x_pos, layer, self.local_lr, self.clamp_value, self.T, self.is_holding_error)
            elif layer_type == "attn":
                x = step_attn(t, target_activity, x, proj_layers, layer_type, self.local_lr, self.clamp_value, self.T, self.is_holding_error, self.update_bias)
            else:
                x = step_linear(t, target_activity, x, layer, layer_type, self.local_lr, self.clamp_value, self.T, self.is_holding_error, self.update_bias)

        if layer_type == "embed":
            self._cache("embed", (x_word, x_pos), None)
            return x_word, x_pos
        else:
            self._cache(layer_type, x, layer.weight if hasattr(layer, "weight") else None)
            return x
        
    def _cache(self, layer_type, x, layer_weight):
        if layer_type == "embed":
            x_word, x_pos = x
            self._x_cache["word"] = x_word.detach()
            self._x_cache["pos"] = x_pos.detach()
            if layer_weight is not None:
                self._W_cache["word"] = layer_weight["word"].data.clone()
                self._W_cache["pos"] = layer_weight["pos"].data.clone()
        else:
            self._x_cache[layer_type] = x.detach()
            if layer_weight is not None:
                self._W_cache[layer_type] = layer_weight.data.clone()
    
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._x_cache.get(layer_type, None)

    def get_weights(self, layer_type: str) -> Optional[torch.Tensor]:
        return self._W_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        return self._energy

    def clear_energy(self):
        self._energy = None
        self._x_cache.clear()
        self._W_cache.clear()

    def get_errors(self) -> list:
        return self._errors

    def clear_errors(self):
        self._errors = []
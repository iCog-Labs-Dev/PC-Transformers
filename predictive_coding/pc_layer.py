import torch
import torch.nn as nn
from typing import Callable, Optional

class PCLayer(nn.Module):
    def __init__(
        self,
        T: int = 10,
        local_learning_rate: float = 1e-3,
        energy_fn: Optional[Callable] = None,
        x_init: Optional[Callable] = None,
        is_holding_error: bool = False,
        update_bias: bool = True
    ):
        super().__init__()
        self.T = T
        self.local_lr = local_learning_rate
        self.energy_fn = energy_fn
        self.x_init = x_init
        self.is_holding_error = is_holding_error
        self.update_bias = update_bias
        self.clamp_value = 1.0

        self._energy = None
        self._errors = []
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
            x_word = self.x_init(B, S, W_word)
            x_pos = self.x_init(1, S, W_pos)

        elif layer_type == "attn":
            x = self.x_init(B, S, H_out)
        else:
            x = self.x_init((B, S, layer.weight.shape[1]))
        
        for t in range(self.T):
            if layer_type == "embed":
                x_word, x_pos = self._step_embed(t, target_activity, x_word, x_pos, layer)
            elif layer_type == "attn":
                x = self._step_attn(t, target_activity, x, proj_layers, layer_type)
            else:
                x = self._step_linear(t, target_activity, x, layer, layer_type)

        if layer_type == "embed":
            self._cache("embed", (x_word, x_pos), None)
            return x_word, x_pos
        else:
            self._cache(layer_type, x, layer.weight if hasattr(layer, "weight") else None)
            return x
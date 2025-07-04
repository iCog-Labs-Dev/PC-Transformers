import torch
import torch.nn as nn
from typing import Optional
from utils.pc_utils import x_init, step_embed, step_linear, step_attn, finalize_step

"""
predictive_coding.pc_layer

This module implements the PCLayer class, which provides predictive coding inference and local learning for neural network layers.
It supports embedding, attention, and linear layers, and manages iterative inference, error computation, and lateral connections.
"""

class PCLayer(nn.Module):
    """
    Predictive Coding Layer for neural network modules.

    Supports iterative inference, local learning, and optional lateral (recurrent) connections.
    Can be used for embedding, attention, or linear layers.
    """
    def __init__(
        self,
        T: int = 1,
        local_learning_rate: float = 1e-3,
        is_holding_error: bool = False,
        update_bias: bool = True,
        energy_fn_name: str = "scaled_mse",
    ):
        """
        Initialize the PCLayer.

        Args:
            T (int): Number of inference steps.
            local_learning_rate (float): Learning rate for local/lateral updates.
            is_holding_error (bool): Whether to accumulate and store errors.
            update_bias (bool): Whether to update bias terms during learning.
            energy_fn_name (str): Name of the energy function to use for error computation.
        """
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
        self._energy = 0.0
        self._errors = []

    def register_lateral(self, layer_type: str, size: int):
        """
        Register a lateral (recurrent) weight matrix for a given layer type.

        Args:
            layer_type (str): The type of layer (e.g., 'attn', 'fc1', 'linear').
            size (int): The size of the square lateral weight matrix.
        """
        if not hasattr(self, 'W_latents'):
            self.W_latents = {}

        if layer_type not in self.W_latents:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            W = torch.empty(size, size, device = device)
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
        """
        Perform a single predictive coding inference step for the layer.

        Args:
            target_activity (torch.Tensor): Target activity tensor for the layer.
            layer (nn.Module, optional): The layer module (for linear layers).
            proj_layers (dict, optional): Dictionary of projection layers (for attention).
            layer_type (str): Type of layer ('embed', 'attn', 'fc1', 'linear', etc.).
            input_ids (torch.Tensor, optional): Input token IDs (for embedding layers).
            position_ids (torch.Tensor, optional): Position IDs (for embedding layers).
            t (int): Current inference step.
            T (int): Total number of inference steps.
            requires_update (bool): Whether to update weights.

        Returns:
            torch.Tensor or tuple: Updated activity tensor(s) for the layer.
        """
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
            # Caching for mu_word and mu_pos during inference
            if not hasattr(self, '_embed_cache'):
                self._embed_cache = {"mu_word": None, "mu_pos": None, "step": -1}
            use_cache = (not requires_update) and (self._embed_cache["step"] == t)
            mu, mu_word, mu_pos = step_embed(
                t, T, target_activity, layer, layer_type, input_ids, position_ids,
                self.local_lr, self.clamp_value, self.energy_fn_name, self.is_holding_error,
                requires_update,
                mu_word_cache=self._embed_cache["mu_word"] if use_cache else None,
                mu_pos_cache=self._embed_cache["mu_pos"] if use_cache else None
            )
            # Update cache if not requires_update or first step
            if not requires_update or t == 0:
                self._embed_cache["mu_word"] = mu_word
                self._embed_cache["mu_pos"] = mu_pos
                self._embed_cache["step"] = t
        elif layer_type == "attn":
            x, mu = step_attn(t, T, target_activity, x, self.W_latents, proj_layers, layer_type,
                              self.local_lr, self.clamp_value, self.use_lateral, self.is_holding_error,
                              self.energy_fn_name, self.update_bias, requires_update, layer_instance=self)
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
            self._x_cache["embed"] = (mu_word, mu_pos)
            return mu_word, mu_pos
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
        device: torch.device = None,
    ):
        """
        Initialize the layer's state variables and store them in x_cache.

        Args:
            batch_size (int): Batch size.
            seq_len (int): Sequence length.
            layer (nn.Module, optional): The layer module (for linear layers).
            proj_layers (dict, optional): Dictionary of projection layers (for attention).
            layer_type (str): Type of layer ('embed', 'attn', 'fc1', 'linear', etc.).
            input_ids (torch.Tensor, optional): Input token IDs (for embedding layers).
            position_ids (torch.Tensor, optional): Position IDs (for embedding layers).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None, "Embedding layer requires input_ids and position_ids"
            x_word = layer["word"].weight[input_ids]
            x_pos = layer["pos"].weight[position_ids]
            self._x_cache["embed"] = (x_word, x_pos)
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            self._x_cache["attn"] = x_init(batch_size, seq_len, H_out, device)

            if self.use_lateral:
                self.register_lateral(layer_type, H_in)
                
                if layer_type in self.W_latents:
                    self.W_latents[layer_type] = self.W_latents[layer_type].to(device)
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            H_in = layer.weight.shape[1]

            if self.use_lateral:
                self.register_lateral(layer_type, H_in)

                

    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """
        Get the cached activity tensor for a given layer type.

        Args:
            layer_type (str): The type of layer.
        Returns:
            torch.Tensor or None: Cached activity tensor, or None if not present.
        """
        return self._x_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        """
        Get the accumulated energy for the layer (if error holding is enabled).

        Returns:
            float or None: The accumulated energy value, or None if not computed.
        """
        return self._energy

    def clear_energy(self):
        """
        Clear the stored energy and cached states for the layer.
        """
        self._energy = 0.0
        self._x_cache.clear()

    def get_errors(self) -> list:
        """
        Get the list of error values accumulated during inference.

        Returns:
            list: List of error dictionaries for each inference step.
        """
        return self._errors

    def clear_errors(self):
        """
        Clear the stored errors for the layer.
        """
        self._errors = []

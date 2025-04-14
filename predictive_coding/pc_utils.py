import torch
import warnings
from typing import Generator, Tuple, List, Union
import torch.nn as nn

from .pc_layer import PCLayer

def get_model_pc_layers(model: nn.Module) -> Generator[nn.Module, None, None]:         
    for module in model.modules():
        if isinstance(module, PCLayer):
            yield module

def get_named_model_pc_layers(model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
    for name, module in model.named_modules():
        if isinstance(module, PCLayer):
            yield name, module

def get_model_xs(model: nn.Module, warn_if_uninitialized: bool = True) -> Generator[torch.Tensor, None, None]:
    for pc_layer in get_model_pc_layers(model):
        x = getattr(pc_layer, 'x', None)
        if x is None and warn_if_uninitialized:
            warnings.warn("Uninitialized x detected in PCLayer", RuntimeWarning)
        if x is not None:
            yield x

def compute_energies(model: nn.Module, named: bool = False, per_datapoint: bool = False, warn_batch_size: bool = True):
    energies = {}
    batch_sizes = []

    for name, pc_layer in get_named_model_pc_layers(model):
        energy = pc_layer.energy_per_datapoint if per_datapoint else pc_layer.energy
        if energy is not None:
            energies[name] = energy
            bs = energy.size(0) if per_datapoint else energy.size()
            batch_sizes.append(bs)

    if not energies:
        raise ValueError("No energies found in PCLayers.")

    if warn_batch_size and len(set(batch_sizes)) > 1:
        warnings.warn(f"Inconsistent energy batch sizes: {batch_sizes}", RuntimeWarning)

    return energies if named else list(energies.values())

def check_model_training_state(model: nn.Module) -> Union[bool, None]:
    states = model_pc_layer_training_states(model)
    if not states:
        return model.training
    if all(states) and model.training:
        return True
    if not model.training and all(not s for s in states):
        return False
    return None

def model_has_pc_layers(model: nn.Module) -> bool:                                  
    return any(True for _ in get_model_pc_layers(model))

def model_pc_layer_training_states(model: nn.Module) -> List[bool]:
    return [layer.training for layer in get_model_pc_layers(model)]

def preprocess_step_index_list(indices: Union[str, List[int]], T: int) -> List[int]:   
    if isinstance(indices, str):
        indices = indices.lower()
        if indices == "all":
            return list(range(2, T))
        elif indices == "last":
            return [T - 1]
        elif indices == "last_half":
            return list(range(T // 2, T))
        elif indices == "never":
            return []
        else:
            raise NotImplementedError(f"Unknown step index preset: {indices}")
    elif isinstance(indices, list):
        for i in indices:
            if not isinstance(i, int) or not (0 <= i < T):
                raise ValueError(f"Invalid timestep index: {i}")
        return indices
    else:
        raise TypeError("Invalid type for indices")
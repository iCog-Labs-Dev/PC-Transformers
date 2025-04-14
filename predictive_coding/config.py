import torch.nn as nn
from typing import Callable, Dict, Union, List
import torch.optim as optim
from dataclasses import dataclass, field

@dataclass
class PCTrainerConfig:
    model: nn.Module
    optimizer_x_fn: Callable = optim.SGD
    optimizer_x_kwargs: Dict = field(default_factory = dict)
    manual_optimizer_x_fn: Callable = None
    x_lr_discount: float = 0.5
    x_lr_amplifier: float = 1.0
    loss_x_fn: Callable = None
    loss_inputs_fn: Callable = None
    optimizer_p_fn: Callable = optim.Adam
    optimizer_p_kwargs: Dict = field(default_factory = dict)
    manual_optimizer_p_fn: Callable = None
    T: int = 512
    update_x_at: Union[str, List[int]] = "all"
    update_p_at: Union[str, List[int]] = "all"
    energy_coefficient: float = 1.0
    early_stop_condition: str = "False"
    update_p_at_early_stop: bool = True
    is_disable_warning_energy_from_different_batch_sizes: bool = False

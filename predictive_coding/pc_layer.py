import torch
import torch.nn as nn
from typing import Callable

class PCLayer(nn.Module):
    def __init__(self,
                energy_fn: Callable = None,
                sample_x_fn: Callable = None,
                S: torch.Tensor = None,
                M: torch.Tensor = None,
                is_holding_error: bool = False,
                is_keep_energy_per_datapoint: bool = False): 
        
        super().__init__()
        self.energy_fn = energy_fn if energy_fn else (lambda inputs: (inputs['mu'] - inputs['x'])**2 - 0.5)
        assert callable(self.energy_fn)
        
        self.sample_x_fn = sample_x_fn if sample_x_fn else (lambda inputs: inputs['mu'].detach().clone())
        assert callable(self.sample_x_fn)

        self._x = None
        self._S = S
        self._M = M
        self._is_sample_x = False
        self.is_holding_error = is_holding_error
        self.is_keep_energy_per_datapoint = is_keep_energy_per_datapoint

    @property
    def x(self) -> nn.Parameter:
        return self._x
    
    @property
    def S(self)->torch.Tensor:
        return self._S
    
    @S.setter
    def S(self, value: torch.Tensor)-> None:
        if value is not None:
            assert isinstance(value, torch.Tensor)
            assert value.dim() == 2
        self._S = value

    @property
    def M(self)-> torch.Tensor:
        return self._M
    
    @M.setter
    def M(self, value: torch.Tensor)->None:
        if value is not None:
            assert isinstance(value, torch.Tensor)
        self._M = value

    @property
    def is_sample_x(self)-> bool:          
        """ Returns if x should be sampled. """
        return self._is_sample_x

    @is_sample_x.setter
    def is_sample_x(self, value: bool)-> None:         
        """ Sets if x should be sampled. """
        assert isinstance(value, bool)
        self._is_sample_x = value
        if value:
            self._x = None
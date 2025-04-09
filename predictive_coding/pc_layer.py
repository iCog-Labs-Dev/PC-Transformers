import torch
import torch.nn as nn
import warnings
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
        self._energy = None     
        self._energy_per_datapoint = None    
        self.is_holding_error = is_holding_error
        self.is_keep_energy_per_datapoint = is_keep_energy_per_datapoint

        self.clear_energy() 

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
    
    @property
    def energy(self)-> torch.Tensor:             
        """ Get energy held by PCLayer"""     
        return self._energy
    
    @property
    def energy_per_datapoint(self)-> torch.Tensor:             
        assert self.is_keep_energy_per_datapoint, "Enable is_keep_energy_per_datapoint."
        return self._energy_per_datapoint
    
    def clear_energy(self):            
        """Resets the stored energy values."""
        self._energy = None
        self._energy_per_datapoint = None
    
    def forward(self, mu: torch.Tensor, energy_fn_additional_inputs: dict = None) -> torch.Tensor:
        energy_fn_additional_inputs = energy_fn_additional_inputs or {}

        if not self.training:
            return mu

        if not self.is_sample_x and (self._x is None or mu.device != self._x.device or mu.size() != self._x.size()):
            warnings.warn(f"Auto-setting is_sample_x=True (mu.shape={mu.shape}, x.shape={getattr(self._x, 'shape', None)})", RuntimeWarning)
            self.is_sample_x = True

        if self.is_sample_x:
            self._x = nn.Parameter(self.sample_x_fn({"mu": mu, "x": self._x}).to(mu.device), requires_grad=True)
            self.is_sample_x = False

        x = self._x
        if self.S is not None:
            assert mu.dim() == x.dim() == 2
            mu = mu.unsqueeze(2).expand(-1, -1, x.size(1))
            x = x.unsqueeze(1).expand(-1, mu.size(1), -1)

        energy = self.energy_fn({"mu": mu, "x": x, **energy_fn_additional_inputs})
        scale = self.S if self.S is not None else self.M
        if scale is not None:
            energy *= scale.unsqueeze(0)

        self._energy = energy.sum()
        if self.is_keep_energy_per_datapoint:
            self._energy_per_datapoint = energy.sum(dim=tuple(range(1, energy.dim())))
                
        if self.is_holding_error:
            self.error = (self._x.detach() - mu).clone()

        return self._x
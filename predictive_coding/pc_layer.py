import torch
import torch.nn as nn
from typing import Callable, Dict

class PCLayer(nn.Module):
    def __init__(self,
                energy_fn: Callable = None,
                sample_x_fn: Callable = None,
                is_holding_error: bool = False,
                local_weight_shapes: Dict[str, tuple] = None,
                local_learning_rate: float = 1e-7): 
        
        super().__init__()
        self.energy_fn = energy_fn 
        self.sample_x_fn = sample_x_fn

        self._x = None
        self.local_lr = local_learning_rate
        self.is_holding_error = is_holding_error
        self.clamp_value = 1.0

        self.local_weights = nn.ParameterDict()
        for name, shape in local_weight_shapes.items():
            self.local_weights[name] = nn.Parameter(
                torch.randn(*shape) * 0.01,
                requires_grad=False
            )
        self.clear_energy() 
    
    def _hebbian_update(self, pre: torch.Tensor, post: torch.Tensor):
        with torch.no_grad():
            pre_flat = pre.reshape(-1, pre.size(-1))
            post_flat = post.reshape(-1, post.size(-1))

            for name, weight in self.local_weights.items():
                if 'weight' in name:
                    delta = self.local_lr * torch.matmul(pre_flat.T, post_flat)- 0.0001 * weight
                    weight.add_(delta.clamp_(-0.01, 0.01))
                    del delta
                elif 'bias' in name:
                    delta = self.local_lr * post_flat.mean(dim=0)
                    weight.add_(delta.clamp(-0.005, 0.005))
    
    def _apply_local_weights(self, x: torch.Tensor) -> torch.Tensor:
        for name, weight in self.local_weights.items():
            if 'weight' in name:
                x = torch.matmul(x, weight)
            elif 'bias' in name:
                x = x + weight
        return x
    
    def clear_energy(self):            
        """Resets the stored energy values."""
        self._energy = None
        self._energy_per_datapoint = None
    
    def forward(self, mu: torch.Tensor) -> torch.Tensor:

        if not self.training:
            return mu
        
        if self._x is None or mu.size() != self._x.size():
            self._x = self.sample_x_fn({"mu": mu, "x": self._x})
        
        with torch.no_grad():
            for _ in range(10):
                grad = self._x - mu
                grad = torch.clamp(grad, -self.clamp_value, self.clamp_value)
                self._x = self._x - self.local_lr * grad
                self._x = torch.clamp(self._x, -3.0, 3.0)

        x = self._x
        if len(self.local_weights) > 0:
            x = self._apply_local_weights(x)

        energy = self.energy_fn({"mu": mu, "x": x})
        self._energy = energy.mean()

        if self.is_holding_error:
            self.error = (self._x.detach() - mu).clone()
        
        self._hebbian_update(mu, self._x)

        return x
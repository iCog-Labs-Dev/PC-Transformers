import torch
import math


def d_gelu(x, approximate='none'):
    """
    Derivative of PyTorch's GELU.
    
    When approximate='none' (default):
        d/dx GELU(x) = 0.5 * (1 + erf(x/sqrt(2))) + 
                       x * exp(-x²/2) / sqrt(2*pi)
    
    When approximate='tanh':
        d/dx GELU(x) = 0.5 * (1 + tanh(inner)) + 
                       0.5 * x * sech²(inner) * sqrt(2/pi) * (1 + 3*k*x²)
    
    Args:
        x: Input tensor
        approximate: 'none' (exact, default) or 'tanh' (approximation)
    
    Returns:
        Derivative of GELU w.r.t. x
    """
    if approximate == 'tanh':
        # Derivative of tanh approximation
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        k = 0.044715
        
        inner = sqrt_2_over_pi * (x + k * x**3)
        tanh_inner = torch.tanh(inner)
        sech2_inner = 1 - tanh_inner**2
        d_inner_dx = sqrt_2_over_pi * (1 + 3 * k * x**2)
        
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2_inner * d_inner_dx
    else:
        # Derivative of erf-based version (exact)
        sqrt_2 = math.sqrt(2)
        sqrt_2pi = math.sqrt(2 * math.pi)
        
        # Term 1: 0.5 * (1 + erf(x/sqrt(2)))
        term1 = 0.5 * (1 + torch.erf(x / sqrt_2))
        
        # Term 2: x * exp(-x²/2) / sqrt(2*pi)
        term2 = x * torch.exp(-x**2 / 2) / sqrt_2pi
        
        return term1 + term2

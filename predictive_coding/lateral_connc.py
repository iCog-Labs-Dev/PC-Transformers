import torch
import torch.nn as nn
import torch.nn.functional as F

class LateralConnections(nn.Module):
    """
    Manages lateral connections for a layer.
    Implements anti-Hebbian learning for decorrelation.
    """
    def __init__(self, size: int, local_lr: float, inference_lr: float):
        super().__init__()
        self.size = size
        self.local_lr = local_lr
        self.inference_lr = inference_lr

        # Initialize lateral weight matrix
        W = torch.empty(size, size)
        nn.init.xavier_uniform_(W)
        self.W_lateral = nn.Parameter(W)
    
    def forward(self, x: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Apply lateral connections to combine error with lateral influence.
        
        Args:
            x: Current layer activity (B, S, H)
            error: Prediction error (B, S, H)
            
        Returns:
            delta_x: Combined error + lateral influence (B, S, H)
        """
        x_latent = torch.einsum("bsh,hv->bsv", x, self.W_lateral)
        delta_x = error + x_latent
        
        return delta_x
    
    def update_weights(self, x: torch.Tensor):
        """Anti-Hebbian weight update for decorrelation. """
        with torch.no_grad():
            anti_hebbian = -torch.einsum("bsh,bsv->hv", x, x)
            self.W_lateral.data.add_(self.local_lr * anti_hebbian)
            self.W_lateral.data = F.normalize(self.W_lateral.data, p=2, dim=1)
            
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)
    
    def set_inference_learning_rate(self, inference_lr: float):
        """Set the inference learning rate for the layer."""
        self.inference_lr = float(inference_lr)
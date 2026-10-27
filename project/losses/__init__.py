"""losses package"""
from .mse_loss import MSELoss
from .pinn_loss import PINNLoss, GraphLaplacian

__all__ = ['MSELoss', 'PINNLoss', 'GraphLaplacian']

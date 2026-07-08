"""Shared utilities for hook-based LightningModules.
Provides common classes and functions used across EKF and BayesCap hook modules
to eliminate code duplication.
"""

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)

class HuberLoss(nn.Module):
    """Huber loss with configurable threshold.

    Computes the smooth L1 loss with a quadratic region for small residuals
    and linear region for large residuals.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss between prediction and target.

        Args:
            pred: Predicted values of shape (...).
            target: Target values of shape (...).
        Returns:
            Scalar loss tensor.
        """
        l1_norm = torch.abs(target - pred)
        # Use torch.where for vectorized conditional — handles batched input
        quadratic = 0.5 * (l1_norm ** 2)
        linear = self.threshold * (l1_norm - 0.5 * self.threshold)
        return torch.where(l1_norm < self.threshold, quadratic, linear).mean()

def check_gradient(model: nn.Module) -> None:
    """Log gradient statistics for all parameters of a model.

    Useful for debugging training dynamics. Uses the module logger
    rather than print() for proper log level control.

    Args:
        model: The PyTorch module to inspect.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            log.debug("Layer: %-30s | Gradient: NOT REQUIRED", name)
        elif param.grad is None:
            log.debug("Layer: %-30s | Gradient: NONE", name)
        else:
            grad_norm = param.grad.norm().item()
            log.debug("Layer: %-30s | Gradient Norm: %.6f", name, grad_norm)
"""
EKF Generalized Gaussian Distribution NLL Loss.

alpha (scale) comes from EKF-propagated sigma_pred, not from a neural network.
beta (shape) is a single learnable scalar parameter.

GGD NLL: (|y - mu| / alpha)^beta + log(alpha) + lgamma(1/beta) - log(beta)
"""
import torch
import torch.nn as nn


class EKFGGDNLLLoss(nn.Module):
    """Generalized Gaussian NLL where alpha = sqrt(sigma_pred_sq) from EKF chain.

    The existing BayesCap1D neural head is NOT used here; alpha is sourced
    directly from the EKF Jacobian propagation. beta is the only learned parameter.
    """

    def __init__(self, eps: float = 1e-8, learn_calibration: bool = False):
        """
        Args:
            eps: numerical floor for alpha to prevent log(0)
            learn_calibration: if True, learn affine (a, b) s.t. alpha = a*sqrt(sigma) + b
        """
        super().__init__()
        # log_beta initialized to 0.5 -> beta = exp(0.5) ~ 1.65 (between Laplace and Gaussian)
        self.log_beta = nn.Parameter(torch.tensor(0.5))
        self.eps = eps
        self.learn_calibration = learn_calibration
        if learn_calibration:
            self.log_a = nn.Parameter(torch.tensor(0.0))  # a = 1.0
            self.b = nn.Parameter(torch.tensor(0.0))       # b = 0.0

    def forward(
        self,
        y_true: torch.Tensor,
        mu_pred: torch.Tensor,
        sigma_pred_sq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_true: (B,) or (B, 1) ground-truth targets
            mu_pred: (B,) or (B, 1) point predictions from frozen model
            sigma_pred_sq: (B,) EKF-propagated predictive variance

        Returns:
            Scalar mean NLL loss
        """
        y_true = y_true.squeeze()
        mu_pred = mu_pred.squeeze()

        beta = torch.exp(self.log_beta)

        if self.learn_calibration:
            a = torch.exp(self.log_a)
            alpha = a * torch.sqrt(sigma_pred_sq + self.eps) + self.b
        else:
            alpha = torch.sqrt(sigma_pred_sq + self.eps)

        alpha = alpha.clamp(min=self.eps)

        residual = torch.abs(y_true - mu_pred)
        nll = (
            (residual / alpha) ** beta
            + torch.log(alpha)
            + torch.lgamma(1.0 / beta)
            - torch.log(beta)
        )

        return nll.mean()

    def extra_repr(self) -> str:
        return (f"beta={torch.exp(self.log_beta).item():.3f}, "
                f"learn_calibration={self.learn_calibration}")

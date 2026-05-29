"""
EKF Generalized Gaussian Distribution NLL Loss.

alpha (scale) comes from EKF-propagated sigma_pred, not from a neural network.
beta (shape) is a single learnable scalar parameter.

GGD NLL: (|y - mu| / alpha)^beta + log(alpha) + lgamma(1/beta) - log(beta)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.models.components.toy import MLP, Residual, get_normalization
from src.plugins.var import BreakpointContext, BreakpointOutput
from src.plugins.ekf_propagation import * 

class EKFBiModalInferer(nn.Module):
    def __init__(self, 
                    reconstructor: nn.Module,
                    predictor: nn.Module,
                    latent_dim: 16,
                    n_modals:int = 2,
                    output_dim: int  =  1,
                    hidden_dims:  Sequence | List | int = 256,
                    bottleneck_dim: int | None = None,
                    per_dim_uncertainty: bool = True,
                    dropout: float = 0.0,
                    activation: str = "gelu",
                    norm: str = "batch",
                    eps: float = 1e-6,
                    residual = False
                    ):
        # We gonna take 
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.reconstructor = reconstructor
        self.predictor = predictor
        latent_size = latent_dim * n_modals
        self.per_dim_uncertainty = per_dim_uncertainty
        self.output_dim = output_dim if per_dim_uncertainty else 1
        self.eps = eps
        bottleneck_dim = bottleneck_dim or hidden_dims[-1]

        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Handle J_f scalar function (B, Src, Dst) -> (B, )
        stem_dim = hidden_dims[0]
        hidden_dim = hidden_dims[-1]
        inv_alpha_stem = nn.Sequential( 
                                    nn.Linear(latent_size + output_dim, stem_dim, bias=False),
                                    act(),
                                )
        inv_alpha_blocks = MLP(in_dim=stem_dim,
                                hidden_dims=hidden_dims,
                                out_dim=hidden_dim,
                                activation=activation,
                                norm = norm,
                                residual= residual,
                                dropout=dropout
                                )
        inv_alpha_head = nn.Sequential(
                                    nn.Linear(hidden_dim, self.output_dim),
                                    nn.ReLU(),
                                )
        self.inv_alpha_net = nn.Sequential(inv_alpha_stem, inv_alpha_blocks, inv_alpha_head)

        beta_stem = nn.Sequential(
                                    nn.Linear(latent_size, stem_dim, bias=False),
                                    act(),
                                    nn.LayerNorm(stem_dim)
                                )
        beta_blocks = MLP(in_dim=stem_dim,
                                hidden_dims=hidden_dims,
                                out_dim=hidden_dim,
                                activation=activation,
                                norm = norm,
                                residual= residual,
                                dropout=dropout
                                )
        beta_head = nn.Sequential(
                                    nn.Linear(hidden_dim, self.output_dim),
                                    nn.ReLU(),
                                )

        self.beta_net = nn.Sequential(beta_stem, beta_blocks, beta_head)

    # Export 
    def get_recon_fn(self, signal: tuple = (1, 1)):
        def infer(z):
            return self.reconstructor.forward_raw(z, signal=signal)
        return infer
    
    def forward(self, z: torch.Tensor, sigma_z: torch.Tensor, signal: tuple = (1, 1)):
        """
        Args:
            z:       (B, d_z) latent batch
            sigma_z: (B, d_z, d_z) per-sample input covariance from the Sigma_z provider
            signal:  (p1, p2) modality-presence tuple
        """
        assert self.predictor is not None, "Prediction head is None"
        recon_fn = self.get_recon_fn(signal=signal)
        pred_fn = self.predictor.forward
        sigma_pred_sq, diag_sigma_recon, J_f, J_g = full_ekf_propagation_full(z,
                                                                              sigma_z=sigma_z,
                                                                              reconstructor_fn=recon_fn,
                                                                              predictor_fn=pred_fn)
        if len(sigma_pred_sq.shape) < 2:
            sigma_pred_sq = sigma_pred_sq.unsqueeze_(-1)
        # J_f = (B, Src, Dst) -> (B, Dst)
        # Taking eigenvalues as measure for variance, the more exploding they are, the more uniform the shape    
        S_f = torch.linalg.svdvals(J_f.permute(0, 2, 1))
        # So minus one is to compare the function to Identity Mapping, so...
        # print(S_f.shape, J_f.shape)
        beta = self.beta_net(S_f / torch.amax(S_f, dim=-1, keepdim=True))
        # print(sigma_pred_sq.shape, diag_sigma_recon.shape)
        inv_alpha =  self.inv_alpha_net(torch.concatenate([sigma_pred_sq, diag_sigma_recon], dim=-1))
        return inv_alpha, beta
        
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

    def get_variance(self, sigma_pred_sq):
        beta = torch.exp(self.log_beta)

        if self.learn_calibration:
            a = torch.exp(self.log_a)
            alpha = a * torch.sqrt(sigma_pred_sq + self.eps) + self.b
        else:
            alpha = torch.sqrt(sigma_pred_sq + self.eps)

        alpha = alpha.clamp(min=self.eps)

        return alpha.pow(2) * torch.exp(torch.lgamma(3 / beta) - torch.lgamma(1 / beta))
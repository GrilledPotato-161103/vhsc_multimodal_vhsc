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
import torch.nn.functional as F
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
                    pred_dim: int = 1,
                    output_dim: int  =  1,
                    hidden_dims:  Sequence | List | int = 256,
                    bottleneck_dim: int | None = None,
                    per_dim_uncertainty: bool = True,
                    dropout: float = 0.0,
                    activation: str = "gelu",
                    norm: str = "batch",
                    eps: float = 1e-6,
                    residual = False,
                    mode: str = "learned",
                    beta_min: float = 0.5,
                    beta_max: float = 4.0,
                    ):
        # We gonna take
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pred_dim = pred_dim
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.reconstructor = reconstructor
        self.predictor = predictor
        latent_size = latent_dim * n_modals
        self.per_dim_uncertainty = per_dim_uncertainty
        self.output_dim = output_dim if per_dim_uncertainty else 1
        self.eps = eps
        if mode not in ("learned", "closed_form"):
            raise ValueError(f"mode must be 'learned' or 'closed_form', got {mode!r}")
        self.mode = mode
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
        hidden_dims = hidden_dims[1:-1]
        # Output encoders to sync mu head
        output_enc_stem = nn.Sequential(
            nn.Linear(self.pred_dim, stem_dim),
            act(),
        )
        if len(hidden_dims) > 0: 
            output_enc_blocks = MLP(in_dim=stem_dim,
                                hidden_dims=hidden_dims,
                                out_dim=hidden_dim,
                                activation=activation,
                                norm = norm,
                                residual= True,
                                dropout=dropout
                                )
        else:
            output_enc_blocks = nn.Identity()
        self.output_enc = nn.Sequential(output_enc_stem, output_enc_blocks)
        # Mu head
        self.mu_head = nn.Linear(hidden_dim, self.pred_dim)

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
        # No terminal ReLU: a hard 0 makes alpha = 1/inv_alpha blow up to inf.
        # softplus(+eps) is applied in forward() instead -> inv_alpha is strictly
        # positive and smooth.
        inv_alpha_head = nn.Sequential(
                                    nn.Linear(hidden_dim, self.output_dim),
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
        # No terminal ReLU: beta -> 0 makes lgamma(1/beta) overflow. Output a raw
        # logit; forward() squashes it into [beta_min, beta_max] with a sigmoid.
        beta_head = nn.Sequential(
                                    nn.Linear(hidden_dim, self.output_dim),
                                )

        self.beta_net = nn.Sequential(beta_stem, beta_blocks, beta_head)

    # Export 
    def get_recon_fn(self, signal: tuple = (1, 1)):
        def infer(z):
            return self.reconstructor.forward_raw(z, signal=signal)
        return infer
    
    def forward(self, z: torch.Tensor, sigma_z: torch.Tensor, pred: torch.Tensor, signal: tuple = (1, 1)):
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

        if self.mode == "closed_form":
            # Linear-Gaussian closed form (formalism/02 §3): beta=2, alpha=sqrt(2 sigma_pred_sq).
            # No learnable parameters in the heads — pure EKF-driven uncertainty.
            inv_alpha = 1.0 / torch.sqrt(2.0 * sigma_pred_sq + self.eps)
            beta = torch.full_like(sigma_pred_sq, 2.0)
            return inv_alpha, beta, sigma_pred_sq

        # mode == "learned" — bounded heads (see __init__ notes).
        S_f = torch.linalg.svdvals(J_f.permute(0, 2, 1))
        output_latent = self.output_enc(pred)
        beta_raw = self.beta_net(S_f / torch.amax(S_f, dim=-1, keepdim=True))
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta_raw)
        # Feed the EKF variance in log-space so the head sees a well-scaled signal
        # regardless of whether sigma_pred_sq is 1e-4 or 1e4.
        ekf_feat = torch.log(torch.cat([sigma_pred_sq, diag_sigma_recon], dim=-1).clamp_min(self.eps))
        inv_alpha = F.softplus(self.inv_alpha_net(ekf_feat)) + self.eps
        mu = self.mu_head(output_latent)
        return mu, inv_alpha, beta, sigma_pred_sq
        
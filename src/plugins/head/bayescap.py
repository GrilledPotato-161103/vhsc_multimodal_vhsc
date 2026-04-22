from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.models.components.toy import MLP, Residual, get_normalization
from src.plugins.var import BreakpointContext, BreakpointOutput

# -----------------------------
# 1D BayesCap head
# -----------------------------
class BayesCap1D(nn.Module):
    """
    BayesCap for vector outputs.

    Input:
        y_hat: [..., D]

    Output:
        mu:        [..., D]
        inv_alpha: [..., U]   where U = D if per_dim_uncertainty else 1
        beta:      [..., U]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence | List | int = 256,
        bottleneck_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        per_dim_uncertainty: bool = True,
        norm: str = "batch",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        print("Initializing")
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.uncertainty_dim = input_dim if per_dim_uncertainty else 1
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
        
        stem_dim = hidden_dims[0]
        print(stem_dim)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_dim),
            act(),
        )
        hidden_dim = hidden_dims[-1]
        hidden_dims = self.hidden_dims[1:-1]
        if len(hidden_dims) > 0: 
            self.blocks = MLP(in_dim=stem_dim,
                                hidden_dims=hidden_dims,
                                out_dim=hidden_dim,
                                activation=activation,
                                norm = norm,
                                residual= True,
                                dropout=dropout
                                )
        else:
            self.blocks = nn.Identity()
        

        self.mu_head = MLP(in_dim= hidden_dim,
                           hidden_dims=[hidden_dim],
                           out_dim=input_dim,
                           activation=activation,
                           norm=norm,
                           residual= False,
                           dropout=dropout)

        self.alpha_head = MLP(in_dim= hidden_dim,
                           hidden_dims=[hidden_dim],
                           out_dim=self.uncertainty_dim,
                           activation=activation,
                           norm=norm,
                           residual= False,
                           dropout=dropout)

        self.beta_head = MLP(in_dim= hidden_dim,
                           hidden_dims=[hidden_dim],
                           out_dim=self.uncertainty_dim,
                           activation=activation,
                           norm=norm,
                           residual= False,
                           dropout=dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        y_hat = ctx.output
        original_shape = y_hat.shape
        if original_shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim = {self.input_dim}, got {original_shape[-1]}"
            )

        x = y_hat.reshape(-1, self.input_dim)
        h = self.stem(x)
        h = self.blocks(h)

        mu = self.mu_head(h)
        alpha = F.softplus(self.alpha_head(h)) + self.eps
        beta = F.softplus(self.beta_head(h)) + self.eps

        out_prefix = original_shape[:-1]
        mu = mu.view(*out_prefix, self.input_dim)
        alpha = alpha.view(*out_prefix, self.uncertainty_dim)
        beta = beta.view(*out_prefix, self.uncertainty_dim)
        cls = BreakpointOutput(
                fn_name=self.forward.__qualname__,
                context = ctx,
                output = mu,
                trace={
                       "signal": ctx.bp_kwargs,
                       "input": y_hat,
                       "output": (mu, alpha, beta)
                       }
        )
        return cls

# -----------------------------
# Loss
# -----------------------------
class BayesCap1DLoss(nn.Module):
    """
    BayesCap loss for vector outputs.

    identity_mode:
        - "l2": paper-style identity term
        - "l1": repo-style identity term

    nll_mode:
        - "paper": (|mu - y| / alpha)^beta - log(beta/alpha) + log Gamma(1/beta)
                   implemented via inv_alpha = 1/alpha
        - "repo": repo-compatible simplified variant
    """

    def __init__(
        self,
        lambda_identity: float = 1.0,
        lambda_nll: float = 1.0,
        identity_mode: str = "l2",
        nll_mode: str = "paper",
        reduction: str = "mean",
        eps: float = 1e-6,
        resi_min: float = 1e-6,
        resi_max: float = 1e6,
    ) -> None:
        super().__init__()
        if identity_mode not in {"l1", "l2"}:
            raise ValueError("identity_mode must be 'l1' or 'l2'")
        if nll_mode not in {"paper", "repo"}:
            raise ValueError("nll_mode must be 'paper' or 'repo'")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.lambda_identity = lambda_identity
        self.lambda_nll = lambda_nll
        self.identity_mode = identity_mode
        self.nll_mode = nll_mode
        self.reduction = reduction
        self.eps = eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def _reduce(self, x: Tensor) -> Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x

    def _broadcast_uncertainty(
        self,
        mu: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Support scalar uncertainty per sample: [..., 1] -> [..., D]
        if alpha.shape[-1] == 1 and mu.shape[-1] > 1:
            alpha = alpha.expand_as(mu)
        if beta.shape[-1] == 1 and mu.shape[-1] > 1:
            beta = beta.expand_as(mu)
        if alpha.shape != mu.shape or beta.shape != mu.shape:
            raise ValueError(
                f"After broadcasting, expected alpha/beta to match mu shape. "
                f"Got mu={mu.shape}, alpha={alpha.shape}, beta={beta.shape}"
            )
        return alpha, beta

    def identity_loss(self, mu: Tensor, y_hat: Tensor) -> Tensor:
        if self.identity_mode == "l2":
            return F.mse_loss(mu, y_hat, reduction=self.reduction)
        return F.l1_loss(mu, y_hat, reduction=self.reduction)

    def generalized_gaussian_nll(
        self,
        mu: Tensor,
        alpha: Tensor,
        beta: Tensor,
        y_true: Tensor,
    ) -> Tensor:
        alpha, beta = self._broadcast_uncertainty(mu, alpha, beta)

        alpha = alpha.clamp_min(self.eps)
        beta = beta.clamp_min(self.eps)
        abs_err = torch.abs(mu - y_true)

        if self.nll_mode == "paper":
            # Exact paper-style expression, using inv_alpha = 1 / alpha
            # (|mu-y| / alpha)^beta = (|mu-y| * inv_alpha)^beta
            scaled = (abs_err / alpha).clamp(
                min=self.resi_min, max=self.resi_max
            )
            nll = (
                torch.pow(scaled, beta)
                + torch.log(alpha)
                - torch.log(beta)
                + torch.lgamma(1.0 / beta)
            )
        else:
            # Repo-compatible simplified form
            scaled = (abs_err / alpha * beta).clamp(
                min=self.resi_min, max=self.resi_max
            )
            nll = (
                scaled
                + torch.log(alpha)
                + torch.lgamma(1.0 / beta)
                - torch.log(beta)
            )

        return self._reduce(nll)

    def forward(
        self,
        mu: Tensor,
        alpha: Tensor,
        beta: Tensor,
        y_hat: Tensor,
        y_true: Tensor,
    ) -> Dict[str, Tensor]:
        if mu.shape != y_hat.shape or mu.shape != y_true.shape:
            raise ValueError(
                f"mu, y_hat, y_true must match. Got {mu.shape}, {y_hat.shape}, {y_true.shape}"
            )

        loss_identity = self.identity_loss(mu, y_hat)
        loss_nll = self.generalized_gaussian_nll(mu, alpha, beta, y_true)
        loss = self.lambda_identity * loss_identity + self.lambda_nll * loss_nll

        return {
            "loss": loss,
            "identity_loss": loss_identity.detach(),
            "nll_loss": loss_nll.detach(),
        }


# -----------------------------
# Variance utility
# -----------------------------
def bayescap_variance_1d(
    alpha: Tensor,
    beta: Tensor,
    target_dim: Optional[int] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Convert BayesCap parameters to predictive variance.

    variance = alpha^2 * Gamma(3 / beta) / Gamma(1 / beta)
    where alpha = 1 / inv_alpha
    """
    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

    if target_dim is not None and alpha.shape[-1] == 1 and target_dim > 1:
        alpha = alpha.expand(*alpha.shape[:-1], target_dim)
    if target_dim is not None and beta.shape[-1] == 1 and target_dim > 1:
        beta = beta.expand(*beta.shape[:-1], target_dim)

    return alpha.pow(2) * torch.exp(torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))

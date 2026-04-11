from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.models.components.toy import MLP, Residual
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
                                use_norm = True,
                                residual= True,
                                dropout=dropout
                                )
        else:
            self.blocks = nn.Identity()
        

        self.mu_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.inv_alpha_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, self.uncertainty_dim),
        )

        self.beta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, self.uncertainty_dim),
        )

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
        inv_alpha = F.softplus(self.inv_alpha_head(h)) + self.eps
        beta = F.softplus(self.beta_head(h)) + self.eps

        out_prefix = original_shape[:-1]
        mu = mu.view(*out_prefix, self.input_dim)
        inv_alpha = inv_alpha.view(*out_prefix, self.uncertainty_dim)
        beta = beta.view(*out_prefix, self.uncertainty_dim)
        cls = BreakpointOutput(
                fn_name=self.forward.__qualname__,
                context = ctx,
                output = mu,
                trace={
                       "signal": ctx.bp_kwargs,
                       "input": y_hat,
                       "output": (mu, inv_alpha, beta)
                       }
        )
        return cls


# -----------------------------
# Wrapper: frozen model + BayesCap
# -----------------------------
class FrozenModelWithBayesCap1D(nn.Module):
    """
    Wrap a pretrained frozen model Psi and a 1D BayesCap head Omega.

    Psi(x) -> y_hat [..., D]
    Omega(y_hat) -> (mu, inv_alpha, beta)
    """

    def __init__(self, frozen_model: nn.Module, bayescap: BayesCap1D) -> None:
        super().__init__()
        self.frozen_model = frozen_model
        self.bayescap = bayescap

        for p in self.frozen_model.parameters():
            p.requires_grad = False
        self.frozen_model.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            y_hat = self.frozen_model(x)

        mu, inv_alpha, beta = self.bayescap(y_hat)
        return y_hat, mu, inv_alpha, beta


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
        inv_alpha: Tensor,
        beta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Support scalar uncertainty per sample: [..., 1] -> [..., D]
        if inv_alpha.shape[-1] == 1 and mu.shape[-1] > 1:
            inv_alpha = inv_alpha.expand_as(mu)
        if beta.shape[-1] == 1 and mu.shape[-1] > 1:
            beta = beta.expand_as(mu)
        if inv_alpha.shape != mu.shape or beta.shape != mu.shape:
            raise ValueError(
                f"After broadcasting, expected inv_alpha/beta to match mu shape. "
                f"Got mu={mu.shape}, inv_alpha={inv_alpha.shape}, beta={beta.shape}"
            )
        return inv_alpha, beta

    def identity_loss(self, mu: Tensor, y_hat: Tensor) -> Tensor:
        if self.identity_mode == "l2":
            return F.mse_loss(mu, y_hat, reduction=self.reduction)
        return F.l1_loss(mu, y_hat, reduction=self.reduction)

    def generalized_gaussian_nll(
        self,
        mu: Tensor,
        inv_alpha: Tensor,
        beta: Tensor,
        y_true: Tensor,
    ) -> Tensor:
        inv_alpha, beta = self._broadcast_uncertainty(mu, inv_alpha, beta)

        inv_alpha = inv_alpha.clamp_min(self.eps)
        beta = beta.clamp_min(self.eps)
        abs_err = torch.abs(mu - y_true)

        if self.nll_mode == "paper":
            # Exact paper-style expression, using inv_alpha = 1 / alpha
            # (|mu-y| / alpha)^beta = (|mu-y| * inv_alpha)^beta
            scaled = (abs_err * inv_alpha).clamp(
                min=self.resi_min, max=self.resi_max
            )
            nll = (
                torch.pow(scaled, beta)
                - torch.log(inv_alpha)
                - torch.log(beta)
                + torch.lgamma(1.0 / beta)
            )
        else:
            # Repo-compatible simplified form
            scaled = (abs_err * inv_alpha * beta).clamp(
                min=self.resi_min, max=self.resi_max
            )
            nll = (
                scaled
                - torch.log(inv_alpha)
                + torch.lgamma(1.0 / beta)
                - torch.log(beta)
            )

        return self._reduce(nll)

    def forward(
        self,
        mu: Tensor,
        inv_alpha: Tensor,
        beta: Tensor,
        y_hat: Tensor,
        y_true: Tensor,
    ) -> Dict[str, Tensor]:
        if mu.shape != y_hat.shape or mu.shape != y_true.shape:
            raise ValueError(
                f"mu, y_hat, y_true must match. Got {mu.shape}, {y_hat.shape}, {y_true.shape}"
            )

        loss_identity = self.identity_loss(mu, y_hat)
        loss_nll = self.generalized_gaussian_nll(mu, inv_alpha, beta, y_true)
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
    inv_alpha: Tensor,
    beta: Tensor,
    target_dim: Optional[int] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Convert BayesCap parameters to predictive variance.

    variance = alpha^2 * Gamma(3 / beta) / Gamma(1 / beta)
    where alpha = 1 / inv_alpha
    """
    inv_alpha = inv_alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

    if target_dim is not None and inv_alpha.shape[-1] == 1 and target_dim > 1:
        inv_alpha = inv_alpha.expand(*inv_alpha.shape[:-1], target_dim)
    if target_dim is not None and beta.shape[-1] == 1 and target_dim > 1:
        beta = beta.expand(*beta.shape[:-1], target_dim)

    alpha = 1.0 / inv_alpha
    return alpha.pow(2) * torch.exp(torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))


# -----------------------------
# Example training step
# -----------------------------
def example_training_step() -> None:
    class FrozenRegressor(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, in_dim, out_dim = 16, 32, 8

    frozen_model = FrozenRegressor(in_dim, out_dim).to(device)
    bayescap = BayesCap1D(
        input_dim=out_dim,
        hidden_dims=[128, 128, 128],
        per_dim_uncertainty=True,   # False -> one scalar alpha/beta per sample
    ).to(device)

    model = FrozenModelWithBayesCap1D(frozen_model, bayescap).to(device)

    criterion = BayesCap1DLoss(
        lambda_identity=1.0,
        lambda_nll=0.05,
        identity_mode="l2",   # "l1" to mimic repo
        nll_mode="paper",     # "repo" to mimic repo
    )

    optimizer = torch.optim.Adam(model.bayescap.parameters(), lr=1e-4)

    x = torch.randn(batch_size, in_dim, device=device)
    y_true = torch.randn(batch_size, out_dim, device=device)

    y_hat, mu, inv_alpha, beta = model(x)
    losses = criterion(mu, inv_alpha, beta, y_hat, y_true)

    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()

    variance = bayescap_variance_1d(inv_alpha, beta, target_dim=out_dim)

    print("y_hat:", y_hat.shape)
    print("mu:", mu.shape)
    print("inv_alpha:", inv_alpha.shape)
    print("beta:", beta.shape)
    print("variance:", variance.shape)
    print("loss:", float(losses["loss"]))


if __name__ == "__main__":
    example_training_step()
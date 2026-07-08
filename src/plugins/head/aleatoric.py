from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.models.components.toy import MLP, Residual, get_normalization

class AleatoricHead(nn.Module):
    """Learned aleatoric uncertainty head.

    Maps concat(z, log_sigma_ep) -> sigma_al (strictly positive).
    Captures in-distribution function complexity that the EKF epistemic
    term misses. Combined with sigma_ep via:
        sigma_total = sigma_ep + lambda_aleatoric * sigma_al

    input_mode:
      "z_and_sep" (default): concat(z, log_sigma_ep) — OOD-aware aleatoric
      "z_only":               z only — pure function complexity, ignores shift

    See formalism/06_empirical_validation.md §6 and the implementation plan.
    """

    def __init__(self,
                 z_dim: int = 32,
                 hidden_dim: int = 32,
                 n_layers: int = 2,
                 activation: str = "silu",
                 norm: str = "layer",
                 eps: float = 1e-6,
                 input_mode: str = "z_and_sep",
                 xy_dim: int = 2):
        super().__init__()
        self.eps = eps
        self.input_mode = input_mode
        act = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation]
        if input_mode == "xy":
            in_dim = xy_dim          # raw input coords bypass z entirely
        elif input_mode == "z_only":
            in_dim = z_dim
        else:
            in_dim = z_dim + 1       # z + log_sigma_ep
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=True), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       get_normalization(norm, hidden_dim),
                       act()]
        out_layer = nn.Linear(hidden_dim, 1)
        # Init bias so Softplus(bias) ≈ 0 at start — prevents early domination.
        nn.init.constant_(out_layer.bias, -3.0)
        nn.init.xavier_uniform_(out_layer.weight)
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, sigma_ep: torch.Tensor,
                xy: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z:        (B, z_dim) latent features
            sigma_ep: (B, 1) epistemic variance from EKF
            xy:       (B, xy_dim) raw input coords — required when input_mode='xy'
        Returns:
            sigma_al: (B, 1) strictly positive aleatoric variance
        """
        if self.input_mode == "xy":
            assert xy is not None, "xy required when input_mode='xy'"
            feat = xy.detach()
        elif self.input_mode == "z_only":
            feat = z.detach()
        else:
            log_sep = torch.log(sigma_ep.detach().clamp_min(self.eps))
            feat = torch.cat([z.detach(), log_sep], dim=-1)
        return F.softplus(self.net(feat)) + self.eps                 # (B, 1)


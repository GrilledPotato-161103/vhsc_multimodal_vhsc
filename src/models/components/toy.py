from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def get_activation(name: Literal["relu", "gelu", "silu"]) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")

class Residual(nn.Module):
    def __init__(self, blocks: nn.Module):
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x):
        x1 = self.blocks(x)
        return x + x1

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        dropout: float = 0.0,
        use_norm: bool = False,
        residual: bool = False
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = in_dim

        for h in hidden_dims:
            sub_layers: list[nn.Module] = []
            sub_layers.append(nn.Linear(prev_dim, h))
            if use_norm:
                sub_layers.append(nn.LayerNorm(h))
            sub_layers.append(get_activation(activation))
            if dropout > 0:
                sub_layers.append(nn.Dropout(dropout))
            block = nn.Sequential(*sub_layers)
            if residual and (prev_dim == h):
                block = Residual(block)
            layers.append(block)
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BiModalRegressor(nn.Module):
    """
    Two-tower fusion network.

    Input:
        x1: [B] or [B, d1]
        x2: [B] or [B, d2]

    Output:
        y_hat: [B]
    """

    def __init__(
        self,
        x1_dim: int = 1,
        x2_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        fusion_hidden_dims: list[int] | None = None,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        dropout: float = 0.0,
        use_norm: bool = False,
        use_residual: bool = False
    ) -> None:
        super().__init__()

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [128, 64]

        self.x1_encoder = MLP(
            in_dim=x1_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            use_norm=use_norm,
            residual=use_residual
        )

        self.x2_encoder = MLP(
            in_dim=x2_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            use_norm=use_norm,
            residual=use_residual
        )

        fusion_in_dim = latent_dim * 4

        self.head = MLP(
            in_dim=fusion_in_dim,
            hidden_dims=fusion_hidden_dims,
            out_dim=1,
            activation=activation,
            dropout=dropout,
            use_norm=use_norm,
            residual=use_residual
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.ndim == 1:
            x1 = x1.unsqueeze(-1)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(-1)

        z1 = self.x1_encoder(x1)
        z2 = self.x2_encoder(x2)

        z_mul = z1 * z2
        z_diff = z1 - z2

        z = torch.cat([z1, z2, z_mul, z_diff], dim=-1)
        y_hat = self.head(z).squeeze(-1)
        return y_hat
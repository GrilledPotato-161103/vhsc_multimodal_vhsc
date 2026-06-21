from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from torch import nn
from torch import Tensor

from src.models.components.common import get_activation, get_normalization

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
        activation: Literal["relu", "gelu", "silu", "none"] = "gelu",
        dropout: float = 0.0,
        norm: str = "layer",
        residual: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = in_dim
        num_groups = kwargs.get('num_groups', 1)
        for h in hidden_dims:
            sub_layers: list[nn.Module] = []
            sub_layers.append(nn.Linear(prev_dim, h))
            sub_layers.append(get_normalization(norm, num_features=h, dimension=1, num_groups=num_groups))
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
        norm: str = "batch",
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
            norm=norm,
            residual=use_residual
        )

        self.x2_encoder = MLP(
            in_dim=x2_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            norm=norm,
            residual=use_residual
        )

        fusion_in_dim = latent_dim * 2

        self.head = MLP(
            in_dim=fusion_in_dim,
            hidden_dims=fusion_hidden_dims,
            out_dim=1,
            activation=activation,
            dropout=dropout,
            norm=norm,
            residual=use_residual
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.x1_encoder(x1)
        z2 = self.x2_encoder(x2)
        z = torch.cat([z1, z2], dim=-1)
        y_hat = self.head(z).squeeze(-1)
        return y_hat


class MultiModalRegressor(nn.Module):
    def __init__(
        self,
        x_dims: int | Sequence[int] = 1,
        n_modals: int = 2,
        encoder_hidden_dims: int | Sequence[int] = 64,
        latent_dim: int = 32,
        fusion_hidden_dims: Sequence[int] | None = None,
        out_dim: int = 1,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        dropout: float = 0.0,
        norm: str = "batch",
        use_residual: bool = False
    ) -> None:
        super().__init__()

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [128, 64]
        if not isinstance(x_dims, Sequence):
            x_dims = [x_dims]
        if len(x_dims) == 1:
            x_dims = list(x_dims) * n_modals

        assert len(x_dims) == n_modals, "List of modality dimensions must match no. dims"
        if not isinstance(encoder_hidden_dims, Sequence):
            encoder_hidden_dims = [encoder_hidden_dims]
        
        self.encoders = nn.ModuleList([
            MLP(
                in_dim=x_dim,
                hidden_dims=encoder_hidden_dims,
                out_dim=latent_dim,
                activation=activation,
                dropout=dropout,
                norm=norm,
                residual=use_residual
                ) for x_dim in x_dims])
        self.head = nn.Sequential(MLP(
            in_dim=latent_dim,
            hidden_dims=fusion_hidden_dims,
            out_dim=fusion_hidden_dims[-1],
            activation=activation,
            dropout=dropout,
            norm=norm,
            residual=use_residual
        ), nn.Linear(fusion_hidden_dims[-1], out_dim))

    def forward(self, xs: Sequence[Tensor]) -> torch.Tensor:
        zs = [self.encoders[i](x) for i, x in enumerate(xs)]
        z = torch.stack(zs).sum(dim=0)
        y_hat = self.head(z).squeeze(-1)
        return y_hat
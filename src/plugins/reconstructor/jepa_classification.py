"""Cross-modal feature reconstructor for JEPA ViT token features.

Adapts the :class:`~src.plugins.reconstructor.linear.BilinearReconstructor`
pattern to work with ViT-produced token sequences ``[B, N, D]``.

When a modality is missing (``signal = 0``), the reconstructor:
1. Mean-pools the available modality's token features → ``[B, D]``
2. Projects through a learned MLP to reconstruct the missing modality's
   pooled features → ``[B, D]``
3. Broadcasts the reconstructed vector back to the token dimension
   ``[B, N, D]`` so the classifier receives properly-shaped inputs.

This approach preserves the cross-attention classifier's ability to
attend across token positions while only reconstructing semantic-level
(not positional) information.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.plugins.var import BreakpointContext, BreakpointOutput
from src.models.components.toy import MLP
from src.models.components.ffn import FeedForward


class JEPAClassificationReconstructor(nn.Module):
    """Cross-modal feature reconstruction for JEPA ViT token features.

    Expects ``ctx.inputs`` to be a tuple of two ViT token-feature tensors
    ``(feat_1, feat_2)`` each of shape ``[B, N, D]``.

    Internally mean-pools the token dimension for reconstruction, then
    broadcasts back for classifier compatibility.

    ``ctx.bp_kwargs`` is ``(p1, p2)``: ``1`` = present, ``0`` = missing.
    """

    def __init__(
        self,
        d_1: int,
        d_2: int,
        hidden_dims: int | Sequence[int] = 512,
        activation: str = "silu",
        norm: str = "layer",
        dropout: float = 0.2,
        order: str = "adn",
        dist: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.d_1 = d_1
        self.d_2 = d_2

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if len(hidden_dims) < 1:
            hidden_dims = [d_1 + d_2]

        hidden_dim = hidden_dims[-1]
        mlp_hidden = list(hidden_dims[:-1]) if len(hidden_dims) > 1 else []

        # ln12: modality 1 pooled features → modality 2 pooled features
        self.ln12 = nn.Sequential(
            MLP(
                in_dim=d_1,
                hidden_dims=mlp_hidden,
                out_dim=hidden_dim,
                activation=activation,
                norm=norm,
                dropout=dropout,
                residual=True,
            ),
            nn.Linear(hidden_dim, d_2),
        )

        # ln21: modality 2 pooled features → modality 1 pooled features
        self.ln21 = nn.Sequential(
            MLP(
                in_dim=d_2,
                hidden_dims=mlp_hidden,
                out_dim=hidden_dim,
                activation=activation,
                norm=norm,
                dropout=dropout,
                residual=True,
            ),
            nn.Linear(hidden_dim, d_1),
        )

        # Deviation networks estimate reconstruction uncertainty
        self.dev1 = FeedForward(
            in_dim=d_1 + d_2,
            hidden_dim=d_1 * 2,
            out_dim=d_1,
            activation=activation,
            norm=norm,
            dropout=dropout,
            adn_order=order,
        )

        self.dev2 = FeedForward(
            in_dim=d_1 + d_2,
            hidden_dim=d_2 * 2,
            out_dim=d_2,
            activation=activation,
            norm=norm,
            dropout=dropout,
            adn_order=order,
        )

        self.dist = dist if dist is not None else nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------
    # Breakpoint callback
    # ------------------------------------------------------------------

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        """Breakpoint callback entry point.

        Reads ViT token features from ``ctx.inputs``, mean-pools them,
        reconstructs missing modality features via MLP, and broadcasts
        back to token dimension.
        """
        (mod_1, mod_2) = ctx.inputs  # [B, N1, D1], [B, N2, D2]

        # Parse masked signal
        if ctx.bp_kwargs:
            p1, p2 = ctx.bp_kwargs
        else:
            p1, p2 = 1, 1

        # Mean-pool token dimension for reconstruction
        pool_1 = mod_1.mean(dim=1)  # [B, d_1]
        pool_2 = mod_2.mean(dim=1)  # [B, d_2]

        N1 = mod_1.shape[1]
        N2 = mod_2.shape[1]

        # Reconstruct missing modality (pooled level)
        rec_pool_2 = self.ln12(pool_1) if p1 == 0 else pool_2
        rec_pool_1 = self.ln21(pool_2) if p2 == 0 else pool_1

        # Broadcast back to token dimension
        rec_2 = rec_pool_2.unsqueeze(1).expand(-1, N2, -1) if p1 == 0 else mod_2
        rec_1 = rec_pool_1.unsqueeze(1).expand(-1, N1, -1) if p2 == 0 else mod_1

        # Distance metrics (on pooled features)
        dist_1 = self.dist(rec_pool_2, pool_2)
        dist_2 = self.dist(rec_pool_1, pool_1)

        # Deviation estimates
        dev_1 = self.dev1(torch.cat([rec_pool_1, pool_2], dim=-1))
        dev_2 = self.dev2(torch.cat([pool_1, rec_pool_2], dim=-1))

        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            context=ctx,
            output=(rec_1, rec_2),
            trace={
                "signal": ctx.bp_kwargs,
                "input": (mod_1, mod_2),
                "reconstructed": (rec_1, rec_2),
                "distance": (dist_1, dist_2),
                "dev": (dev_1, dev_2),
            },
        )

    # ------------------------------------------------------------------
    # Raw forward (EKF Jacobian compatible)
    # ------------------------------------------------------------------

    def forward_raw(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        signal: Tuple[int, int] = (1, 1),
    ) -> torch.Tensor:
        """Direct functional variant for EKF Jacobian computation.

        Works on pooled features (mean over token dim).
        """
        mod_1, mod_2 = inputs
        pool_1 = mod_1.mean(dim=1) if mod_1.dim() == 3 else mod_1
        pool_2 = mod_2.mean(dim=1) if mod_2.dim() == 3 else mod_2

        p1, p2 = signal
        p1_t = torch.as_tensor(p1, device=mod_1.device, dtype=mod_1.dtype)
        p2_t = torch.as_tensor(p2, device=mod_1.device, dtype=mod_1.dtype)

        rec_pool_2 = torch.where(p1_t == 0, self.ln12(pool_1), pool_2)
        rec_pool_1 = torch.where(p2_t == 0, self.ln21(pool_2), pool_1)

        return torch.cat([rec_pool_1, rec_pool_2], dim=-1)

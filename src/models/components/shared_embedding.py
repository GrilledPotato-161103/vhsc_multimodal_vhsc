"""Shared embedding network: per-modality encoders with mean fusion.

Mirrors the ``MultiModalRegressor`` pattern (ModuleList of MLPs) but uses
mean pooling for fusion and adds optional per-modality decoders for
reconstruction-based regularisation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn, Tensor

from src.models.components.toy import MLP


class SharedEmbeddingNet(nn.Module):
    """Per-modality encoders → shared embedding via mean fusion.

    Architecture::

        x_0 → Encoder_0 → emb_0 ┐
        x_1 → Encoder_1 → emb_1 ├─ mean → shared  ─┬─→ Decoder_0 → x̂_0
        x_2 → Encoder_2 → emb_2 ┘                   ├─→ Decoder_1 → x̂_1
                                                     └─→ Decoder_2 → x̂_2
    """

    def __init__(
        self,
        x_dims: int | Sequence[int] = 128,
        n_modals: int = 3,
        encoder_hidden_dims: Sequence[int] = (256, 256, 512),
        shared_dim: int = 512,
        decoder_hidden_dims: Sequence[int] | None = (256, 128),
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str = "layer",
        use_residual: bool = False,
    ) -> None:
        super().__init__()

        # Broadcast x_dims to all modalities
        if not isinstance(x_dims, Sequence):
            x_dims = [x_dims]
        if len(x_dims) == 1:
            x_dims = list(x_dims) * n_modals
        assert len(x_dims) == n_modals, (
            f"x_dims length ({len(x_dims)}) must match n_modals ({n_modals})"
        )
        self.n_modals = n_modals
        self.shared_dim = shared_dim

        # --- per-modality encoders --------------------------------------------
        self.encoders = nn.ModuleList([
            MLP(
                in_dim=x_dim,
                hidden_dims=list(encoder_hidden_dims),
                out_dim=shared_dim,
                activation=activation,
                dropout=dropout,
                norm=norm,
                residual=use_residual,
            )
            for x_dim in x_dims
        ])

        # --- optional per-modality decoders -----------------------------------
        if decoder_hidden_dims is not None:
            self.decoders = nn.ModuleList([
                MLP(
                    in_dim=shared_dim,
                    hidden_dims=list(decoder_hidden_dims),
                    out_dim=x_dim,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    residual=use_residual,
                )
                for x_dim in x_dims
            ])
        else:
            self.decoders = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, xs: List[Tensor]
    ) -> Tuple[Tensor, List[Tensor]]:
        """Encode all modalities and fuse into a shared embedding.

        Args:
            xs: List of modality tensors, each ``(B, d_proj)``.

        Returns:
            shared: ``(B, shared_dim)`` mean-pooled embedding.
            embeddings: List of ``(B, shared_dim)`` per-modality embeddings.
        """
        embeddings = [self.encoders[i](x) for i, x in enumerate(xs)]
        shared = torch.stack(embeddings, dim=0).mean(dim=0)  # (B, shared_dim)
        return shared, embeddings

    def encode_shared(self, xs: List[Tensor]) -> Tensor:
        """Convenience: return only the shared embedding."""
        shared, _ = self.forward(xs)
        return shared

    def encode_single(self, x: Tensor, modality_idx: int) -> Tensor:
        """Encode a single modality without fusion."""
        return self.encoders[modality_idx](x)

    def decode(self, shared: Tensor, modality_idx: int) -> Tensor:
        """Decode the shared embedding back to a modality observation."""
        if self.decoders is None:
            raise RuntimeError("SharedEmbeddingNet has no decoders.")
        return self.decoders[modality_idx](shared)

    def decode_all(self, shared: Tensor) -> List[Tensor]:
        """Decode shared embedding to all modalities."""
        if self.decoders is None:
            raise RuntimeError("SharedEmbeddingNet has no decoders.")
        return [self.decoders[i](shared) for i in range(self.n_modals)]

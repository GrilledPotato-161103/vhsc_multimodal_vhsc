"""LightningDataModule for the projected manifold dataset with physics targets.

Provides controlled covariate shift: train on ``z_src_range``, evaluate on
full ``z_range``.  The ``overlap_degree`` parameter controls coverage.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from src.data.components.physics_dataset import ProjectedManifoldDataset


def _resolve_range(
    z_range: Sequence[Tuple[float, float]] | Tuple[float, float],
    z_src_range: Sequence[Tuple[float, float]] | Tuple[float, float] | None,
    overlap_degree: float,
) -> Tuple[
    Sequence[Tuple[float, float]] | Tuple[float, float],
    Sequence[Tuple[float, float]] | Tuple[float, float],
]:
    """Resolve source range: use explicit ``z_src_range`` if given, otherwise
    shrink ``z_range`` by ``overlap_degree``."""
    if z_src_range is not None:
        return z_range, z_src_range

    if isinstance(z_range, tuple) and len(z_range) == 2 and isinstance(z_range[0], (int, float)):
        lo, hi = z_range
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * overlap_degree
        return z_range, (mid - half, mid + half)

    # Per-dimension ranges
    src = []
    for lo, hi in z_range:
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * overlap_degree
        src.append((mid - half, mid + half))
    return z_range, src


class ProjectedManifoldDataModule(L.LightningDataModule):
    """Data module for the projected high-dimensional manifold regression task.

    Creates four datasets:
    - ``src_dataset``: small reference set on ``z_src_range`` (for ICS fitting).
    - ``train_dataset``: large training set on ``z_src_range`` with noise.
    - ``val_dataset`` / ``test_dataset``: full ``z_range``, no noise.

    The same projection matrices are shared across all splits so modality
    semantics are consistent.
    """

    def __init__(
        self,
        n_samples: int = 128_000,
        n_src_samples: int = 256,
        emb_dim: int = 512,
        n_modals: int = 3,
        d_proj: int = 128,
        overlap_degree: float = 0.85,
        batch_size: int = 128,
        num_workers: int = 12,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        z_range: Sequence[Tuple[float, float]] | Tuple[float, float] = (-1.0, 1.0),
        z_src_range: Sequence[Tuple[float, float]] | Tuple[float, float] | None = None,
        noise_std: float = 0.1,
        physics: str = "lorenz96",
        physics_n_vars: int = 40,
        physics_n_steps: int = 500,
        physics_dt: float = 0.01,
        physics_burnin: int = 250,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.src_dataset: Optional[Dataset] = None
        self._shared_P: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _z_full(self):
        return self.hparams.z_range

    @property
    def _z_src(self):
        full, src = _resolve_range(
            self.hparams.z_range,
            self.hparams.z_src_range,
            self.hparams.overlap_degree,
        )
        return src

    def _make_dataset(self, n: int, zr, noise: float, **overrides) -> ProjectedManifoldDataset:
        """Create a ProjectedManifoldDataset, sharing projection matrices."""
        generator = torch.Generator().manual_seed(self.hparams.seed)
        return ProjectedManifoldDataset(
            n_samples=n,
            emb_dim=self.hparams.emb_dim,
            n_modals=self.hparams.n_modals,
            d_proj=self.hparams.d_proj,
            overlap_degree=self.hparams.overlap_degree,
            z_range=zr,
            noise_std=noise,
            physics=self.hparams.physics,
            physics_n_vars=self.hparams.physics_n_vars,
            physics_n_steps=self.hparams.physics_n_steps,
            physics_dt=self.hparams.physics_dt,
            physics_burnin=self.hparams.physics_burnin,
            projection_matrices=self._shared_P,
            generator=generator,
            seed=self.hparams.seed,
            **overrides,
        )

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        # Pre-generate projection matrices once so all splits are consistent
        g = torch.Generator().manual_seed(self.hparams.seed)
        self._shared_P = self._build_shared_P(g)

        n_total = self.hparams.n_samples
        n_val = int(n_total * self.hparams.val_ratio)
        n_test = int(n_total * self.hparams.test_ratio)
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            raise ValueError("Invalid split sizes — increase n_samples or reduce ratios.")

        # Source (reference) dataset — small, on z_src_range
        self.src_dataset = self._make_dataset(
            self.hparams.n_src_samples, self._z_src, noise=0.0
        )
        # Train — on z_src_range with noise
        self.train_dataset = self._make_dataset(
            n_train, self._z_src, noise=self.hparams.noise_std
        )
        # Val / Test — full z_range, no noise
        self.val_dataset = self._make_dataset(n_val, self._z_full, noise=0.0)
        self.test_dataset = self._make_dataset(n_test, self._z_full, noise=0.0)

    def _build_shared_P(self, generator: torch.Generator) -> torch.Tensor:
        """Build projection matrices without requiring a dataset instance."""
        emb_dim = self.hparams.emb_dim
        d_proj = self.hparams.d_proj
        overlap = self.hparams.overlap_degree
        n_modals = self.hparams.n_modals

        d_shared = int(d_proj * overlap)
        d_private = d_proj - d_shared

        W_shared = torch.randn(d_shared, emb_dim, generator=generator)
        W_shared = torch.linalg.qr(W_shared.T)[0].T

        matrices: list[torch.Tensor] = []
        for _ in range(n_modals):
            W_private = torch.randn(d_private, emb_dim, generator=generator)
            W_private = torch.linalg.qr(W_private.T)[0].T
            P_i = torch.cat([W_shared, W_private], dim=0)
            P_i = P_i * (3.0 / d_proj) ** 0.5
            matrices.append(P_i)
        return torch.stack(matrices, dim=0)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

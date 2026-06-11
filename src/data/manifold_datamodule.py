from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from src.data.components.dataset import ManifoldToyDataset

class ManifoldDataModule(L.LightningDataModule):
    """
    LightningDataModule for toy bi-modal regression:
        y = f(x1, x2)

    Example:
        dm = ToyBiModalDataModule(
            expression="x1**2 + 0.5*x2 + torch.sin(3*x1)",
            batch_size=128,
        )
    """

    def __init__(
        self,
        x_expressions: Sequence[str],
        y_expression: str,
        n_samples: int = 10000,
        n_src_samples: int = 512,
        batch_size: int = 64,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        z_range: Tuple[float, float] = (-1.0, 1.0),
        z_src_range: Tuple[float, float] = (-1.0, 1.0),
        z_dim: int = 2,
        noise_std: float = 0.0,
        noise_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return
        # n_train = * 
        # self.train_dataset = BiModalEquationDataset(
        #     n_samples=self.hparams.n_samples,
        #     expression=self.hparams.expression,
        #     x1_range=self.hparams.x1_range,
        #     x2_range=self.hparams.x2_range,
        #     noise_std=self.hparams.noise_std,
        #     noise_ratio=self.hparams.noise_ratio,
        #     seed=self.hparams.seed,
        # )

        n_total = self.hparams.n_samples
        n_val = int(n_total * self.hparams.val_ratio)
        n_test = int(n_total * self.hparams.test_ratio)
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                "Invalid split sizes. Ensure n_samples is large enough and "
                "val_ratio + test_ratio < 1."
            )
        generator = torch.Generator().manual_seed(self.hparams.seed)
        self.src_dataset = ManifoldToyDataset(  n_samples=self.hparams.n_src_samples,
                                                x_expressions=self.hparams.x_expressions,
                                                y_expression=self.hparams.y_expression,
                                                z_range=self.hparams.z_src_range,
                                                z_dim=self.hparams.z_dim,
                                                noise_std=self.hparams.noise_std,
                                                noise_ratio=self.hparams.noise_ratio,
                                                generator=generator,
                                                seed=self.hparams.seed,
                                                sampling="uniform"
                                            )
        
        self.train_dataset = ManifoldToyDataset(
                                                    n_samples=n_train,
                                                    x_expressions=self.hparams.x_expressions,
                                                    y_expression=self.hparams.y_expression,
                                                    z_range=self.hparams.z_range,
                                                    z_dim=self.hparams.z_dim,
                                                    noise_std=self.hparams.noise_std,
                                                    noise_ratio=self.hparams.noise_ratio,
                                                    generator=generator,
                                                    seed=self.hparams.seed,
                                                    sampling="uniform"
                                                    )
        self.val_dataset = ManifoldToyDataset(
                                                n_samples=n_train,
                                                x_expressions=self.hparams.x_expressions,
                                                y_expression=self.hparams.y_expression,
                                                z_range=self.hparams.z_range,
                                                z_dim=self.hparams.z_dim,
                                                noise_std=0.,
                                                noise_ratio=0.,
                                                generator=generator,
                                                seed=self.hparams.seed,
                                                sampling="uniform"
                                                )

        self.test_dataset = ManifoldToyDataset(
                                                    n_samples=n_train,
                                                    x_expressions=self.hparams.x_expressions,
                                                    y_expression=self.hparams.y_expression,
                                                    z_range=self.hparams.z_range,
                                                    z_dim=self.hparams.z_dim,
                                                    noise_std=0.,
                                                    noise_ratio=0.,
                                                    generator=generator,
                                                    seed=self.hparams.seed,
                                                    sampling="uniform"
                                                    )
        
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
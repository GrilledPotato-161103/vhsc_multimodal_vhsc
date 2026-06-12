from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.data.components.sampler import SortedBatchSampler
import numpy as np

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from src.data.components.dataset import ToyDataset

# @dataclass
# class ToyBiModalConfig:
#     n_samples: int = 10000
#     batch_size: int = 64
#     num_workers: int = 0

#     val_ratio: float = 0.1
#     test_ratio: float = 0.1

#     x1_range: Tuple[float, float] = (-1.0, 1.0)
#     x2_range: Tuple[float, float] = (-1.0, 1.0)

#     noise_std: float = 0.0
#     seed: int = 42


class ToyBiModalDataModule(L.LightningDataModule):
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
        expression: str,
        n_samples: int = 10000,
        n_src_samples: int = 1028,
        batch_size: int = 64,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        x1_range: Tuple[float, float] = (-1.0, 1.0),
        x2_range: Tuple[float, float] = (-1.0, 1.0),
        x1_src_range: Tuple[float, float] = (-1.0, 1.0),
        x2_src_range: Tuple[float, float] = (-1.0, 1.0),
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
        # self.train_dataset = ToyDataset(
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

        self.src_dataset = ToyDataset(
                                        n_samples=self.hparams.n_src_samples,
                                        expression=self.hparams.expression,
                                        x1_range=self.hparams.x1_src_range,
                                        x2_range=self.hparams.x2_src_range,
                                        noise_std=self.hparams.noise_std,
                                        noise_ratio=self.hparams.noise_ratio,
                                        generator=generator,
                                        seed=self.hparams.seed,
                                        sampling="normal"
                                        )
        
        self.train_dataset = ToyDataset(
                                        n_samples=n_train,
                                        expression=self.hparams.expression,
                                        x1_range=self.hparams.x1_range,
                                        x2_range=self.hparams.x2_range,
                                        noise_std=self.hparams.noise_std,
                                        noise_ratio=self.hparams.noise_ratio,
                                        generator=generator,
                                        seed=self.hparams.seed,
                                        sampling="uniform"
                                        )
        self.val_dataset = ToyDataset(
                                        n_samples=n_val,
                                        expression=self.hparams.expression,
                                        x1_range=self.hparams.x1_range,
                                        x2_range=self.hparams.x2_range,
                                        noise_std=0.,
                                        noise_ratio=0.,
                                        generator=generator,
                                        seed=self.hparams.seed,
                                        sampling="uniform"
                                        )

        self.test_dataset = ToyDataset(
                                        n_samples=n_test,
                                        expression=self.hparams.expression,
                                        x1_range=self.hparams.x1_range,
                                        x2_range=self.hparams.x2_range,
                                        noise_std=0.,
                                        noise_ratio=0.,
                                        generator=generator,
                                        seed=self.hparams.seed,
                                        sampling="uniform"
                                        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=SortedBatchSampler(self.train_dataset.x1, 
                                             batch_size=self.hparams.batch_size,
                                             shuffle=True),
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
    

# class ToyBiModalInjectDataModule(L.LightningDataModule):
#     """
#     LightningDataModule for toy bi-modal regression:
#         y = f(x1, x2)

#     Example:
#         dm = ToyBiModalDataModule(
#             expression="x1**2 + 0.5*x2 + torch.sin(3*x1)",
#             batch_size=128,
#         )
#     """

#     def __init__(
#         self,
#         expression: str,
#         bp_name: str,
#         n_samples: int = 10000,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         val_ratio: float = 0.1,
#         test_ratio: float = 0.1,
#         x1_range: Tuple[float, float] = (-1.0, 1.0),
#         x2_range: Tuple[float, float] = (-1.0, 1.0),
#         noise_std: float = 0.0,
#         seed: int = 42,
#         offrate: float = 0.3,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()

#         self.train_dataset: Optional[Dataset] = None
#         self.val_dataset: Optional[Dataset] = None
#         self.test_dataset: Optional[Dataset] = None

#     def setup(self, stage: Optional[str] = None) -> None:
#         if self.train_dataset is not None:
#             return

#         full_dataset = ToyDataset(
#             n_samples=self.hparams.n_samples,
#             expression=self.hparams.expression,
#             x1_range=self.hparams.x1_range,
#             x2_range=self.hparams.x2_range,
#             noise_std=self.hparams.noise_std,
#             seed=self.hparams.seed,
#         )

#         n_total = len(full_dataset)
#         n_val = int(n_total * self.hparams.val_ratio)
#         n_test = int(n_total * self.hparams.test_ratio)
#         n_train = n_total - n_val - n_test

#         if n_train <= 0:
#             raise ValueError(
#                 "Invalid split sizes. Ensure n_samples is large enough and "
#                 "val_ratio + test_ratio < 1."
#             )

#         split_generator = torch.Generator().manual_seed(self.hparams.seed)

#         self.train_dataset, self.val_dataset, self.test_dataset = random_split(
#             full_dataset,
#             [n_train, n_val, n_test],
#             generator=split_generator,
#         )

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             BPInjectDataset(self.train_dataset, self.hparams.bp_name, self.hparams.offrate),
#             batch_size=self.hparams.batch_size,
#             shuffle=True,
#             num_workers=self.hparams.num_workers,
#             pin_memory=True,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             BPInjectDataset(self.val_dataset, self.hparams.bp_name, self.hparams.offrate),
#             batch_size=self.hparams.batch_size,
#             shuffle=False,
#             num_workers=self.hparams.num_workers,
#             pin_memory=True,
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             BPInjectDataset(self.test_dataset, self.hparams.bp_name, self.hparams.offrate),
#             batch_size=self.hparams.batch_size,
#             shuffle=False,
#             num_workers=self.hparams.num_workers,
#             pin_memory=True,
#         )
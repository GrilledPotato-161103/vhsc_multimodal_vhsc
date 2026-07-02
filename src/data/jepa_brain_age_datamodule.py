"""Brain age estimation datamodule for Neuro-JEPA.

Produces paired 3D MRI volumes (T1w, T2w) as 5D tensors [B, C, D, H, W]
and continuous brain age labels.

Supports two modes:
  - "synthetic": random 3D volumes for testing the pipeline end-to-end.
  - "real": placeholder for real dataset (UK Biobank, ABCD, etc.).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L


class SyntheticBrainAgeDataset(Dataset):
    """Synthetic 3D brain MRI volumes for pipeline validation.

    Each sample is a dict with two modalities (simulating T1w/T2w)
    and a scalar brain age label.
    """

    def __init__(
        self,
        n_samples: int = 256,
        image_size: Tuple[int, int, int] = (96, 108, 96),
        in_chans: int = 1,
        age_range: Tuple[float, float] = (45.0, 85.0),
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.in_chans = in_chans
        self.age_range = age_range

        gen = torch.Generator().manual_seed(seed)
        self.ages = torch.rand(n_samples, generator=gen) * (age_range[1] - age_range[0]) + age_range[0]
        # Store parameters so each worker can regenerate deterministically
        self._seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Use a per-sample seed so each index always returns the same data
        gen = torch.Generator().manual_seed(self._seed + idx)
        d, h, w = self.image_size
        t1w = torch.randn(self.in_chans, d, h, w, generator=gen) * 0.5 + 0.5
        t2w = torch.randn(self.in_chans, d, h, w, generator=gen) * 0.5 + 0.5
        age = self.ages[idx]
        return {"t1w": t1w, "t2w": t2w}, age


class JEPABrainAgeDataModule(L.LightningDataModule):
    """LightningDataModule for brain age estimation with JEPA.

    Provides paired 3D MRI volumes and continuous age labels.
    """

    def __init__(
        self,
        mode: str = "synthetic",
        n_samples: int = 256,
        image_size: Tuple[int, int, int] = (96, 108, 96),
        in_chans: int = 1,
        age_range: Tuple[float, float] = (45.0, 85.0),
        batch_size: int = 4,
        num_workers: int = 4,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
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

        n_total = self.hparams.n_samples
        n_val = max(1, int(n_total * self.hparams.val_ratio))
        n_test = max(1, int(n_total * self.hparams.test_ratio))
        n_train = n_total - n_val - n_test

        common = dict(
            image_size=self.hparams.image_size,
            in_chans=self.hparams.in_chans,
            age_range=self.hparams.age_range,
            seed=self.hparams.seed,
        )

        if self.hparams.mode == "synthetic":
            self.train_dataset = SyntheticBrainAgeDataset(n_samples=n_train, **common)
            self.val_dataset = SyntheticBrainAgeDataset(
                n_samples=n_val, seed=self.hparams.seed + 1000, **{
                    k: v for k, v in common.items() if k != "seed"
                },
            )
            self.val_dataset._seed = self.hparams.seed + 1000
            self.test_dataset = SyntheticBrainAgeDataset(
                n_samples=n_test, seed=self.hparams.seed + 2000, **{
                    k: v for k, v in common.items() if k != "seed"
                },
            )
            self.test_dataset._seed = self.hparams.seed + 2000
        else:
            raise NotImplementedError(
                f"Real brain age dataset mode '{self.hparams.mode}' not implemented. "
                "Implement a Dataset subclass for your data source."
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

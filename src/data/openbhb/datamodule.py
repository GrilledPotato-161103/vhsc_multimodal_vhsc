"""Lightning DataModule for OpenBHB brain age regression.

Transforms are defined as Hydra-native configs under
``configs/data/transform/`` and instantiated at runtime — no hard-coded
transform pipelines.

Usage::

    # Standalone (load config by path)
    dm = OpenBHBDataModule(
        data_dir="data/openbhb", batch_size=4,
        train_transform="configs/data/transform/train/openbhb.yaml",
        val_transform="configs/data/transform/val/openbhb.yaml",
    )
    dm.setup()

    # Via Hydra app (pass DictConfig from composed config)
    cfg = OmegaConf.load("configs/data/transform/openbhb.yaml")
    dm = OpenBHBDataModule(
        data_dir="data/openbhb", batch_size=4,
        train_transform=cfg.train, val_transform=cfg.val,
    )
    dm.setup()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hydra.utils import instantiate

from src.data.openbhb.dataset import OpenBHBDataset

# Standard config directory for transform presets
_TRANSFORM_DIR = Path(__file__).resolve().parents[3] / "configs" / "data" / "transform"


def _resolve_transform(source: Union[str, DictConfig, None]) -> DictConfig:
    """Resolve a transform config from a path string, DictConfig, or None.

    - ``str``: path to a YAML file → load with OmegaConf.
    - ``DictConfig``: use directly.
    - ``None``: load the standard val/openbhb.yaml as fallback.
    """
    if source is None:
        path = _TRANSFORM_DIR / "val" / "openbhb.yaml"
        return OmegaConf.load(str(path))
    if isinstance(source, str):
        return OmegaConf.load(source)
    return source


class _LightningDataModule:
    """Base class resolved lazily to avoid triggering the
    torchmetrics→transformers→huggingface_hub import chain at module level."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):
        # Defer the actual base class resolution to instantiation time.
        if not hasattr(cls, '_L'):
            try:
                import lightning as _L
            except ImportError:
                import pytorch_lightning as _L
            cls._L = _L
        # Create instance from the resolved Lightning base
        obj = super(_LightningDataModule, cls).__new__(cls)
        cls._L.LightningDataModule.__init__(obj)
        return obj


class OpenBHBDataModule(_LightningDataModule):
    """DataModule for OpenBHB quasiraw brain age regression.

    Transforms are instantiated from Hydra YAML configs — see
    ``configs/data/transform/train/openbhb.yaml`` and
    ``configs/data/transform/val/openbhb.yaml``.

    Parameters
    ----------
    data_dir : str
        Root directory with ``train/`` and ``val/`` subdirectories.
    batch_size : int
    num_workers : int
    image_size : tuple
        (D, H, W) — only used for bare fallback transforms (not recommended).
    train_transform : str, DictConfig, or None
        Path to YAML or DictConfig for training MONAI Compose.
    val_transform : str, DictConfig, or None
        Path to YAML or DictConfig for validation MONAI Compose.
    n_train : int or None
        Cap training samples (None = all).
    n_val : int or None
        Cap validation samples (None = all).
    seed : int
    """

    def __init__(
        self,
        data_dir: str = "data/openbhb",
        batch_size: int = 4,
        num_workers: int = 0,
        image_size: Tuple[int, int, int] = (96, 108, 96),
        train_transform: Union[str, DictConfig, None] = None,
        val_transform: Union[str, DictConfig, None] = None,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        seed: int = 42,
    ):
        # LightningDataModule.__init__ is called by _LightningDataModule.__new__
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self._train_cfg_src = train_transform
        self._val_cfg_src = val_transform
        self.n_train = n_train
        self.n_val = n_val
        self.seed = seed

        self.train_dataset: Optional[OpenBHBDataset] = None
        self.val_dataset: Optional[OpenBHBDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        train_cfg = _resolve_transform(self._train_cfg_src)
        val_cfg = _resolve_transform(self._val_cfg_src)
        train_t = instantiate(train_cfg)
        val_t = instantiate(val_cfg)

        self.train_dataset = OpenBHBDataset(
            data_dir=self.data_dir, split="train",
            image_size=self.image_size, transform=train_t,
        )
        self.val_dataset = OpenBHBDataset(
            data_dir=self.data_dir, split="val",
            image_size=self.image_size, transform=val_t,
        )

        if self.n_train is not None:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(self.train_dataset), size=min(self.n_train, len(self.train_dataset)), replace=False)
            self.train_dataset.samples = [self.train_dataset.samples[i] for i in idx]
        if self.n_val is not None:
            rng = np.random.default_rng(self.seed + 1000)
            idx = rng.choice(len(self.val_dataset), size=min(self.n_val, len(self.val_dataset)), replace=False)
            self.val_dataset.samples = [self.val_dataset.samples[i] for i in idx]

        print(f"OpenBHB train samples: {len(self.train_dataset)}")
        print(f"OpenBHB val samples:   {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

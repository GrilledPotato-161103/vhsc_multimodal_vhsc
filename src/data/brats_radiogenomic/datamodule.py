"""LightningDataModule for RSNA-MICCAI BraTS Radiogenomic Classification.

Provides train/val/test dataloaders over NIfTI volumes converted from the
competition DICOM slices.  Splits are pre-computed in ``labels.csv`` — the
DataModule filters by the ``split`` column at setup time.

Uses dynamic subclassing to defer the ``lightning`` import past the
``torchmetrics -> transformers`` import chain, matching the pattern in
:class:`~src.data.openbhb.datamodule.OpenBHBDataModule`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hydra.utils import instantiate
from monai.transforms.compose import Compose

def collate_modality_dict(batch: List[tuple]) -> tuple:
    """Custom collate for ``({modality: tensor|None}, label)`` batches.

    PyTorch's default collate cannot ``torch.stack`` ``None`` values.
    This stacks tensor-valued modality keys and preserves ``None`` for keys
    where all samples are ``None``.  Mixed batches (some ``None``, some
    tensors) have ``None`` values replaced with a zero tensor matching the
    shape of the first non-``None`` tensor in the batch.
    """
    img_dicts, labels = zip(*batch)

    # Collate image dicts
    keys = img_dicts[0].keys()
    collated_imgs: Dict[str, Any] = {}
    for key in keys:
        values = [d[key] for d in img_dicts]
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        if not tensors:
            # All None
            collated_imgs[key] = None
        elif len(tensors) == len(values):
            # All tensors — fast path
            collated_imgs[key] = torch.stack(tensors, 0)
        else:
            # Mixed: fill None with zero tensor
            ref = tensors[0]
            filled = [
                v if isinstance(v, torch.Tensor)
                else torch.zeros_like(ref)
                for v in values
            ]
            collated_imgs[key] = torch.stack(filled, 0)

    collated_labels = torch.stack(labels, 0)
    return collated_imgs, collated_labels


class BraTSRadiogenomicDataModule:
    """DataModule for BraTS Radiogenomic MGMT classification (in-domain).

    Parameters
    ----------
    labels_csv : str
        Path to ``labels.csv`` produced by ``scripts/generate_mgmt_splits.py``.
    data_dir : str
        Root directory containing ``nifti/`` subdirectory with converted volumes.
    modality_keys : Sequence[str]
        Modality names.  Default ``["t1", "t2", "flair"]``.
    batch_size : int
    num_workers : int
    train_transform : str, DictConfig, or None
        Hydra-instantiable transform config for training.
    val_transform : str, DictConfig, or None
        Hydra-instantiable transform config for validation/test.
    """

    def __init__(
        self,
        labels_csv: str = "data/jepa/brats_radiogenomic/splits/labels.csv",
        data_dir: str = "data/jepa/brats_radiogenomic",
        modality_keys: Sequence[str] = ("t1", "t2", "flair"),
        batch_size: int = 1,
        num_workers: int = 0,
        train_transform: Optional[Compose] = None,
        val_transform: Optional[Compose] = None,
    ) -> None:
        # Lazy import — avoids torchmetrics -> transformers at module level
        try:
            import lightning as _L
        except ImportError:
            import pytorch_lightning as _L

        self.__class__ = type(
            "_BraTSRadiogenomicDataModule",
            (_L.LightningDataModule,),
            dict(BraTSRadiogenomicDataModule.__dict__),
        )
        _L.LightningDataModule.__init__(self)
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        from src.data.brats_radiogenomic.dataset import BraTSRadiogenomicDataset

        # train_t = instantiate(self.hparams.train_transform) if self.hparams.train_transform is not None else None
        # val_t = instantiate(self.hparams.val_transform) if self.hparams.val_transform is not None else None

        common = dict(
            labels_csv=self.hparams.labels_csv,
            data_dir=self.hparams.data_dir,
            modality_keys=self.hparams.modality_keys,
        )

        self.train_dataset = BraTSRadiogenomicDataset(
            split="train", transform=self.train_transform, **common
        )
        self.val_dataset = BraTSRadiogenomicDataset(
            split="val", transform=self.val_transform, **common
        )
        self.test_dataset = BraTSRadiogenomicDataset(
            split="test", transform=self.val_transform, **common
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_modality_dict,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_modality_dict,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_modality_dict,
        )

if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf, DictConfig
    import rootutils
    rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
    @hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
    def main(cfg: DictConfig) -> Optional[float]:
        # print(cfg)
        data: BraTSRadiogenomicDataModule = hydra.utils.instantiate(cfg.data)
        print(type(data))
        print(data.train_transform)
    main()
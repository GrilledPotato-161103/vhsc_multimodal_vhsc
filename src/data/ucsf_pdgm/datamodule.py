"""LightningDataModule for UCSF-PDGM-v3 Glioblastoma MRI.

Provides a single ``test_ood`` dataloader for out-of-domain evaluation.
No training or validation split — UCSF-PDGM is the held-out test cohort
disjoint from RSNA-MICCAI BraTS.

Uses dynamic subclassing (matching :class:`OpenBHBDataModule`) to defer
the ``lightning`` import.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

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

    keys = img_dicts[0].keys()
    collated_imgs: Dict[str, Any] = {}
    for key in keys:
        values = [d[key] for d in img_dicts]
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        if not tensors:
            collated_imgs[key] = None
        elif len(tensors) == len(values):
            collated_imgs[key] = torch.stack(tensors, 0)
        else:
            ref = tensors[0]
            filled = [
                v if isinstance(v, torch.Tensor)
                else torch.zeros_like(ref)
                for v in values
            ]
            collated_imgs[key] = torch.stack(filled, 0)

    collated_labels = torch.stack(labels, 0)
    return collated_imgs, collated_labels


class UCSFPDGMDataModule:
    """DataModule for UCSF-PDGM MGMT classification (out-of-domain).

    Parameters
    ----------
    labels_csv : str
        Path to ``labels.csv`` produced by ``scripts/generate_mgmt_splits.py``.
    data_dir : str
        Root directory containing NIfTI volumes.
    modality_keys : Sequence[str]
        Modality names.  Default ``["t1", "t2", "flair"]``.
    batch_size : int
    num_workers : int
    transform : Compose or None
        Pre-instantiated MONAI transform pipeline.
    """

    def __init__(
        self,
        labels_csv: str = "data/jepa/ucsf_pdgm/splits/labels.csv",
        data_dir: str = "data/jepa/ucsf_pdgm",
        modality_keys: Sequence[str] = ("t1", "t2", "flair"),
        batch_size: int = 1,
        num_workers: int = 0,
        transform: Optional[Compose] = None,
    ) -> None:
        try:
            import lightning as _L
        except ImportError:
            import pytorch_lightning as _L

        self.__class__ = type(
            "_UCSFPDGMDataModule",
            (_L.LightningDataModule,),
            dict(UCSFPDGMDataModule.__dict__),
        )
        _L.LightningDataModule.__init__(self)
        self.save_hyperparameters()

        self.test_dataset = None
        self.labels_csv = labels_csv
        self.data_dir = data_dir
        self.modality_keys = tuple(modality_keys)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        if self.test_dataset is not None:
            return

        from src.data.ucsf_pdgm.dataset import UCSFPDGMDataset

        self.test_dataset = UCSFPDGMDataset(
            labels_csv=self.labels_csv,
            data_dir=self.data_dir,
            split="test_ood",
            modality_keys=self.modality_keys,
            transform=self.transform,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_modality_dict,
        )

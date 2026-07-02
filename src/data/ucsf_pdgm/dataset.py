"""PyTorch Dataset for UCSF-PDGM-v3 glioblastoma MRI.

Loads native NIfTI volumes.  Each subject has 2 modalities (FLAIR + T1c)
mapped to the Neuro-JEPA modality topology ``["t1", "t2", "flair"]``.
The ``t2`` slot is always empty — it is filled at the classifier hook by
:class:`~src.plugins.reconstructor.pool.MeanImputeReconstructor`.

Expected data layout::

    data/jepa/ucsf_pdgm/
    ├── splits/
    │   └── labels.csv              # subject_id, mgmt_label, split, t1/t2/flair_path
    ├── UCSF-PDGM-0004_T1gad_bias.nii
    ├── UCSF-PDGM-0004_FLAIR_bias.nii
    └── ...
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class UCSFPDGMDataset(Dataset):
    """UCSF-PDGM-v3 MGMT classification dataset (out-of-domain).

    Each sample returns ``({modality: tensor | None}, mgmt_label)``.
    ``t2`` is always ``None`` — the ``MeanImputeReconstructor`` fills it
    at inference time.

    Parameters
    ----------
    labels_csv : str or Path
        Path to the unified ``labels.csv`` produced by
        ``scripts/generate_mgmt_splits.py``.
    data_dir : str or Path
        Root directory of the dataset (e.g. ``data/jepa/ucsf_pdgm/``).
        Modality paths in the CSV are resolved relative to this directory.
        Also falls back to ``data_dir / "raw" / path`` for files in a
        ``raw/`` subdirectory.
    split : str
        Which partition to load: ``"test_ood"`` (primary), or ``"train"`` /
        ``"val"`` / ``"test"`` if the dataset is later split for fine-tuning.
    modality_keys : Sequence[str]
        Fixed modality names.  Default ``("t1", "t2", "flair")``.
    transform : callable or None
        Optional MONAI-style ``Compose`` operating on keys
        ``[t1, t2, flair, label]`` with ``allow_missing_keys=True``.
    """

    def __init__(
        self,
        labels_csv: str | Path,
        data_dir: str | Path,
        split: str = "test_ood",
        modality_keys: Sequence[str] = ("t1", "t2", "flair"),
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality_keys = list(modality_keys)
        self.transform = transform

        # Read and filter labels.csv
        self.samples: List[Dict[str, str]] = []
        with open(labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append(row)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}' in {labels_csv}. "
                "Run scripts/generate_mgmt_splits.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor]:
        row = self.samples[idx]
        mgmt_label = int(row["mgmt_label"])

        # Build sample dict for MONAI transform — exclude None modalities
        # (MONAI transforms like Resized choke on None values even with
        # allow_missing_keys, which only guards against missing dict keys.)
        sample: Dict[str, torch.Tensor | int] = {"label": mgmt_label}
        missing: List[str] = []
        for mod in self.modality_keys:
            rel_path = row.get(f"{mod}_path", "").strip()
            tensor = self._load_nifti(rel_path) if rel_path else None
            if tensor is not None:
                sample[mod] = tensor
            else:
                missing.append(mod)

        if self.transform is not None:
            sample = self.transform(sample)

        # Build image dict — restore missing modalities as None
        image_dict: Dict[str, Optional[torch.Tensor]] = {}
        for mod in self.modality_keys:
            t = sample.get(mod)
            image_dict[mod] = t if isinstance(t, torch.Tensor) else None

        label = torch.as_tensor(
            sample.get("label", mgmt_label), dtype=torch.long
        )

        return image_dict, label

    def _load_nifti(self, rel_path: str) -> Optional[torch.Tensor]:
        """Load a NIfTI file and return a ``[C, D, H, W]`` float32 tensor.

        Returns ``None`` if the file does not exist.
        Tries ``data_dir / rel_path`` first, then ``data_dir / "raw" / rel_path``.
        """
        import nibabel as nib
        import numpy as np

        candidates = [
            self.data_dir / rel_path,
            self.data_dir / "raw" / rel_path,
        ]
        full = None
        for c in candidates:
            if c.exists():
                full = c
                break

        if full is None:
            return None

        arr = nib.load(str(full)).get_fdata(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]  # (D, H, W) -> (1, D, H, W)
        return torch.from_numpy(arr.copy())

"""PyTorch Dataset for OpenBHB quasiraw brain MRI images.

Loads .npy files downloaded from the HuggingFace benoit-dufumier/openBHB dataset.
Each file is a preprocessed T1w brain volume in MNI152 space.

Expected data layout:
    data/openbhb/
    ├── participants.tsv
    ├── train/
    │   ├── sub-*_preproc-quasiraw_T1w.npy
    │   └── ...
    ├── val/
    │   ├── sub-*_preproc-quasiraw_T1w.npy
    │   └── ...
    ├── train_metadata.csv
    └── val_metadata.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OpenBHBDataset(Dataset):
    """Dataset of OpenBHB quasiraw T1w images for brain age regression.

    Each sample returns (image_tensor [1, D, H, W], age_scalar).

    Parameters
    ----------
    data_dir : Path
        Root data directory (e.g. data/openbhb/).
    split : str
        One of "train" or "val".
    image_size : tuple
        Target spatial size (D, H, W). Default (96, 108, 96) for JEPA ViT.
    transform : callable or None
        Optional MONAI-style transform to apply to each sample.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/openbhb",
        split: str = "train",
        image_size: Tuple[int, int, int] = (96, 108, 96),
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # Load split metadata
        meta_path = self.data_dir / f"{split}_metadata.csv"
        if not meta_path.exists():
            alt_meta = self.data_dir / "participants.tsv"
            if alt_meta.exists():
                df = pd.read_csv(alt_meta, sep="\t")
                if split == "train":
                    df = df[df.split == "train"]
                else:
                    df = df[df.split.isin(["external_test", "internal_test"])]
                self.meta = df.reset_index(drop=True)
            else:
                raise FileNotFoundError(
                    f"Neither {meta_path} nor {alt_meta} found. "
                    "Run scripts/download_openbhb_samples.py first."
                )
        else:
            self.meta = pd.read_csv(meta_path)

        # Build list of (file_path, age) tuples
        self.samples: List[Tuple[Path, float]] = []
        for _, row in self.meta.iterrows():
            file_path = self.data_dir / split / f"sub-{int(row['participant_id'])}_preproc-quasiraw_T1w.npy"
            if file_path.exists():
                self.samples.append((file_path, float(row["age"])))
            elif "file" in row and Path(row["file"]).exists():
                self.samples.append((Path(row["file"]), float(row["age"])))
            else:
                # Try glob match
                candidates = list((self.data_dir / split).glob(f"sub-{int(row['participant_id'])}_*.npy"))
                if candidates:
                    self.samples.append((candidates[0], float(row["age"])))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .npy files found for split='{split}' in {self.data_dir / split}. "
                "Run scripts/download_openbhb_samples.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, age = self.samples[idx]

        # Load: shape (1, 1, 182, 218, 182) float64 → squeeze to (1, 182, 218, 182) float32
        arr = np.load(str(file_path))
        arr = arr.squeeze(axis=0).astype(np.float32)

        # Remove NaN, clip outliers, min-max normalize
        arr = np.nan_to_num(arr, nan=0.0)
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        arr = np.clip(arr, lo, hi)
        denom = max(hi - lo, 1e-8)
        arr = (arr - lo) / denom

        # Build sample dict for MONAI transform
        sample = {"image": arr, "label": np.array(age, dtype=np.float32)}

        if self.transform is not None:
            sample = self.transform(sample)

        # Ensure correct shape: [C, D, H, W]
        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # add channel dim

        age_tensor = torch.as_tensor(sample["label"], dtype=torch.float32)

        return image, age_tensor

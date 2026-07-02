"""Download pretrained Neuro-JEPA backbone and save to data/checkpoints/jepa/.

Usage:
  python scripts/download_jepa_checkpoint.py

The script builds a MultiModalJEPARegressor with a pretrained ViT backbone
from HuggingFace Hub (NYUMedML/Neuro-JEPA) and saves it for use with the
hook DAG training pipeline.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.components.jpa import build_jepa_regressor


def main():
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "checkpoints", "jepa", "jepa_pretrained.pth"
    )
    out_path = os.path.abspath(out_path)

    if os.path.exists(out_path):
        print(f"Checkpoint already exists at {out_path}")
        print("Delete it first to re-download.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building JEPA regressor on {device} ...")

    regressor = build_jepa_regressor(
        model_name_or_path="NYUMedML/Neuro-JEPA",
        device=device,
        modality_keys=["t1w", "t2w"],
        image_size=(96, 108, 96),
        num_classes=1,  # brain age regression
        freeze_backbone=True,
    )

    torch.save(regressor, out_path)
    print(f"Saved pretrained model to {out_path}")
    print(f"Model structure:\n{regressor}")


if __name__ == "__main__":
    main()

"""Thin callable for MONAI Lambdad — referenced by Hydra configs via _target_.

The heavy lifting (resize, augmentation, tensor conversion) is defined in
``configs/data/transform/openbhb.yaml`` and instantiated by Hydra.  This
module only supplies the one operation that cannot be expressed in pure YAML:
ensuring the numpy array has a channel dimension before MONAI processes it.
"""

from __future__ import annotations

import numpy as np


def to_monai_compatible(arr: np.ndarray) -> np.ndarray:
    """Ensure a channel dim exists so MONAI reads spatial dims correctly.

    Dataset outputs may be ``(D, H, W)`` after squeezing.  MONAI's spatial
    transforms need ``(C, D, H, W)`` with explicit C ≥ 1, otherwise ``img.shape[1:]``
    reports one too few spatial dimensions.

    Also coerces to float32 for downstream tensor consistency.
    """
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]  # (D, H, W) → (1, D, H, W)
    return arr.astype(np.float32)

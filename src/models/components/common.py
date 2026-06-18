"""Shared building-block factories for neural network components.

Provides unified activation and normalization factory functions that can be
used across all model component files. Consolidates the duplicated implementations
previously found in ffn.py and toy.py.
"""

from typing import Literal, Optional

from torch import nn


def get_activation(name: Optional[str]) -> nn.Module:
    """Build an activation module from a string identifier.

    Args:
        name: Activation name. Supported: "relu", "gelu", "silu"/"swish",
              "tanh", "leaky_relu". None, "none", or "identity" returns Identity.

    Returns:
        The corresponding nn.Module.

    Raises:
        ValueError: If the activation name is not recognised.
    """
    if name is None or name.lower() in {"none", "identity"}:
        return nn.Identity()

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU(inplace=False)
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=False)
    if name == "tanh":
        return nn.Tanh(inplace=False)
    if name == "leaky_relu":
        return nn.LeakyReLU(inplace=False)
    raise ValueError(f"Unsupported activation: {name}")


def get_normalization(
    name: Optional[str],
    num_features: int,
    dimension: Literal[1, 2, 3] = 1,
    num_groups: int = 2,
    **kwargs,
) -> nn.Module:
    """Build a normalisation module from a string identifier.

    Args:
        name: Normalisation name. Supported: "batch", "layer", "group".
              None returns Identity.
        num_features: Number of features / channels to normalise.
        dimension: Spatial dimension (1/2/3), used for BatchNorm selection.
        num_groups: Number of groups for GroupNorm (default 2).
        **kwargs: Additional keyword arguments forwarded to the normalisation
                  constructor (e.g. ``eps``, ``momentum``).

    Returns:
        The corresponding nn.Module.
    """
    if name is None:
        return nn.Identity()

    if name == "batch":
        bn_cls = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
        if dimension not in bn_cls:
            raise ValueError(
                f"BatchNorm requires dimension 1, 2, or 3, got {dimension}."
            )
        return bn_cls[dimension](num_features=num_features, **kwargs)

    if name == "layer":
        return nn.LayerNorm(normalized_shape=num_features, **kwargs)

    if name == "group":
        return nn.GroupNorm(
            num_groups=num_groups, num_channels=num_features, **kwargs
        )

    return nn.Identity()

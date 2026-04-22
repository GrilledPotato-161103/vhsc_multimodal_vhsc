import torch
import torch.nn as nn
from typing import Optional, Sequence, Literal

def build_activation(name: Optional[str]) -> nn.Module:
    if name is None or name.lower() in {"none", "identity"}:
        return nn.Identity()

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


def get_normalization(
    name: Optional[Literal["batch", "layer", "group"]], 
    num_features: int,
    dimension: Literal[1, 2, 3] = 1,
    **kwargs
) -> nn.Module:
    """
    Returns a PyTorch normalization module explicitly using a dimension argument.
    
    Args:
        name: The name of the normalization layer. If None, returns nn.Identity().
        num_features: The number of features/channels to normalize.
        dimension: The spatial dimension of the input (1, 2, or 3). Primarily used for BatchNorm.
        **kwargs: Extra arguments (like num_groups for GroupNorm).
    """
    if name is None:
        return nn.Identity()
        
    if name == "batch":
        if dimension == 1:
            return nn.BatchNorm1d(num_features=num_features, **kwargs)
        elif dimension == 2:
            return nn.BatchNorm2d(num_features=num_features, **kwargs)
        elif dimension == 3:
            return nn.BatchNorm3d(num_features=num_features, **kwargs)
        else:
            raise ValueError(f"I'm completely unsure how to create a BatchNorm for dimension {dimension}.")
            
    if name == "layer":
        # LayerNorm takes normalized_shape, which is usually just the feature dimension
        return nn.LayerNorm(normalized_shape=num_features, **kwargs)
        
    if name == "group":
        # GroupNorm requires 'num_groups', defaulting to 32 if not provided in kwargs
        num_groups = kwargs.pop("num_groups", 32)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features, **kwargs)
    
    return nn.Identity()


class FeedForward(nn.Module):
    """
    Linear is always first.
    Only A/D/N ordering is configurable.

    A = Activation
    D = Dropout
    N = Normalization

    Example:
        adn_order = "ADN"
        adn_order = "NDA"
    """

    _allowed_order = {"a", "d", "n"}

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        activation: Optional[str] = "gelu",
        norm: Optional[str] = None,
        dropout: float = 0.0,
        adn_order: str = "adn",
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self._validate_adn_order(adn_order)

        self.block1 = self._make_stage(
            in_dim=in_dim,
            out_dim=hidden_dim,
            activation=activation,
            norm=norm,
            dropout=dropout,
            adn_order=adn_order,
        )
        self.block2 = self._make_stage(
            in_dim=hidden_dim,
            out_dim=out_dim,
            activation=activation,
            norm=norm,
            dropout=dropout,
            adn_order=adn_order,
        )

    @classmethod
    def _validate_adn_order(cls, adn_order: str) -> None:
        for op in adn_order:
            if op not in cls._allowed_order:
                raise ValueError(
                    f"Unknown op '{op}'. Allowed chars: {sorted(cls._allowed_order)}"
                )

    def _make_stage(
        self,
        in_dim: int,
        out_dim: int,
        activation: Optional[str],
        norm: Optional[str],
        dropout: float,
        adn_order: str,
    ) -> nn.Sequential:
        print(dropout, activation, norm, out_dim)
        ops = {
            "a": build_activation(activation),
            "d": nn.Dropout(dropout),
            "n": get_normalization(norm, num_features=out_dim, dimension=1),
        }

        layers = [nn.Linear(in_dim, out_dim)]
        layers.extend(ops[ch] for ch in adn_order)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x
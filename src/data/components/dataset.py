from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

class ToyDataset(Dataset):
    """
    Toy dataset for y = f(x1, x2).

    Each item is:
        ((x1, x2), y)

    where x1, x2, y are torch.Tensor scalars or vectors depending on feature_dim.
    """

    def __init__(
        self,
        n_samples: int,
        expression: str,
        x1_range: Tuple[float, float] = (-1.0, 1.0),
        x2_range: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.0,
        noise_ratio: float = 0.5,
        generator: torch.Generator | None = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        sampling: str = "uniform"
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.expression = expression
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.noise_ratio = noise_ratio
        self.noise_std = noise_std
        self.dtype = dtype
        g = torch.Generator().manual_seed(seed) if not generator else generator
        # Sử dụng Normal distribution để thể hiện rõ hơn về mean và variance thực tế
        l1, r1 = x1_range
        l2, r2 = x2_range
        if sampling == "normal":
            self.x1 = torch.randn([n_samples], generator=g) * (r1 - l1) + l1
            self.x2 = torch.randn([n_samples], generator=g) * (r2 - l2) + l2
        else:
            self.x1 = torch.empty((n_samples,)).uniform_(l1, r1, generator=g)
            self.x2 = torch.empty((n_samples,)).uniform_(l2, r2, generator=g)
            
        indexes = torch.bernoulli(torch.full((n_samples,), noise_ratio)).int()

        # Uniform để tối đa hóa entropy
        augment = lambda x: torch.where(indexes > 0, x + noise_std * torch.empty_like(x).uniform_(-1, 1, generator=g), x)
        self.y = self._evaluate_expression(self.x1, self.x2)
        if noise_std > 0:
            # Augmenting y seems to break the cycle
            self.x1 = augment(self.x1)
            self.x2 = augment(self.x2)
            # self.y = augment(self.y)

        if self.y.ndim == 0:
            self.y = self.y.unsqueeze(0)

        if self.y.shape[0] != n_samples:
            raise ValueError(
                f"Expression must produce one value per sample. "
                f"Got output shape {tuple(self.y.shape)} for n_samples={n_samples}."
            )

    def _evaluate_expression(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Evaluate expression like:
            'x1**2 + 2*x2 + torch.sin(x1)'
        in a restricted namespace.
        """
        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "x1": x1,
            "x2": x2,
            "torch": torch,
            "math": math,
        }

        try:
            y = eval(self.expression, safe_globals, safe_locals)
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate expression: {self.expression!r}. Error: {e}"
            ) from e

        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=self.dtype)

        return y.to(self.dtype)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        y = self.y[idx]
        return (x1, x2), y, torch.Tensor([[x1, x2]])

class ManifoldToyDataset(Dataset): 
    def __init__(
        self,
        n_samples: int,
        x_expressions: Sequence[str],
        y_expression: str, 
        z_range: Tuple[float, float] = (-1.0, 1.0),
        z_dim: int = 2,
        noise_std: float = 0.0,
        noise_ratio: float = 0.5,
        generator: torch.Generator | None = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        sampling: str = "uniform"
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.x_expressions = x_expressions
        self.y_expression = y_expression
        self.z_range = z_range
        self.z_dim = z_dim
        self.y_expression = y_expression
        self.noise_ratio = noise_ratio
        self.noise_std = noise_std
        self.dtype = dtype

        g = torch.Generator().manual_seed(seed) if not generator else generator
        # Sử dụng Normal distribution để thể hiện rõ hơn về mean và variance thực tế
        lz, rz = z_range
        if sampling == "normal":
            self.z = torch.randn([self.z_dim, n_samples], generator=g) * (rz - lz) + lz
        else:
            self.z = torch.empty((self.z_dim, n_samples)).uniform_(lz, rz, generator=g)
        
        indexes = torch.bernoulli(torch.full((n_samples,), noise_ratio)).int()
        # Uniform để tối đa hóa entropy
        augment = lambda x: torch.where(indexes > 0, x + noise_std * torch.empty_like(x).uniform_(-1, 1, generator=g), x)
        self.xs = [self._evaluate_expression(x_exp, self.z) for x_exp in self.x_expressions]
        self.y = self._evaluate_expression(self.y_expression, self.z)
        print("Modality shapes:", [x.shape for x in self.xs])
        self.z = self.z.permute(1, 0)
        if noise_std > 0:
            # Augmenting y seems to break the cycle
            self.xs = [augment(x) for x in self.xs]

        if self.y.ndim <= 1:
            self.y = self.y.unsqueeze(1)
            self.xs = [x.unsqueeze(1) for x in self.xs]
            
        if self.y.shape[0] != n_samples:
            raise ValueError(
                f"Expression must produce one value per sample. "
                f"Got output shape {tuple(self.y.shape)} for n_samples={n_samples}."
            )

    def _evaluate_expression(self, expr: str, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate expression like:
            'x1**2 + 2*x2 + torch.sin(x1)'
        in a restricted namespace.
        """
        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "x": x,
            "torch": torch,
            "math": math,
        }

        try:
            y = eval(expr, safe_globals, safe_locals)
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate expression: {expr!r}. Error: {e}"
            ) from e

        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=self.dtype)
        return y.to(self.dtype)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        xs = [x[idx] for x in self.xs]
        y = self.y[idx]
        z = self.z[idx]
        return xs, y, z
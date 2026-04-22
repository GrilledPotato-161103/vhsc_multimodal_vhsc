from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L


# @dataclass
# class ToyBiModalConfig:
#     n_samples: int = 10000
#     batch_size: int = 64
#     num_workers: int = 0

#     val_ratio: float = 0.1
#     test_ratio: float = 0.1

#     x1_range: Tuple[float, float] = (-1.0, 1.0)
#     x2_range: Tuple[float, float] = (-1.0, 1.0)

#     noise_std: float = 0.0
#     seed: int = 42


class BiModalEquationDataset(Dataset):
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
        return (x1, x2), y

class BPInjectDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 bp_name: str,
                 rate: float =  0.3):
        super().__init__()
        self.dataset = dataset
        self.bp_name = bp_name
        self.rate = rate 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        inputs, y = self.dataset[idx]
        signal = [1, 1]
        index = np.random.choice(3, p=[1 - self.rate, self.rate / 2, self.rate / 2])
        if index > 0:
            inputs = list(inputs)
            signal[index - 1] = 0
            inputs[index - 1] = torch.rand_like(inputs[index - 1])
        return tuple(inputs), y, {self.bp_name : torch.Tensor([signal])}



class ToyBiModalDataModule(L.LightningDataModule):
    """
    LightningDataModule for toy bi-modal regression:
        y = f(x1, x2)

    Example:
        dm = ToyBiModalDataModule(
            expression="x1**2 + 0.5*x2 + torch.sin(3*x1)",
            batch_size=128,
        )
    """

    def __init__(
        self,
        expression: str,
        n_samples: int = 10000,
        batch_size: int = 64,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        x1_range: Tuple[float, float] = (-1.0, 1.0),
        x2_range: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.0,
        noise_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return
        # n_train = * 
        # self.train_dataset = BiModalEquationDataset(
        #     n_samples=self.hparams.n_samples,
        #     expression=self.hparams.expression,
        #     x1_range=self.hparams.x1_range,
        #     x2_range=self.hparams.x2_range,
        #     noise_std=self.hparams.noise_std,
        #     noise_ratio=self.hparams.noise_ratio,
        #     seed=self.hparams.seed,
        # )

        n_total = self.hparams.n_samples
        n_val = int(n_total * self.hparams.val_ratio)
        n_test = int(n_total * self.hparams.test_ratio)
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                "Invalid split sizes. Ensure n_samples is large enough and "
                "val_ratio + test_ratio < 1."
            )
        generator = torch.Generator().manual_seed(self.hparams.seed)

        self.train_dataset = BiModalEquationDataset(
                                                    n_samples=n_train,
                                                    expression=self.hparams.expression,
                                                    x1_range=self.hparams.x1_range,
                                                    x2_range=self.hparams.x2_range,
                                                    noise_std=self.hparams.noise_std,
                                                    noise_ratio=self.hparams.noise_ratio,
                                                    generator=generator,
                                                    seed=self.hparams.seed,
                                                    sampling="uniform"
                                                    )
        self.val_dataset = BiModalEquationDataset(
                                                    n_samples=n_val,
                                                    expression=self.hparams.expression,
                                                    x1_range=self.hparams.x1_range,
                                                    x2_range=self.hparams.x2_range,
                                                    noise_std=0.,
                                                    noise_ratio=0.,
                                                    generator=generator,
                                                    seed=self.hparams.seed,
                                                    )

        self.test_dataset = BiModalEquationDataset(
                                                    n_samples=n_test,
                                                    expression=self.hparams.expression,
                                                    x1_range=self.hparams.x1_range,
                                                    x2_range=self.hparams.x2_range,
                                                    noise_std=0.,
                                                    noise_ratio=0.,
                                                    generator=generator,
                                                    seed=self.hparams.seed,
                                                    )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
    

class ToyBiModalInjectDataModule(L.LightningDataModule):
    """
    LightningDataModule for toy bi-modal regression:
        y = f(x1, x2)

    Example:
        dm = ToyBiModalDataModule(
            expression="x1**2 + 0.5*x2 + torch.sin(3*x1)",
            batch_size=128,
        )
    """

    def __init__(
        self,
        expression: str,
        bp_name: str,
        n_samples: int = 10000,
        batch_size: int = 64,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        x1_range: Tuple[float, float] = (-1.0, 1.0),
        x2_range: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.0,
        seed: int = 42,
        offrate: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        full_dataset = BiModalEquationDataset(
            n_samples=self.hparams.n_samples,
            expression=self.hparams.expression,
            x1_range=self.hparams.x1_range,
            x2_range=self.hparams.x2_range,
            noise_std=self.hparams.noise_std,
            seed=self.hparams.seed,
        )

        n_total = len(full_dataset)
        n_val = int(n_total * self.hparams.val_ratio)
        n_test = int(n_total * self.hparams.test_ratio)
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                "Invalid split sizes. Ensure n_samples is large enough and "
                "val_ratio + test_ratio < 1."
            )

        split_generator = torch.Generator().manual_seed(self.hparams.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=split_generator,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            BPInjectDataset(self.train_dataset, self.hparams.bp_name, self.hparams.offrate),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            BPInjectDataset(self.val_dataset, self.hparams.bp_name, self.hparams.offrate),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            BPInjectDataset(self.test_dataset, self.hparams.bp_name, self.hparams.offrate),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
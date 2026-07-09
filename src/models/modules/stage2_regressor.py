"""Stage 2 LightningModule: downstream regression on frozen Stage 1 encoders.

Loads a pre-trained SharedEmbeddingNet, freezes its encoders, and trains a
lightweight regression head to predict the physics-based target y.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmetrics import MeanMetric, MinMetric

from lightning import LightningModule

from src.models.components.shared_embedding import SharedEmbeddingNet


class ManifoldRegressorModule(LightningModule):
    """Regression head on top of frozen shared-embedding encoders.

    Expected batch: ``(xs_noisy, y, xs_clean, z)`` from ProjectedManifoldDataset.
    """

    def __init__(
        self,
        stage1_net: SharedEmbeddingNet,
        regressor_head: nn.Module,
        optimizer: Any,
        scheduler: Any | None = None,
        freeze_encoders: bool = True,
        compile_model: bool = False,
        loss_name: Literal["mse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["stage1_net", "regressor_head"])

        self.stage1_net = stage1_net
        self.regressor_head = regressor_head

        if freeze_encoders:
            for param in self.stage1_net.parameters():
                param.requires_grad = False

        self.loss_name = loss_name
        self.huber_delta = huber_delta

        # Regression metrics
        self.val_rmse = MeanMetric()
        self.val_rmse_best = MinMetric()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self.hparams.compile_model and stage == "fit":
            self.regressor_head = torch.compile(self.regressor_head)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, xs: List[Tensor]) -> Tensor:
        shared, _ = self.stage1_net(xs)       # (B, shared_dim)
        y_hat = self.regressor_head(shared)    # (B, out_dim) or (B,)
        if y_hat.ndim == 2 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        return y_hat

    # ------------------------------------------------------------------
    # Loss & Metrics
    # ------------------------------------------------------------------

    def _compute_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y = y.view_as(y_hat)
        if self.loss_name == "mse":
            return F.mse_loss(y_hat, y)
        if self.loss_name == "mae":
            return F.l1_loss(y_hat, y)
        if self.loss_name == "huber":
            return F.huber_loss(y_hat, y, delta=self.huber_delta)
        raise ValueError(f"Unsupported loss_name: {self.loss_name}")

    @staticmethod
    def _compute_metrics(y_hat: Tensor, y: Tensor) -> Dict[str, Tensor]:
        y = y.view_as(y_hat)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        rmse = torch.sqrt(mse)

        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) ** 2)
        ss_res = torch.sum((y - y_hat) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def model_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        xs_noisy, y, _xs_clean, _z = batch
        modalities = [xs_noisy[:, i, :] for i in range(xs_noisy.shape[1])]
        y_hat = self.forward(modalities)
        loss = self._compute_loss(y_hat, y)
        return loss, y_hat, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        loss, y_hat, y = self.model_step(batch)
        metrics = self._compute_metrics(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mae", metrics["mae"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/rmse", metrics["rmse"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/r2", metrics["r2"], on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        loss, y_hat, y = self.model_step(batch)
        metrics = self._compute_metrics(y_hat, y)
        self.val_rmse(metrics["rmse"])

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", metrics["mae"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/r2", metrics["r2"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        score = self.val_rmse.compute()
        self.val_rmse_best(score)
        self.log("val/rmse_best", self.val_rmse_best.compute(), sync_dist=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        loss, y_hat, y = self.model_step(batch)
        metrics = self._compute_metrics(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mae", metrics["mae"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/rmse", metrics["rmse"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/r2", metrics["r2"], on_step=False, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any] | torch.optim.Optimizer:
        # Only optimize the regressor head (encoders are frozen)
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters())
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/rmse_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

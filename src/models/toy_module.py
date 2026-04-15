from __future__ import annotations

from typing import Any, Dict, Tuple, Literal

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric, MinMetric


from lightning import LightningModule
from hydra.utils import instantiate
class BiModalLightningModule(LightningModule):
    """
    Hydra-compatible LightningModule for bimodal regression.

    Expected batch format:
        ((x1, x2), y)

    Expected model signature:
        y_hat = model(x1, x2)
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: Any,
        scheduler: Any | None = None,
        compile_model: bool = False,
        loss_name: Literal["mse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()

        self.net = net

        self.compile_model = compile_model
        self.loss_name = loss_name
        self.huber_delta = huber_delta

        self.val_rmse_best = MinMetric()
        self.val_rmse = MeanMetric()

        self.save_hyperparameters(logger=False, ignore=["net"])

    def setup(self, stage: str | None = None) -> None:
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.net(x1, x2)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.view_as(y_hat)
        if self.loss_name == "mse":
            return F.mse_loss(y_hat, y)
        if self.loss_name == "mae":
            return F.l1_loss(y_hat, y)
        if self.loss_name == "huber":
            return F.huber_loss(y_hat, y, delta=self.huber_delta)

        raise ValueError(f"Unsupported loss_name: {self.loss_name}")

    @staticmethod
    def _compute_metrics(y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = y.view_as(y_hat)

        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        rmse = torch.sqrt(mse)

        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) ** 2)
        ss_res = torch.sum((y - y_hat) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    def model_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x1, x2), y = batch
        y_hat = self.forward(x1, x2)

        if y_hat.ndim == 2 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)

        loss = self._compute_loss(y_hat, y)
        return loss, y_hat, y

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, y_hat, y = self.model_step(batch)
        metrics = self._compute_metrics(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mae", metrics["mae"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/rmse", metrics["rmse"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/r2", metrics["r2"], on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, y_hat, y = self.model_step(batch)
        metrics = self._compute_metrics(y_hat, y)
        self.val_rmse(metrics["rmse"])

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", metrics["mae"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/r2", metrics["r2"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        score = self.val_rmse.compute()  # get current val acc
        self.val_rmse_best(score)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/rmse_best", self.val_rmse_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, y_hat, y = self.model_step(batch)
        metrics = self._compute_metrics(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mae", metrics["mae"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/rmse", metrics["rmse"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/r2", metrics["r2"], on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], tuple):
            (x1, x2), _ = batch
        else:
            x1, x2 = batch

        y_hat = self.forward(x1, x2)
        if y_hat.ndim == 2 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        return y_hat

    def configure_optimizers(self) -> Dict[str, Any] | torch.optim.Optimizer:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

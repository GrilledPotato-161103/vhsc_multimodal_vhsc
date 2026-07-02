"""LightningModule for brain age estimation with JEPA hooks.

Trains a reconstructor and uncertainty estimator on top of a frozen
pretrained Neuro-JEPA ViT backbone via the Breakpoint DAG system.

Breakpoint structure (all encoders + final output layer):
  src_enc0    — after encoders.0  (T1w ViT tokens [B, N, 768])
  src_enc1    — after encoders.1  (T2w ViT tokens [B, N, 768])
  reconstructor — before classifier (cross-modal reconstruction)
  uncertainty   — after classifier  (BayesCap on brain age prediction)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

import rootutils

rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.plugins.hook_dag import Breakpoint, BreakpointController
from src.plugins.head.bayescap import BayesCap1DLoss, bayescap_variance_1d
from src.models.hook_modules.common import check_gradient


class JEPABrainAgeModule(LightningModule):
    """Train reconstructor + uncertainty head on frozen JEPA backbone for brain age."""

    def __init__(
        self,
        net: nn.Module,
        recon_bp: str,
        unc_bp: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        controller: BreakpointController | None = None,
        compile: bool = False,
        recon_criterion: nn.Module | Callable | None = None,
        unc_criterion: nn.Module | Callable | None = None,
        epoch_phase: int = 10,
        mask_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "controller", "recon_criterion", "unc_criterion"],
        )
        self.net = net
        self.controller = controller

        # --- Metrics ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.test_recon_loss = MeanMetric()

        self.train_unc_loss = MeanMetric()
        self.val_unc_loss = MeanMetric()
        self.test_unc_loss = MeanMetric()

        self.train_id = MeanMetric()
        self.val_id = MeanMetric()
        self.test_id = MeanMetric()

        self.train_nll = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()

        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()
        self.test_mae = MeanMetric()

        self.val_nll_best = MinMetric()

        self.criterion = nn.MSELoss(reduction="none")
        self.recon_criterion = recon_criterion or nn.MSELoss(reduction="none")
        self.unc_criterion = unc_criterion or BayesCap1DLoss(
            lambda_identity=1.0,
            lambda_nll=0.05,
            identity_mode="l2",
            nll_mode="paper",
        )

    def forward(self, images: Dict[str, torch.Tensor] | List[torch.Tensor]) -> torch.Tensor:
        return self.net(images)

    def on_train_start(self):
        self.controller.train()
        return super().on_train_start()

    def model_step(
        self, batch: Tuple, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """Single step: forward through frozen JEPA + hooked reconstructor + uncertainty.

        Batch format: (images_dict, y) where images_dict maps modality→[B,C,D,H,W].
        """
        self.net.eval()
        self.net.requires_grad_(False)

        images, y = batch
        y = y.view(-1, 1).float()

        # --- Random modality masking for reconstruction training ---
        if "bp_signal" in kwargs:
            bp_signal = kwargs["bp_signal"]
        else:
            mask_index = np.random.choice(
                3, 1,
                p=(1 - self.hparams.mask_rate,
                   self.hparams.mask_rate / 2,
                   self.hparams.mask_rate / 2),
            )[0]
            bp_signal = [1, 1]
            if mask_index > 0:
                bp_signal[mask_index - 1] = 0

        recon_bp = Breakpoint.get_by_name(self.hparams.recon_bp)
        recon_bp.kwargs = tuple(bp_signal)

        logits = self.forward(images)

        # Regression loss (brain age prediction MSE)
        loss = self.criterion(logits, y)
        mae = torch.abs(logits - y).mean()

        # --- Reconstruction loss from trace ---
        recon_trace = recon_bp.trace
        sigs = recon_trace.trace["signal"]
        recs = recon_trace.trace["reconstructed"]
        srcs = recon_trace.trace["input"]
        devs = recon_trace.trace["dev"]
        dists = recon_trace.trace["distance"]

        recon_loss = torch.tensor(0.0, device=self.device)
        recon_unc_loss = torch.tensor(0.0, device=self.device)
        for sig, rec, src, dev, dist in zip(sigs, recs, srcs, devs, dists):
            if sig == 0:
                continue
            recon_loss += self.recon_criterion(rec, src).mean()
            recon_unc_loss += self.criterion(dev, dist).mean()

        # --- Uncertainty loss from BayesCap ---
        unc_trace = Breakpoint.get_by_name(self.hparams.unc_bp).trace
        mu, alpha, beta = unc_trace.trace["output"]
        variance = bayescap_variance_1d(alpha, beta, target_dim=1, eps=1e-6)
        unc_loss = self.unc_criterion(mu, alpha, beta, logits, y)

        return (
            loss, logits, y,
            {"recon_loss": recon_loss, "unc_loss": recon_unc_loss, "trace": recon_trace, "mae": mae},
            {"mu": mu, "var": variance, "loss": unc_loss["loss"],
             "identity": unc_loss["identity_loss"], "nll": unc_loss["nll_loss"]},
        )

    def training_step(self, batch, batch_idx):
        loss, logits, y, recon, unc = self.model_step(batch, bp_signal=(1, 1))
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"

        self.train_loss(loss.mean())
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_recon_loss(recon["recon_loss"])
        self.log(f"train/loss_recon_{signal_str}", self.train_recon_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_unc_loss(recon["unc_loss"])
        self.log(f"train/loss_unc_{signal_str}", self.train_unc_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_id(unc["identity"].mean())
        self.log(f"train/loss_id_{signal_str}", self.train_id,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_nll(unc["nll"].mean())
        self.log(f"train/loss_nll_{signal_str}", self.train_nll,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_mae(recon["mae"])
        self.log("train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)

        # Phase 1: don't propagate uncertainty for masked inputs
        if self.current_epoch < self.hparams.epoch_phase and sum(signal) < 2:
            unc["loss"] *= 0

        return loss.mean() + recon["unc_loss"].mean() + unc["loss"].mean()

    def on_validation_start(self):
        self.controller.eval()
        return super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        loss, logits, _, recon, unc = self.model_step(batch, bp_signal=(1, 1))
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"

        self.val_loss(loss.mean())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_recon_loss(recon["recon_loss"])
        self.log(f"val/loss_recon_{signal_str}", self.val_recon_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_unc_loss(recon["unc_loss"])
        self.log(f"val/loss_unc_{signal_str}", self.val_unc_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_id(unc["identity"].mean())
        self.log(f"val/loss_id_{signal_str}", self.val_id,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_nll(unc["nll"].mean())
        self.log(f"val/loss_nll_{signal_str}", self.val_nll,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_mae(recon["mae"])
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        score = self.val_nll.compute()
        self.val_nll_best(score)
        self.log("val/loss_nll_best", self.val_nll_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"

        self.test_loss(loss.mean())
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_recon_loss(recon["recon_loss"])
        self.log(f"test/loss_recon_{signal_str}", self.test_recon_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.test_unc_loss(recon["unc_loss"])
        self.log(f"test/loss_unc_{signal_str}", self.test_unc_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.test_id(unc["identity"].mean())
        self.log(f"test/loss_id_{signal_str}", self.test_id,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.test_nll(unc["nll"].mean())
        self.log(f"test/loss_nll_{signal_str}", self.test_nll,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.test_mae(recon["mae"])
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str | None = None):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        parameters = []
        for item in self.controller.breakpoints:
            bp = item["breakpoint"]
            if isinstance(bp.callback, nn.Module):
                parameters += list(bp.callback.parameters())

        optimizer = self.hparams.optimizer(params=parameters)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_nll_11",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

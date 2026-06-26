"""LightningModule for JEPA multi-modal hook training.

Trains reconstruction and uncertainty estimation hooks attached to a
frozen Neuro-JEPA classification backbone via
:class:`~src.plugins.hook_dag.BreakpointController`.

Training phases
---------------
1. **Reconstruction phase** (epoch < ``epoch_phase``):
   Only the reconstructor callback parameters are trained.  Modalities
   are randomly masked to force cross-modal reconstruction.

2. **Uncertainty phase** (epoch >= ``epoch_phase``):
   Both the reconstructor and BayesCap uncertainty head are trained.

The frozen backbone is NEVER updated — only breakpoint callbacks
(reconstructor + BayesCap1D) receive gradient updates.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.plugins.hook_dag import BreakpointController, Breakpoint
from src.plugins.head.bayescap import BayesCap1DLoss, bayescap_variance_1d
from src.models.hook_modules.common import HuberLoss, check_gradient


class JEPAHookModule(LightningModule):
    """Hook-training LightningModule for JEPA multi-modal classification.

    Parameters
    ----------
    net:
        Frozen :class:`~src.models.components.jepa.MultiModalJEPARegressor`.
    controller:
        :class:`BreakpointController` with attached breakpoints.
    recon_bp:
        Name of the reconstruction breakpoint (e.g. ``"reconstructor.0"``).
    unc_bp:
        Name of the uncertainty breakpoint (e.g. ``"uncertainty.0"``).
    optimizer:
        Partial optimizer constructor.
    scheduler:
        Partial scheduler constructor.
    compile:
        Whether to ``torch.compile`` the backbone (default ``False``).
    recon_criterion:
        Loss for reconstruction quality (default: ``MSELoss``).
    unc_criterion:
        Loss for uncertainty estimation (default: ``BayesCap1DLoss``).
    clf_criterion:
        Classification loss (default: ``CrossEntropyLoss``).
    epoch_phase:
        Epoch at which uncertainty training begins.
    mask_rate:
        Probability of masking a modality during training.
    n_modals:
        Number of modalities.
    """

    def __init__(
        self,
        net: nn.Module,
        recon_bp: str,
        unc_bp: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler | None = None,
        controller: BreakpointController | None = None,
        compile: bool = False,
        recon_criterion: nn.Module | Callable | None = None,
        unc_criterion: nn.Module | Callable | None = None,
        clf_criterion: nn.Module | Callable | None = None,
        epoch_phase: int = 20,
        mask_rate: float = 0.3,
        n_modals: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "controller", "recon_criterion", "unc_criterion", "clf_criterion"],
        )
        self.net = net
        self.controller = controller

        # Clamp n_modals from the actual wrapper
        if hasattr(self.net, "n_modals"):
            self.hparams.n_modals = self.net.n_modals
        self._num_classes = getattr(self.net, "num_classes", 2)

        # --- metrics -----------------------------------------------------------
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

        self.train_acc = Accuracy(task="multiclass", num_classes=self._num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self._num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self._num_classes)

        self.val_nll_best = MinMetric()

        # --- loss functions ----------------------------------------------------
        self.recon_criterion = recon_criterion or nn.MSELoss(reduction="none")
        self.unc_criterion = unc_criterion or BayesCap1DLoss(
            lambda_identity=1.0,
            lambda_nll=0.05,
            identity_mode="l2",
            nll_mode="paper",
            reduction="mean",
        )
        self.clf_criterion = clf_criterion or nn.CrossEntropyLoss(reduction="none")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: Dict[str, torch.Tensor] | List[torch.Tensor]) -> torch.Tensor:
        """Run forward pass through the hooked JEPA model."""
        return self.net(images)

    # ------------------------------------------------------------------
    # Model step
    # ------------------------------------------------------------------

    def model_step(
        self, batch: Any, bp_signal: Tuple[int, ...] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """Single forward + loss computation step."""
        self.net.eval()  # backbone stays in eval
        self.net.requires_grad_(False)

        # Unpack batch
        if isinstance(batch, (list, tuple)):
            images_dict = batch[0] if isinstance(batch[0], dict) else batch[0]
            y = batch[1]
        else:
            images_dict = batch
            y = torch.zeros(1)

        if isinstance(y, torch.Tensor):
            if y.ndim > 1:
                y = y.squeeze()
            if y.dtype != torch.long:
                y = y.long()

        n = self.hparams.n_modals

        # --- Modality masking --------------------------------------------------
        if bp_signal is not None:
            signal = list(bp_signal)
        else:
            mask_index = np.random.choice(
                n + 1, 1,
                p=[1 - self.hparams.mask_rate]
                + [self.hparams.mask_rate / n] * n,
            )[0]
            signal = [1] * n
            if mask_index > 0:
                signal[mask_index - 1] = 0

        recon_bp = Breakpoint.get_by_name(self.hparams.recon_bp)
        recon_bp.kwargs = tuple(signal)

        # --- Forward pass through hooked model --------------------------------
        logits = self.forward(images_dict)

        # --- Classification loss -----------------------------------------------
        clf_loss = self.clf_criterion(logits, y)

        # --- Reconstruction trace ----------------------------------------------
        recon_trace = recon_bp.trace
        sigs = recon_trace.trace["signal"]
        recs = recon_trace.trace["reconstructed"]
        srcs = recon_trace.trace["input"]
        devs = recon_trace.trace["dev"]
        dists = recon_trace.trace["distance"]

        recon_loss = torch.tensor(0.0, device=logits.device)
        recon_unc_loss = torch.tensor(0.0, device=logits.device)

        for sig, rec, src, dev, dist in zip(sigs, recs, srcs, devs, dists):
            if sig == 0:
                continue
            # Mean-pool for loss computation (token dim)
            rec_pool = rec.mean(dim=1) if rec.dim() == 3 else rec
            src_pool = src.mean(dim=1) if src.dim() == 3 else src
            recon_loss = recon_loss + self.recon_criterion(rec_pool, src_pool)
            recon_unc_loss = recon_unc_loss + self.recon_criterion(dev, dist)

        # --- Uncertainty trace -------------------------------------------------
        unc_trace = Breakpoint.get_by_name(self.hparams.unc_bp).trace
        mu, alpha, beta = unc_trace.trace["output"]

        variance = bayescap_variance_1d(alpha, beta, target_dim=self._num_classes, eps=1e-6)
        unc_loss = self.unc_criterion(mu, alpha, beta, logits, y)

        # --- Total loss --------------------------------------------------------
        total_loss = clf_loss.mean() + recon_loss.mean() + unc_loss["loss"]

        return (
            total_loss,
            logits,
            y,
            {
                "recon_loss": recon_loss,
                "unc_loss": recon_unc_loss,
                "trace": recon_trace,
                "clf_loss": clf_loss,
                "signal": signal,
            },
            {
                "mu": mu,
                "var": variance,
                "loss": unc_loss["loss"],
                "identity": unc_loss["identity_loss"],
                "nll": unc_loss["nll_loss"],
            },
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        self.controller.train()
        return super().on_train_start()

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["signal"]
        signal_str = "".join(str(int(s)) for s in signal)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_recon_loss(recon["recon_loss"].mean())
        self.log(f"train/loss_recon_{signal_str}", self.train_recon_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_unc_loss(recon["unc_loss"].mean())
        self.log(f"train/loss_unc_{signal_str}", self.train_unc_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_id(unc["identity"].mean())
        self.log(f"train/loss_id_{signal_str}", self.train_id,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_nll(unc["nll"].mean())
        self.log(f"train/loss_nll_{signal_str}", self.train_nll,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_acc(logits, y)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # Phase 1: suppress uncertainty loss for partial inputs
        if self.current_epoch < self.hparams.epoch_phase and sum(signal) < self.hparams.n_modals:
            unc["loss"] = unc["loss"] * 0

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure) -> None:
        if batch_idx == 1:
            print(f"Checking gradient for frozen model {self.net.__class__.__qualname__}")
            check_gradient(self.net)
            for item in self.controller.breakpoints:
                pos, bp = item["position"], item["breakpoint"]
                if isinstance(bp.callback, nn.Module):
                    print(f"Checking {bp.name} module on {pos}: {bp.callback.__class__.__qualname__}")
                    check_gradient(bp.callback)
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_start(self) -> None:
        self.controller.eval()
        super().on_validation_start()

    def validation_step(self, batch: Tuple, batch_idx: int) -> Tuple:
        loss, logits, y, recon, unc = self.model_step(
            batch, bp_signal=tuple([1] * self.hparams.n_modals)
        )
        signal = recon["signal"]
        signal_str = "".join(str(int(s)) for s in signal)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_recon_loss(recon["recon_loss"].mean())
        self.log(f"val/loss_recon_{signal_str}", self.val_recon_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_unc_loss(recon["unc_loss"].mean())
        self.log(f"val/loss_unc_{signal_str}", self.val_unc_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_id(unc["identity"].mean())
        self.log(f"val/loss_id_{signal_str}", self.val_id,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_nll(unc["nll"].mean())
        self.log(f"val/loss_nll_{signal_str}", self.val_nll,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc(logits, y)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss, logits, recon, unc

    def on_validation_epoch_end(self) -> None:
        score = self.val_nll.compute()
        self.val_nll_best(score)
        self.log("val/loss_nll_best", self.val_nll_best.compute(),
                 sync_dist=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["signal"]
        signal_str = "".join(str(int(s)) for s in signal)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_recon_loss(recon["recon_loss"].mean())
        self.log(f"test/loss_recon_{signal_str}", self.test_recon_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.test_unc_loss(recon["unc_loss"].mean())
        self.log(f"test/loss_unc_{signal_str}", self.test_unc_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.test_id(unc["identity"].mean())
        self.log(f"test/loss_id_{signal_str}", self.test_id,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.test_nll(unc["nll"].mean())
        self.log(f"test/loss_nll_{signal_str}", self.test_nll,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc(logits, y)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Setup & Optimizers
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        parameters = list(self.trainer.model.parameters())
        for item in self.controller.breakpoints:
            bp = item["breakpoint"]
            if isinstance(bp.callback, nn.Module):
                print(f"Assigning {bp.name} breakpoints to Optimizer for update")
                parameters = parameters + list(bp.callback.parameters())

        optimizer = self.hparams.optimizer(params=parameters)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_nll_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

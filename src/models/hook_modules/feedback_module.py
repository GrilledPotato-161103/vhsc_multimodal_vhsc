"""FeedbackInjectModule — two-phase training with gradient flow through reconstructor.

Phase 1 (prefill): encoders run, source BPs push outputs to reconstructor,
reconstructor runs and pushes processed latents to mutator BPs' buffers.
Phase 2 (mutate): mutator BPs emit reconstructed values (true in-place
replacement), model's forward uses processed latents through all non-Module
computation, head produces prediction, loss back-propagates through the
reconstructor into the encoders.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric, MinMetric

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.plugins.hook_dag import Breakpoint, BreakpointController
from src.plugins.head.bayescap import BayesCap1DLoss, bayescap_variance_1d
from src.models.hook_modules.common import check_gradient


class FeedbackInjectModule(LightningModule):
    """Two-phase training module with gradient flow through reconstructor.

    Unlike the single-pass :class:`ModelInjectModule`, this module runs the
    model forward twice per step:

    1. **Prefill** (``controller.phase("prefill")``): collect encoder outputs
       and run reconstruction.  Mutator BPs pass through.
    2. **Mutate** (``controller.phase("mutate")``): mutator BPs emit
       reconstructed values.  The model's native forward uses the processed
       latents.  Reconstructor passes through.
    """

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
        epoch_phase: int = 20,
        mask_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["recon_criterion", "unc_criterion", "net", "controller"],
        )
        self.net = net
        self.controller = controller
        self.criterion = torch.nn.MSELoss(reduction="none")
        self.recon_criterion = recon_criterion or nn.MSELoss()
        self.unc_criterion = unc_criterion or nn.MSELoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.train_unc_loss = MeanMetric()
        self.val_unc_loss = MeanMetric()
        self.train_nll = MeanMetric()
        self.val_nll = MeanMetric()
        self.val_nll_best = MinMetric()

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return self.net(xs)

    def on_train_start(self):
        self.controller.train()
        return super().on_train_start()

    def on_validation_start(self):
        self.controller.eval()
        return super().on_validation_start()

    def _random_mask(self) -> Tuple[int, int]:
        mask_index = np.random.choice(
            3, 1,
            p=(1 - self.hparams.mask_rate,
               self.hparams.mask_rate / 2,
               self.hparams.mask_rate / 2),
        )[0]
        signal = [1, 1]
        if mask_index > 0:
            signal[mask_index - 1] = 0
        return tuple(signal)

    def _compute_recon_loss(self, recon_trace):
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
            recon_loss = recon_loss + self.recon_criterion(rec, src)
            recon_unc_loss = recon_unc_loss + self.criterion(dev, dist)
        return recon_loss, recon_unc_loss

    def _compute_unc_loss(self, logits, y):
        unc_bp = Breakpoint.get_by_name(self.hparams.unc_bp)
        mu, alpha, beta = unc_bp.trace.trace["output"]
        variance = bayescap_variance_1d(alpha, beta, target_dim=1, eps=1e-6)
        unc_loss = self.unc_criterion(mu, alpha, beta, logits, y)
        return {
            "mu": mu,
            "var": variance,
            "loss": unc_loss["loss"],
            "identity": unc_loss["identity_loss"],
            "nll": unc_loss["nll_loss"],
        }

    def model_step(self, batch, **kwargs):
        xs, y, _, _ = batch
        signal = kwargs.get("bp_signal", self._random_mask())

        recon_bp = Breakpoint.get_by_name(self.hparams.recon_bp)
        recon_bp.kwargs = tuple(signal)

        # ── Phase 1: Prefill ──
        # Source BPs push z0,z1 to reconstructor._buffer
        # Reconstructor runs, pushes rec_0,rec_1 to mutator BPs' _buffer
        # Mutator BPs pass through (phase="prefill")
        with self.controller.phase("prefill"):
            _ = self.forward(torch.split(xs, 1, dim=1))

        # ── Phase 2: Mutate ──
        # Mutator BPs read _buffer (populated in prefill), emit rec_0,rec_1
        # Model: zs = [rec_0, rec_1] → torch.stack → sum → head → pred
        # Reconstructor passes through (doesn't overwrite mutator buffers)
        with self.controller.phase("mutate"):
            logits = self.forward(torch.split(xs, 1, dim=1)).unsqueeze(1)

        # ── Loss ──
        loss = self.criterion(logits, y)
        recon_loss, recon_unc_loss = self._compute_recon_loss(recon_bp.trace)
        unc = self._compute_unc_loss(logits, y)

        recon_info = {
            "recon_loss": recon_loss,
            "unc_loss": recon_unc_loss,
            "trace": recon_bp.trace,
        }
        return loss, logits, y, recon_info, unc

    def training_step(self, batch, batch_idx):
        loss, logits, y, recon, unc = self.model_step(batch, bp_signal=(1, 1))
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"

        self.train_loss(loss.mean())
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_recon_loss(recon["recon_loss"].mean())
        self.log(f"train/loss_recon_{signal_str}", self.train_recon_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_unc_loss(recon["unc_loss"].mean())
        self.log(f"train/loss_unc_{signal_str}", self.train_unc_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.train_nll(unc["nll"].mean())
        self.log(f"train/loss_nll_{signal_str}", self.train_nll,
                 on_step=True, on_epoch=True, prog_bar=True)

        if self.current_epoch < self.hparams.epoch_phase and sum(signal) < 2:
            unc["loss"] = unc["loss"] * 0

        return loss.mean() + recon["unc_loss"].mean() + unc["loss"].mean()

    def validation_step(self, batch, batch_idx):
        loss, logits, _, recon, unc = self.model_step(batch, bp_signal=(1, 1))
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.val_recon_loss(recon["recon_loss"].mean())
        self.log(f"val/loss_recon_{signal_str}", self.val_recon_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.val_unc_loss(recon["unc_loss"].mean())
        self.log(f"val/loss_unc_{signal_str}", self.val_unc_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.val_nll(unc["nll"].mean())
        self.log(f"val/loss_nll_{signal_str}", self.val_nll,
                 on_step=True, on_epoch=True, prog_bar=True)

        return loss, logits, recon, unc

    def on_validation_epoch_end(self):
        score = self.val_nll.compute()
        self.val_nll_best(score)
        self.log("val/loss_nll_best", self.val_nll_best.compute(),
                 sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"
        self.log(f"test/loss_{signal_str}", loss.mean(), on_step=False, on_epoch=True)
        self.log(f"test/loss_nll_{signal_str}", unc["nll"].mean(), on_step=False, on_epoch=True)
        return loss, logits, recon, unc

    def configure_optimizers(self) -> Dict[str, Any]:
        parameters = list(self.trainer.model.parameters())
        for item in self.controller.breakpoints:
            bp = item["breakpoint"]
            if isinstance(bp.callback, nn.Module):
                parameters = parameters + list(bp.callback.parameters())

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if batch_idx == 1:
            self._check_gradients()
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def _check_gradients(self):
        for item in self.controller.breakpoints:
            bp = item["breakpoint"]
            if isinstance(bp.callback, nn.Module):
                check_gradient(bp.callback)

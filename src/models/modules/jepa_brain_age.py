"""JEPABrainAgeModule — two-phase brain age regression with Neuro-JEPA.

Phase 1 (epochs 0 .. unfreeze_at - 1):
    Frozen ViT backbone.  Only the AttentiveClassifier regressor is trained.

Phase 2 (epochs unfreeze_at .. max_epochs):
    Backbone unfrozen.  Both backbone and regressor are trained, each with
    its own optimizer and learning rate.

AttentiveRegressor
    A learnable query token cross-attends to the ViT token grid [B, N, 768]
    and projects to a scalar age, following Neuro-JEPA's downstream architecture.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: build the attentive regressor from the Neuro-JEPA codebase
# ---------------------------------------------------------------------------

def build_attentive_regressor(
    embed_dim: int = 768,
    num_heads: int = 16,
    depth: int = 1,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
) -> nn.Module:
    """Build an AttentiveClassifier configured for scalar regression.

    Mirrors the downstream head used in Neuro-JEPA's finetuning scripts
    (``scripts/finetune/default.py`` and ``scripts/finetune/tte.py``).

    Architecture:
      - 1 learnable query token  [1, 1, embed_dim]
      - CrossAttentionBlock (query attends to ViT token grid)
      - optional self-attention blocks (if depth > 1)
      - Linear(embed_dim -> 1)  scalar output
    """
    _ensure_neurojepa_importable()
    from neurojepa.models.attentive_pooler import AttentiveClassifier

    return AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        num_classes=1,
        complete_block=True,
        use_activation_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class JEPABrainAgeModule(LightningModule):
    """Two-phase brain age regression with attentive pooling.

    Parameters
    ----------
    backbone : nn.Module
        Pretrained vit_base (with MoE) from Neuro-JEPA.
    regressor : nn.Module
        AttentiveClassifier(num_classes=1) — learnable query cross-attention.
    backbone_optimizer : partial
        Optimizer for ViT backbone (e.g. AdamW, lr=1e-5).
    regressor_optimizer : partial
        Optimizer for the attentive regressor (e.g. AdamW, lr=1e-4).
    backbone_scheduler : partial or None
    regressor_scheduler : partial or None
    unfreeze_at_epoch : int
        Epoch at which the backbone is unfrozen (phase 1 -> phase 2).
    freeze_backbone_initially : bool
        If True (default), start with frozen backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        regressor: nn.Module,
        backbone_optimizer: Any,
        regressor_optimizer: Any,
        backbone_scheduler: Any | None = None,
        regressor_scheduler: Any | None = None,
        unfreeze_at_epoch: int = 10,
        freeze_backbone_initially: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["backbone", "regressor"],
        )
        self.backbone = backbone
        self.regressor = regressor

        # Phase tracking
        self._phase = 1
        self._backbone_frozen = freeze_backbone_initially
        if freeze_backbone_initially:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # Disable automatic optimization — we manage two optimizers manually
        self.automatic_optimization = False

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()
        self.val_mae_best = MinMetric()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict brain age.

        x : [B, 1, D, H, W]  3D T1w volume
        returns : [B]         predicted age
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]

        tokens = self.backbone(x)             # [B, N, 768]  (or (tokens, moe_scores))
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        out = self.regressor(tokens)          # [B, 1]
        return out.squeeze(-1)                # [B]

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    @staticmethod
    def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        rmse = torch.sqrt(mse)
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        unfreeze_at = self.hparams.unfreeze_at_epoch

        if epoch == unfreeze_at and self._backbone_frozen:
            log.info(">>> Phase 2: unfreezing backbone at epoch %d", epoch)
            self.backbone.requires_grad_(True)
            self.backbone.train()
            self._backbone_frozen = False
            self._phase = 2

        phase = 2 if not self._backbone_frozen else 1
        if phase != self._phase:
            self._phase = phase

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        opt_bb, opt_reg = self.optimizers()
        sch_bb, sch_reg = self._schedulers_or_none()

        # In phase 1 only the regressor optimizer steps; in phase 2 both do.
        should_step_bb = not self._backbone_frozen

        # --- forward ---
        y_hat = self.forward(x)
        loss = self._compute_loss(y_hat, y)

        # --- manual backward ---
        opt_bb.zero_grad()
        opt_reg.zero_grad()
        self.manual_backward(loss)

        # clip gradients
        self.clip_gradients(opt_bb, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        self.clip_gradients(opt_reg, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        # step optimizer(s)
        if should_step_bb:
            opt_bb.step()
        opt_reg.step()

        # --- logging ---
        metrics = self._compute_metrics(y_hat, y)
        self.train_loss(loss)
        self.train_mae(metrics["mae"])

        pfx = f"train/p{self._phase}"
        self.log(f"{pfx}/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{pfx}/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{pfx}/rmse", metrics["rmse"], on_step=True, on_epoch=True)
        self.log(f"{pfx}/r2", metrics["r2"], on_step=True, on_epoch=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True)

        # log LRs
        self.log("lr/backbone", opt_bb.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("lr/regressor", opt_reg.param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        loss = self._compute_loss(y_hat, y)
        metrics = self._compute_metrics(y_hat, y)
        self.val_loss(loss)
        self.val_mae(metrics["mae"])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", metrics["rmse"], on_step=False, on_epoch=True)
        self.log("val/r2", metrics["r2"], on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.val_mae_best(self.val_mae.compute())
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        loss = self._compute_loss(y_hat, y)
        metrics = self._compute_metrics(y_hat, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mae", metrics["mae"], on_step=False, on_epoch=True)
        self.log("test/rmse", metrics["rmse"], on_step=False, on_epoch=True)
        self.log("test/r2", metrics["r2"], on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_bb = self.hparams.backbone_optimizer(params=self.backbone.parameters())
        opt_reg = self.hparams.regressor_optimizer(params=self.regressor.parameters())

        optimizers = [opt_bb, opt_reg]
        schedulers = []

        if self.hparams.backbone_scheduler is not None:
            schedulers.append(self.hparams.backbone_scheduler(optimizer=opt_bb))
        if self.hparams.regressor_scheduler is not None:
            schedulers.append(self.hparams.regressor_scheduler(optimizer=opt_reg))

        if schedulers:
            return optimizers, schedulers
        return optimizers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedulers_or_none(self):
        """Return (sch_bb_or_None, sch_reg_or_None) for manual scheduler stepping."""
        try:
            lr_config = self.trainer.lr_scheduler_configs
            sch_bb = lr_config[0].scheduler if len(lr_config) > 0 else None
            sch_reg = lr_config[1].scheduler if len(lr_config) > 1 else None
        except Exception:
            sch_bb = sch_reg = None
        return sch_bb, sch_reg


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _ensure_neurojepa_importable() -> None:
    _root = Path(__file__).resolve().parents[3]
    _nj = str(_root / "submodules" / "Neuro-JEPA" / "src")
    if _nj not in sys.path:
        sys.path.insert(0, _nj)


def build_backbone(
    checkpoint_path: str = "NYUMedML/Neuro-JEPA",
    device: str | torch.device = "cpu",
    **kwargs,
) -> nn.Module:
    """Load a pretrained Neuro-JEPA ViT backbone via the official API.

    Delegates to :func:`src.models.components.jepa.load_backbone`, which wraps
    ``neurojepa.utils.init_utils.load_backbone_from_hf``.  The official loader
    reads ``config.json`` from the HF repo to auto-detect architecture
    (vit_base / vit_large) and MoE settings.

    Returns a bare ``VisionTransformer`` — no ModalExtractor wrappers, no
    classifier.  Call ``backbone.eval()`` + ``backbone.requires_grad_(False)``
    if freezing.
    """
    from src.models.components.jepa import load_backbone

    return load_backbone(
        model_name_or_path=checkpoint_path,
        device=device,
        **kwargs,
    )


# Backward-compatible alias
build_jepa_brain_age_backbone = build_backbone

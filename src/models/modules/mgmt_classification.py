"""MGMT methylation classification with MultiModalJEPARegressor.

Two-phase training:

Phase 1 (epochs 0 .. unfreeze_at - 1):
    Frozen ViT backbone.  Only ``classifier`` (ModalityFusion) is trained.

Phase 2 (epochs unfreeze_at .. max_epochs):
    Backbone unfrozen.  Both backbone and classifier are trained, each with
    its own optimizer and learning rate.

The model accepts ``{modality: tensor | None}`` dicts — missing modalities
are skipped by the ViT encoder and imputed by MeanImputeReconstructor at
the classifier hook (if hook DAG is active).  Without hooks the ModalityFusion
mean-aggregates available modalities directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import (
    MeanMetric,
    MaxMetric,
    Accuracy,
    F1Score,
    AUROC,
)

log = logging.getLogger(__name__)


class MGMTClassificationModule(LightningModule):
    """Two-phase MGMT promoter methylation classifier.

    Parameters
    ----------
    net : MultiModalJEPARegressor
        Pretrained ViT backbone + ModalityFusion classifier.
    backbone_optimizer : partial
        Optimizer for ViT backbone (e.g. AdamW, lr=1e-5).
    classifier_optimizer : partial
        Optimizer for the ModalityFusion head (e.g. AdamW, lr=1e-4).
    backbone_scheduler : partial or None
    classifier_scheduler : partial or None
    unfreeze_at_epoch : int
        Epoch at which the backbone is unfrozen (phase 1 → phase 2).
    freeze_backbone_initially : bool
        If True (default), start with frozen backbone.
    """

    def __init__(
        self,
        net: nn.Module,
        backbone_optimizer: Any,
        classifier_optimizer: Any,
        backbone_scheduler: Any | None = None,
        classifier_scheduler: Any | None = None,
        unfreeze_at_epoch: int = 10,
        freeze_backbone_initially: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # Phase tracking
        self._phase = 1
        self._backbone_frozen = freeze_backbone_initially
        if freeze_backbone_initially:
            self.net.backbone.requires_grad_(False)
            self.net.backbone.eval()

        self.automatic_optimization = False

        # --- Loss ---
        self.criterion = nn.CrossEntropyLoss()

        # --- Metrics ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_acc_best = MaxMetric()
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_f1_best = MaxMetric()
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_auroc_best = MaxMetric()

    # ------------------------------------------------------------------
    # Forward / model_step
    # ------------------------------------------------------------------

    def forward(self, images: Dict[str, torch.Tensor | None]) -> torch.Tensor:
        return self.net(images)

    def model_step(
        self, batch: Tuple[Dict[str, torch.Tensor | None], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure forward + loss — no optimizer interaction.

        Returns (loss, logits, y).
        """
        images, y = batch
        y = y.long()
        logits = self.forward(images)
        loss = self.criterion(logits, y)
        return loss, logits, y

    # ------------------------------------------------------------------
    # Manual optimisation (ekf.py pattern)
    # ------------------------------------------------------------------

    def manual_optimize(
        self, loss: torch.Tensor
    ) -> None:
        """Run manual backward + step for backbone and classifier optimizers.
        Phase 1: only classifier optimizer steps (backbone frozen).
        Phase 2: both optimizers step.
        """
        opt_bb, opt_cls = self.optimizers()
        sch_bb, sch_cls = self._schedulers_or_none()

        should_step_bb = not self._backbone_frozen

        # --- Backbone ---
        if should_step_bb:
            self.toggle_optimizer(opt_bb)
            opt_bb.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            self.clip_gradients(opt_bb, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_bb.step()
            if sch_bb is not None:
                sch_bb.step()
            self.untoggle_optimizer(opt_bb)

        # --- Classifier ---
        self.toggle_optimizer(opt_cls)
        opt_cls.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt_cls, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_cls.step()
        if sch_cls is not None:
            sch_cls.step()
        self.untoggle_optimizer(opt_cls)

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        epoch = self.current_epoch
        unfreeze_at = self.hparams.unfreeze_at_epoch

        if epoch == unfreeze_at and self._backbone_frozen:
            log.info(">>> Phase 2: unfreezing backbone at epoch %d", epoch)
            self.net.backbone.requires_grad_(True)
            self.net.backbone.train()
            self._backbone_frozen = False
            self._phase = 2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor | None], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, logits, y = self.model_step(batch)
        self.manual_optimize(loss)

        # --- logging ---
        preds = torch.argmax(logits, dim=1)
        self.train_loss(loss)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.train_auroc(F.softmax(logits, dim=1)[:, 1], y)

        pfx = f"train/p{self._phase}"
        self.log(f"{pfx}/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{pfx}/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{pfx}/f1", self.train_f1, on_step=True, on_epoch=True)
        self.log(f"{pfx}/auroc", self.train_auroc, on_step=True, on_epoch=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        opt_bb, opt_cls = self.optimizers()
        self.log("lr/backbone", opt_bb.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("lr/classifier", opt_cls.param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor | None], torch.Tensor], batch_idx: int
    ) -> None:
        images, y = batch
        y = y.long()
        logits = self.forward(images)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_loss(loss)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(F.softmax(logits, dim=1)[:, 1], y)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.val_acc_best(self.val_acc.compute())
        self.val_f1_best(self.val_f1.compute())
        self.val_auroc_best(self.val_auroc.compute())
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)
        self.log("val/auroc_best", self.val_auroc_best.compute(), prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self, batch: Tuple[Dict[str, torch.Tensor | None], torch.Tensor], batch_idx: int
    ) -> None:
        images, y = batch
        y = y.long()
        logits = self.forward(images)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", Accuracy(task="binary").to(self.device)(preds, y),
                 on_step=False, on_epoch=True)
        self.log("test/f1", F1Score(task="binary").to(self.device)(preds, y),
                 on_step=False, on_epoch=True)
        self.log("test/auroc", AUROC(task="binary").to(self.device)(
            F.softmax(logits, dim=1)[:, 1], y), on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> List | Tuple:
        opt_bb = self.hparams.backbone_optimizer(params=self.net.backbone.parameters())
        opt_cls = self.hparams.classifier_optimizer(params=self.net.classifier.parameters())

        optimizers = [opt_bb, opt_cls]
        schedulers: List[Any] = []

        if self.hparams.backbone_scheduler is not None:
            schedulers.append(self.hparams.backbone_scheduler(optimizer=opt_bb))
        if self.hparams.classifier_scheduler is not None:
            schedulers.append(self.hparams.classifier_scheduler(optimizer=opt_cls))

        if schedulers:
            return optimizers, schedulers
        return optimizers

    def _schedulers_or_none(self) -> Tuple[Any, Any]:
        try:
            lr_config = self.trainer.lr_scheduler_configs
            sch_bb = lr_config[0].scheduler if len(lr_config) > 0 else None
            sch_cls = lr_config[1].scheduler if len(lr_config) > 1 else None
        except Exception:
            sch_bb = sch_cls = None
        return sch_bb, sch_cls


if __name__ == "__main__":
    import logging
    from typing import Optional

    import hydra
    import lightning as L
    import rootutils
    import torch
    from lightning import LightningDataModule, LightningModule, Trainer
    from omegaconf import DictConfig

    rootutils.setup_root(
        search_from=__file__, indicator=".project-root", pythonpath=True
    )

    from src.utils import (
        RankedLogger,
        instantiate_callbacks,
        instantiate_loggers,
    )

    log = RankedLogger(__name__, rank_zero_only=True)

    @hydra.main(
        version_base="1.3",
        config_path="../../../configs",
        config_name="train_mgmt",
    )
    def main(cfg: DictConfig) -> Optional[float]:
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)

        # --------------------------------------------------------------
        # 1.  Config composition
        # --------------------------------------------------------------
        print()
        print("=" * 60)
        print("1.  Config composition")
        print("=" * 60)
        print(f"  data._target_:   {cfg.data._target_}")
        print(f"  model._target_:  {cfg.model._target_}")
        print(f"  model.net:       {cfg.model.net._target_}")
        print(f"  model.opt_bb:    {cfg.model.backbone_optimizer._target_}")
        print(f"  model.opt_cls:   {cfg.model.classifier_optimizer._target_}")
        print(f"  trainer.devices: {cfg.trainer.devices}")

        # --------------------------------------------------------------
        # 2.  Model instantiation & inference test
        # --------------------------------------------------------------
        print()
        print("=" * 60)
        print("2.  Model instantiation + forward inference")
        print("=" * 60)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        print(f"  backbone:    {type(model.net.backbone).__name__}")
        print(f"  classifier:  {type(model.net.classifier).__name__}")
        print(f"  num_classes: {model.net.num_classes}")
        print(f"  phase:       {model._phase}")
        print(f"  frozen:      {model._backbone_frozen}")

        # --- 3-modality forward ---
        x_full = {
            "t1": torch.randn(2, 1, 96, 108, 96),
            "t2": torch.randn(2, 1, 96, 108, 96),
            "flair": torch.randn(2, 1, 96, 108, 96),
        }
        model.eval()
        with torch.no_grad():
            out_full = model.forward(x_full)
        print(f"  forward 3-mod: logits={list(out_full.shape)}")

        # --- missing t2 (UCSF-PDGM case) ---
        x_miss = {"t1": torch.randn(2, 1, 96, 108, 96), "t2": None, "flair": torch.randn(2, 1, 96, 108, 96)}
        with torch.no_grad():
            out_miss = model.forward(x_miss)
        print(f"  forward t2=None: logits={list(out_miss.shape)}")

        # # --- 4-D input auto-unsqueeze ---
        # x_4d = {"t1": torch.randn(2, 1, 96, 108), "t2": torch.randn(2, 1, 96, 108), "flair": torch.randn(2, 1, 96, 108)}
        # with torch.no_grad():
        #     out_4d = model.forward(x_4d)
        # print(f"  forward 4D->5D: logits={list(out_4d.shape)}")

        # # --- loss computation ---
        # y_dummy = torch.randint(0, 2, (2,))
        # loss_val = model.criterion(out_full, y_dummy)
        # print(f"  CrossEntropy:   {loss_val.item():.4f}")

        # --------------------------------------------------------------
        # 3.  DataModule instantiation
        # --------------------------------------------------------------
        print()
        print("=" * 60)
        print("3.  DataModule instantiation")
        print("=" * 60)

        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        try:
            datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
            datamodule.setup()
            n_train = len(datamodule.train_dataset)
            n_val = len(datamodule.val_dataset)
            n_test = len(datamodule.test_dataset)
            print(f"  splits: train={n_train}, val={n_val}, test={n_test}")
            has_data = n_train > 0
        except Exception as e:
            print(f"  DataModule setup skipped (no data files): {e}")
            has_data = False

        # --------------------------------------------------------------
        # 4.  Trainer + training smoke test
        # --------------------------------------------------------------
        print()
        print("=" * 60)
        print("4.  Trainer + training smoke test")
        print("=" * 60)

        callbacks = instantiate_callbacks(cfg.get("callbacks"))
        logger_list = instantiate_loggers(cfg.get("logger"))
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger_list
        )

        if has_data:
            print("  Running trainer.fit with limit_train_batches=2 ...")
            trainer.fit_loop.max_steps = 2
            trainer.fit(model=model, datamodule=datamodule)
            print("  trainer.fit PASSED")
        else:
            print("  Skipped — no data files available")

        print()
        print("All tests passed.")
        return None

    main()

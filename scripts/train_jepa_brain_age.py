"""Two-phase brain age training with Neuro-JEPA + AttentiveRegressor.

Phase 1: train regressor only (frozen ViT, lr=1e-4)
Phase 2: finetune backbone (lr=1e-5) + regressor (lr=5e-5)

Backbone is loaded via the official Neuro-JEPA API
(neurojepa.utils.init_utils.load_backbone_from_hf), which reads config.json
from the HF repo to auto-detect architecture (vit_base/vit_large) and MoE.

Input: (B, 1, D, H, W) 5D volumetric T1w in MNI152 space.

Usage:
    python scripts/train_jepa_brain_age.py
    python scripts/train_jepa_brain_age.py model.unfreeze_at_epoch=5 trainer.max_epochs=30
"""

import os
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.models.modules.jepa_brain_age import build_backbone, JEPABrainAgeModule


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_jepa_brain_age",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    print("=" * 60)
    print("JEPA Brain Age — Two-Phase Training")
    print("=" * 60)
    # Verify config matches the reference MNI152 template spec
    tpl = OmegaConf.load(str(_ROOT / "configs" / "data" / "template" / "mni152.yaml"))
    assert list(cfg.model.image_size) == list(tpl.t1.shape), \
        f"image_size {cfg.model.image_size} != MNI152 T1 shape {tpl.t1.shape}"
    print(f"  MNI152 T1:  {cfg.model.image_size} (verified)")
    print(f"  ViT input:  {cfg.model.roi_size}, patch {cfg.model.patch_size}")
    print(f"  Checkpoint: {cfg.model.checkpoint}")
    print(f"  Phase 2 at: epoch {cfg.model.unfreeze_at_epoch}")

    # 1. Backbone via official API
    print("\n[1/4] Loading pretrained JEPA backbone (official API)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(
        checkpoint_path=cfg.model.checkpoint,
        device=device,
    )
    print(f"  Backbone: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params")

    # 2. AttentiveRegressor
    print("\n[2/4] Building AttentiveRegressor...")
    regressor = instantiate(cfg.model.regressor)
    print(f"  Regressor: {sum(p.numel() for p in regressor.parameters())/1e6:.2f}M params")

    # 3. LightningModule
    print("\n[3/4] Building JEPABrainAgeModule (manual optim, 2 optimizers)...")
    model = JEPABrainAgeModule(
        backbone=backbone,
        regressor=regressor,
        backbone_optimizer=cfg.model.backbone_optimizer,
        regressor_optimizer=cfg.model.regressor_optimizer,
        backbone_scheduler=cfg.model.get("backbone_scheduler"),
        regressor_scheduler=cfg.model.get("regressor_scheduler"),
        unfreeze_at_epoch=cfg.model.unfreeze_at_epoch,
        freeze_backbone_initially=cfg.model.freeze_backbone_initially,
    )
    print(f"  Phase 1 lr: BB={cfg.model.backbone_optimizer.lr}, Reg={cfg.model.regressor_optimizer.lr}")

    # 4. DataModule
    print("\n[4/4] Building OpenBHB DataModule...")
    datamodule = instantiate(cfg.data)
    eff_bs = cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
    print(f"  batch={cfg.data.batch_size} x accumulate={cfg.trainer.accumulate_grad_batches} = effective {eff_bs}")

    # Verify first batch shape
    dm = datamodule
    dm.setup()
    x, y = next(iter(dm.val_dataloader()))
    print(f"  Sample shape: {list(x.shape)}  (expect [B, 1, 96, 108, 96])")
    print(f"  Age range: [{y.min().item():.1f}, {y.max().item():.1f}]")

    # Trainer
    cb = instantiate(cfg.callbacks) if cfg.get("callbacks") else {}
    trainer = Trainer(**cfg.trainer, callbacks=cb)

    if cfg.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if cfg.test:
        trainer.test(model, datamodule=datamodule)

    print("Done.")


if __name__ == "__main__":
    main()

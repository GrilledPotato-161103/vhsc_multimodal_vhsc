"""Train JEPA Multi-Modal Hook DAG Pipeline.

Loads a pretrained Neuro-JEPA backbone from HuggingFace Hub
(``NYUMedML/Neuro-JEPA``) or a local checkpoint, wraps it into a
:class:`~src.models.components.jepa.MultiModalJEPARegressor`, attaches
DAG breakpoints for reconstruction and uncertainty estimation, and
trains the hook modules using PyTorch Lightning + Hydra.

Usage
-----
.. code-block:: bash

    python src/train_hook_jepa.py  # uses configs/train_hook_jepa.yaml
    python src/train_hook_jepa.py model.epoch_phase=5
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

torch.serialization.add_safe_globals([functools.partial])

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.plugins.hook_dag import BreakpointController, Breakpoint
from src.models.components.jepa import MultiModalJEPARegressor, build_jepa_regressor

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
os.environ["WANDB_CONSOLE"] = "off"


def _load_jepa_model(cfg: DictConfig, device: torch.device) -> MultiModalJEPARegressor:
    """Load MultiModalJEPARegressor from HF Hub or local checkpoint.

    Configuration keys read from ``cfg.plugins``:

    - ``model_checkpoint``: HF repo ID (e.g. ``NYUMedML/Neuro-JEPA``) or
      local ``.pt`` / ``.pth`` / ``.safetensors`` path.
    - ``modality_keys``: list of modality names (default ``["t1w", "t2w"]``).
    - ``image_size``: ``[D, H, W]`` (default ``[96, 108, 96]``).
    - ``num_classes``: for the classifier head (default 2).
    - ``hf_token``: ``true`` to use cached HF login, string for explicit token.
    - ``hf_revision`` / ``hf_cache_dir``: optional HF controls.
    """
    model_checkpoint = cfg.plugins.get("model_checkpoint", "NYUMedML/Neuro-JEPA")
    modality_keys = cfg.plugins.get("modality_keys", ["t1w", "t2w"])
    image_size = cfg.plugins.get("image_size", [96, 108, 96])
    if isinstance(image_size, (list, tuple)):
        image_size = tuple(image_size)
    num_classes = cfg.plugins.get("num_classes", 2)
    hf_token = cfg.plugins.get("hf_token", True)
    hf_revision = cfg.plugins.get("hf_revision", None)
    hf_cache_dir = cfg.plugins.get("hf_cache_dir", None)

    log.info("Loading JEPA model from: %s (HF token: %s)", model_checkpoint, bool(hf_token))

    wrapper = build_jepa_regressor(
        model_name_or_path=model_checkpoint,
        device=device,
        modality_keys=modality_keys,
        image_size=image_size,
        num_classes=num_classes,
        freeze_backbone=True,
        hf_token=hf_token,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
    )

    log.info(
        "JEPA model loaded. Backbone: %s  embed_dim=%d  num_classes=%d",
        type(wrapper.backbone).__name__,
        getattr(wrapper.backbone, "embed_dim", "?"),
        wrapper.num_classes,
    )
    log.info(
        "Named modules (first 30): %s",
        [name for name, _ in wrapper.named_modules()][:30],
    )
    return wrapper


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Datamodule ----------------------------------------------------------
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    # --- Load JEPA model -----------------------------------------------------
    net = _load_jepa_model(cfg, device)
    net.eval()
    net.requires_grad_(False)

    # --- Breakpoint controller -----------------------------------------------
    controller = BreakpointController.__init_dict__(net, cfg.plugins)
    controller.to(device)
    log.info("Breakpoints registered: %s", list(Breakpoint.list_of_breakpoints.keys()))
    for item in controller.breakpoints:
        log.info(
            "  %s @ %s [%s]  mutate=%s",
            item["breakpoint"].name,
            item["layer_name"],
            item["position"],
            item["breakpoint"].mutate,
        )

    # --- LightningModule -----------------------------------------------------
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model(net=net, controller=controller)

    # --- Callbacks & loggers -------------------------------------------------
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # --- Trainer -------------------------------------------------------------
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_hook_jepa.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )
    return metric_value


if __name__ == "__main__":
    main()

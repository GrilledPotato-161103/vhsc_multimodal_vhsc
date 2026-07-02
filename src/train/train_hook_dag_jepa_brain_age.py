"""Train hook DAG on Neuro-JEPA for brain age estimation.

Loads a pretrained Neuro-JEPA backbone, attaches BreakpointController with
reconstructor + uncertainty hooks on all encoder layers and final output layer,
then trains via Lightning.

Usage:
  python src/train_hook_dag_jepa_brain_age.py
"""

from __future__ import annotations

import functools
import os
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

from src.models.components.jepa import build_jepa_regressor
from src.plugins.hook_dag import Breakpoint, BreakpointController
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


def _ensure_pretrained_checkpoint(cfg: DictConfig) -> str:
    """Download pretrained JEPA backbone if not already cached."""
    ckpt_path = cfg.plugins.model_checkpoint
    if os.path.exists(ckpt_path):
        log.info("Pretrained JEPA checkpoint found at %s", ckpt_path)
        return ckpt_path

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    log.info("Downloading pretrained Neuro-JEPA from HuggingFace Hub ...")
    model_name = cfg.get("jepa_hf_repo", "NYUMedML/Neuro-JEPA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    regressor = build_jepa_regressor(
        model_name_or_path=model_name,
        device=device,
        modality_keys=cfg.get("modality_keys", ["t1w", "t2w"]),
        image_size=tuple(cfg.get("image_size", [96, 108, 96])),
        num_classes=1,  # brain age regression
        freeze_backbone=True,
    )

    torch.save(regressor, ckpt_path)
    log.info("Saved pretrained model to %s", ckpt_path)
    return ckpt_path


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # --- Ensure pretrained checkpoint exists ---
    _ensure_pretrained_checkpoint(cfg)

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading JEPA regressor from %s", cfg.plugins.model_checkpoint)
    net = torch.load(cfg.plugins.model_checkpoint, map_location=device, weights_only=False)
    net.to(device)
    net.eval()
    net.requires_grad_(False)

    # --- Datamodule ---
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    # --- Controller (hook DAG) ---
    controller = BreakpointController.__init_dict__(net, cfg.plugins)
    controller.to(device)
    log.info("Breakpoints registered: %s", list(Breakpoint.list_of_breakpoints.keys()))
    for item in controller.breakpoints:
        log.info(
            "  %s @ %s [%s] -> sinks: %s",
            item["breakpoint"].name,
            item["layer_name"],
            item["position"],
            [s.name for s in item["breakpoint"].data_sinks],
        )

    # --- LightningModule ---
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model(net=net, controller=controller)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

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

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_hook_dag_jepa_brain_age.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()

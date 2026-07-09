"""Stage 2 training entry point: regression on frozen shared embeddings.

Loads a Stage 1 checkpoint, freezes the encoders, and trains a regression
head on the physics-based target y.

Usage:
    python src/train/train_stage2.py \\
        stage1_ckpt_path=logs/train_stage1/runs/.../checkpoints/stage1_shared_embedding.pth
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import functools
import os

import hydra
import lightning as L
import rootutils
import torch

torch.serialization.add_safe_globals([functools.partial])

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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


# ---------------------------------------------------------------------------
# Checkpoint loading helper
# ---------------------------------------------------------------------------

def _load_stage1_net(stage1_ckpt_path: str) -> torch.nn.Module:
    """Load a ``SharedEmbeddingNet`` saved by Stage 1's ``on_save_checkpoint``.

    Stage 1 saves the net directly via ``torch.save(self.net, path)``, so
    loading requires ``weights_only=False``.
    """
    if not stage1_ckpt_path:
        raise ValueError(
            "stage1_ckpt_path must be set (e.g. via CLI: "
            "stage1_ckpt_path=logs/.../stage1_shared_embedding.pth)"
        )
    if not os.path.exists(stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt_path}")

    log.info(f"Loading Stage 1 net from: {stage1_ckpt_path}")
    net = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train Stage 2: regression head on frozen shared embeddings."""
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # -- data ----------------------------------------------------------------
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # -- load Stage 1 net ----------------------------------------------------
    stage1_net = _load_stage1_net(cfg.stage1_ckpt_path)

    # -- instantiate regressor head ------------------------------------------
    log.info("Instantiating regressor head...")
    regressor_head = hydra.utils.instantiate(cfg.model.regressor_head)

    # -- build Stage 2 module ------------------------------------------------
    log.info("Building ManifoldRegressorModule...")
    model_kwargs: Dict[str, Any] = {
        "stage1_net": stage1_net,
        "regressor_head": regressor_head,
        "optimizer": hydra.utils.instantiate(cfg.model.optimizer),
        "freeze_encoders": cfg.model.freeze_encoders,
        "loss_name": cfg.model.loss_name,
    }
    if cfg.model.get("scheduler") is not None:
        model_kwargs["scheduler"] = hydra.utils.instantiate(cfg.model.scheduler)
    if cfg.model.get("huber_delta") is not None:
        model_kwargs["huber_delta"] = cfg.model.huber_delta
    if cfg.model.get("compile_model") is not None:
        model_kwargs["compile_model"] = cfg.model.compile_model

    model: LightningModule = hydra.utils.instantiate(cfg.model, **model_kwargs)

    # -- callbacks & loggers & trainer ---------------------------------------
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
        log.info("Starting Stage 2 training!")
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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_stage2")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()

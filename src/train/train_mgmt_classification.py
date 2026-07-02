"""Training script for MGMT methylation classification.

Two-phase training with MultiModalJEPARegressor:
  Phase 1 (epochs 0 .. unfreeze_at-1): frozen ViT backbone, train classifier only.
  Phase 2 (epochs unfreeze_at .. max_epochs): unfrozen backbone + classifier.

Usage::

    python src/train/train_mgmt_classification.py
    python src/train/train_mgmt_classification.py trainer.max_epochs=100
    python src/train/train_mgmt_classification.py data=ucsf_pdgm       # OOD eval
"""

from typing import Any, Dict, List, Optional, Tuple

import functools

import hydra
import lightning as L
import rootutils
import torch

torch.serialization.add_safe_globals([functools.partial])

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

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


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train MGMT classifier.  Can evaluate on a held-out test set using
    the best checkpoint from training.

    :param cfg: Hydra-composed DictConfig.
    :return: (metric_dict, object_dict).
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger_list: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger_list
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger_list,
        "trainer": trainer,
    }

    if logger_list:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="train_mgmt",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for MGMT classification training.

    :param cfg: DictConfig composed by Hydra.
    :return: Optional[float] with the optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = train(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric", "val/auroc"),
    )
    return metric_value


if __name__ == "__main__":
    main()

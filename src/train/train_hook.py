from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import argparse
import functools
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

torch.serialization.add_safe_globals([functools.partial])

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.plugins.hook import BreakpointController, Breakpoint
from src.plugins.aggregation import (
    AggregationController,
    AggregationSpec,
    SourceSpec,
    TargetSpec,
)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

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


def _build_hook_dag_specs(
    dag_cfg: DictConfig,
    model_type: str,
) -> List[AggregationSpec]:
    """Build :class:`AggregationSpec` list from a ``hook_dag`` config block.

    The config specifies topology (sources, target, mode); this function
    supplies the model-appropriate ``aggregate_fn``.

    Parameters
    ----------
    dag_cfg:
        The ``cfg.plugins.hook_dag`` sub-config.
    model_type:
        The model class name (e.g. ``"MultiModalRegressor"``,
        ``"BiModalRegressor"``).

    Returns
    -------
    List[AggregationSpec]
    """
    specs: List[AggregationSpec] = []

    for spec_cfg in dag_cfg.specs:
        sources = [
            SourceSpec(
                layer=s.layer,
                position=s.get("position", "after"),
                key=s.key,
            )
            for s in spec_cfg.sources
        ]
        target = TargetSpec(
            layer=spec_cfg.target.layer,
            position=spec_cfg.target.get("position", "before"),
            input_key=spec_cfg.target.get("input_key", None),
        )

        # Build aggregate_fn based on model architecture.
        # MultiModalRegressor sums latents → HookDAG should also sum.
        # BiModalRegressor concatenates → HookDAG should also concat.
        # Fallback: default_aggregate_fn (concatenate tensors along dim=-1).
        if model_type == "MultiModalRegressor":
            def _make_sum() -> Callable[[Dict[str, Any]], Any]:
                def _sum_fn(collected: Dict[str, Any]) -> Any:
                    tensors = [v for v in collected.values()
                               if isinstance(v, torch.Tensor)]
                    if not tensors:
                        return collected
                    return torch.stack(tensors).sum(dim=0)
                return _sum_fn
            agg_fn = _make_sum()
            log.info("HookDAG aggregate_fn: sum (MultiModalRegressor)")
        else:
            agg_fn = None  # use AggregationSpec.default_aggregate_fn (concat)
            log.info("HookDAG aggregate_fn: default concat")

        specs.append(AggregationSpec(
            name=spec_cfg.name,
            sources=sources,
            target=target,
            aggregate_fn=agg_fn,
            mode=spec_cfg.get("mode", "all"),
            min_sources=spec_cfg.get("min_sources", None),
        ))

    return specs


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    log.info(f"Instantiating model <{cfg.model._target_}>")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load(cfg.plugins.model_checkpoint, weights_only=False).to(device)
    net.eval()
    net.requires_grad_(True)

    # --- controller + optional HookDAG ---------------------------------------
    controller = BreakpointController()
    agg_ctrl: Optional[AggregationController] = None

    # If a HookDAG checkpoint is provided, load everything from it.
    # Otherwise build HookDAG from config (if present), then add breakpoints.
    if (ckpt := cfg.plugins.get("plugins_checkpoint")) and os.path.isfile(ckpt):
        log.info(f"Loading HookDAG + breakpoints from checkpoint: {ckpt}")
        from src.plugins.aggregation import AggregationController as AC
        controller, agg_ctrl = AC.load_checkpoint(
            ckpt, net,
            _build_hook_dag_specs(cfg.plugins.hook_dag, type(net).__name__),
        )
    else:
        # 1. Register HookDAG first (so its endpoints fire before breakpoints)
        if (dag_cfg := cfg.plugins.get("hook_dag")) and dag_cfg.get("specs"):
            log.info("Initializing HookDAG...")
            specs = _build_hook_dag_specs(dag_cfg, type(net).__name__)
            agg_ctrl = AggregationController(specs)
            agg_ctrl.register(controller, net)
            log.info("HookDAG registered:\n%s", agg_ctrl.summary())

        # 2. Add standard breakpoints (reconstructor, uncertainty, etc.)
        assert type(net).__name__ == cfg.plugins.target, (
            f"Plugin target mismatch: cfg says '{cfg.plugins.target}', "
            f"model is '{type(net).__name__}'"
        )
        for item in cfg.plugins.breakpoints:
            bp = hydra.utils.instantiate(item.bp)
            controller.add_breakpoint_by_name(net, item.layer_name, bp, item.pos)

    controller.to(device)
    log.info("Breakpoints registered: %s", list(Breakpoint.list_of_breakpoints.keys()))

    # Attach HookDAG controller to BreakpointController so downstream
    # code (LightningModule.on_save_checkpoint, etc.) can access it.
    if agg_ctrl is not None:
        controller._agg_ctrl = agg_ctrl  # type: ignore[attr-defined]

    # --- build LightningModule ------------------------------------------------
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model(
        net=net,
        controller=controller,
        src_dataset=datamodule.src_dataset,
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

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
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_ekf_hook.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

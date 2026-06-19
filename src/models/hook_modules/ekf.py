from typing import Any, Dict, Tuple, Callable
from collections import defaultdict
import math
import numpy as np
from omegaconf import DictConfig
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.plugins.hook import BreakpointController, Breakpoint
from src.plugins.head.bayescap import BayesCap1DLoss, bayescap_variance_1d
from src.plugins.sigma_z import SDSigmaZ
from src.plugins.head.ekf import EKFBiModalInferer
from src.plugins.head.hessian import *
from src.plugins.ekf_propagation import full_ekf_propagation_full, make_reconstructor_fn, make_predictor_fn
from src.models.hook_modules.common import HuberLoss, check_gradient

import functools
torch.serialization.add_safe_globals([functools.partial])

class ModelEKFInjectModule(LightningModule):
    def __init__(self,
                 net: nn.Module,
                 ekf_net: EKFBiModalInferer,
                 recon_bp: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 expression: str | None = None,
                 controller: BreakpointController | None | DictConfig | Dict = None,
                 controller_cache_path: str = "", 
                 compile: bool = False,
                 recon_criterion: nn.Module | Callable | None = nn.MSELoss(),
                 unc_criterion: nn.Module | Callable | None = nn.MSELoss(),
                 epoch_phase: int = 20,
                 mask_rate: float = 0.3,
                 source_x_range: tuple = (-1.0, 1.0),
                 n_source_samples: int = 5000,
                 src_dataset: Dataset | None = None,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["retcon_criterion", "unc_criterion", "net", "controller", "ekf_net"])
        self.net = net
        self.controller = controller
        self.ekf_net = ekf_net
        self.recon_bp = Breakpoint.get_by_name(recon_bp)
        self.train_loss = nn.ModuleList([MeanMetric() for i in range(4)])
        self.val_loss = nn.ModuleList([MeanMetric() for i in range(4)])
        self.test_loss = nn.ModuleList([MeanMetric() for i in range(4)])

        self.train_recon_loss = nn.ModuleList([MeanMetric() for i in range(4)])
        self.val_recon_loss = nn.ModuleList([MeanMetric() for i in range(4)])
        self.test_recon_loss = nn.ModuleList([MeanMetric() for i in range(4)])

        self.train_unc_loss = nn.ModuleList([MeanMetric() for i in range(4)])
        self.val_unc_loss = nn.ModuleList([MeanMetric() for i in range(4)])
        self.test_unc_loss = nn.ModuleList([MeanMetric() for i in range(4)])

        self.train_nll = nn.ModuleList([MeanMetric() for i in range(4)])
        self.val_nll = nn.ModuleList([MeanMetric() for i in range(4)])
        self.test_nll = nn.ModuleList([MeanMetric() for i in range(4)])

        self.val_nll_best = MinMetric() 

        self.criterion = torch.nn.MSELoss(reduction="none")
        self.recon_criterion = recon_criterion
        self.unc_criterion = unc_criterion
        self.unc_criterion.requires_grad_(True)
        
        enc1 = self.net.x1_encoder
        enc2 = self.net.x2_encoder

        self.ekf_net = self.ekf_net(self.recon_bp.callback, self.net.head)
        self.ekf_net.requires_grad_(True)
        # Per-sample SD-setting input-shift covariance provider.
        # Fits N(mu_A, Sigma_A) on source latents once at init; computes
        # Sigma_z(z) = (d_M^2(z) / d_z) * Sigma_A per batch at training time.
        self.sigma_z_provider = SDSigmaZ(
            encoder1=enc1,
            encoder2=enc2,
            dataset=src_dataset,
            x_range=tuple(source_x_range),
            n_source_samples=n_source_samples,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # Set this to False to control training flow
        self.automatic_optimization = False
    
    def configure_model(self):
        if self.controller is None:
            self.controller = BreakpointController.load_from_checkpoint(self.net, self.hparams.controller_cache_path)
        return super().configure_model()
    
    def on_fit_start(self):
        return super().on_fit_start()
        
    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
            Perform forward on hooked model
        """
        return self.net(*xs)

    def on_train_start(self):
        # Prevent training on training phase
        self.controller.train()
        self.ekf_net.train()
        return super().on_train_start()
    
    def on_train_epoch_start(self):
        # Turn EKFNet gradient on at phase 2.
        if self.current_epoch == self.hparams.epoch_phase:
            print("Switching on EKFNet Gradient Propagation")
            self.ekf_net.train()
        elif self.current_epoch > self.hparams.epoch_phase:
            print("EKFNet is training")
        return super().on_train_epoch_start()
    
    def model_step(
        self, batch, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # Include bp_kwargs in Dataset for breakpoint manipulation
        self.net.eval()
        self.net.requires_grad_(False)

        xs, y, xs_orig, zs = batch
        
        xs = list(torch.split(xs, 1, dim=1))
        xs_orig = list(torch.split(xs_orig, 1, dim=1))
        # y = y.unsqueeze(-1)
        
        # x1 = x1.cuda().unsqueeze(1)
        # x2 = x2.cuda().unsqueeze(1)
        # y = y.cuda().unsqueeze(1)
        
        # Set kwargs for breakpoints, use cache if available
        
        if "bp_signal" in kwargs.keys():
            bp_signal = kwargs["bp_signal"]
        else:
            bp_signal = [1, 1]
            mask_index = np.random.choice(3, 1, p= (1 - self.hparams.mask_rate, 
                                                0,
                                                self.hparams.mask_rate))[0]
            if mask_index > 0: 
                bp_signal[mask_index - 1] = 0
                xs[mask_index - 1] *= 0
        
        self.recon_bp.kwargs = tuple(bp_signal)
        # print(self.recon_bp.kwargs)
        logits = self.forward(xs).unsqueeze(1)
        loss = self.criterion(logits, y)
        recon_trace = self.recon_bp.trace
        sigs = recon_trace.trace["signal"]
        recs = recon_trace.trace["reconstructed"]
        srcs = recon_trace.trace["input"]
        devs = recon_trace.trace["dev"]
        dists = recon_trace.trace["distance"]
        recon_loss = 0
        recon_unc_loss = 0
        for sig, rec, src, dev, dist in zip(sigs, recs, srcs, devs, dists): 
            if sig == 0: 
                continue
            recon_loss += self.recon_criterion(rec, src)
            recon_unc_loss += self.criterion(dev, dist)

        # EKF Propagation. sigma_z is per-sample full covariance from the
        # SD-setting provider (Mahalanobis-scaled source covariance).
        z = torch.cat(srcs, dim=-1).detach()
        # print(z.shape)
        sigma_z = self.sigma_z_provider(z)  # (B, d_z, d_z)
        mu, inv_alpha, beta, sigma_pred_sq = self.ekf_net(z, sigma_z, logits, signal=sigs)
        ekf_nll = self.unc_criterion(y_true=y, y_hat=logits, mu=mu, inv_alpha=inv_alpha, beta=beta)
    
        return loss, logits, y, \
                {"srcs": srcs, "recon_loss": recon_loss, "unc_loss": recon_unc_loss, "trace": recon_trace, "signal": bp_signal}, \
                {"var": bayescap_variance_1d(inv_alpha, beta), "loss": ekf_nll["loss"], "sigma_pred_sq": sigma_pred_sq.detach()}
    
    # Changing to manual optimize for freedom in model freeze 
    def manual_optimize(self, recon, unc, loss, batch_idx): 
        signal = recon["trace"].trace["signal"]
        [recon_opt, ekf_opt] = self.optimizers()
        [recon_scheduler, ekf_scheduler] = self.lr_schedulers()
        # ==== Uncertainty Optimization ==== Phase 2
        if self.current_epoch >= self.hparams.epoch_phase:
            self.toggle_optimizer(ekf_opt)
            ekf_opt.zero_grad()
            unc_loss = unc["loss"].mean()
            with torch.autograd.set_detect_anomaly(True, check_nan=False):
                self.manual_backward(unc_loss, retain_graph=True)
            if batch_idx == 0:
                print("Checking Uncertainty Head Gradient")
                print(f"Checking gradient for Loss's Calibrator  {self.unc_criterion.__class__.__qualname__}")
                check_gradient(self.unc_criterion)
                print(f"Checking gradient for Loss's Calibrator  {self.ekf_net.__class__.__qualname__}")
                check_gradient(self.ekf_net)
            ekf_opt.step()
            ekf_scheduler.step(unc_loss)
            self.untoggle_optimizer(ekf_opt)
        else:
            unc_loss = unc["loss"].mean()

        # ==== Reconstruction optimize ==== Phase 1
        
        self.toggle_optimizer(recon_opt)
        recon_opt.zero_grad()
        # Add Loss mean to guide reconstructor to data manifold
        recon_loss = recon["recon_loss"].mean() + recon["unc_loss"].mean()
        self.manual_backward(recon_loss)
        if batch_idx == 0:
            print("Checking reconstructor gradient")
            for item in self.controller.breakpoints:
                    pos, bp = item['position'], item["breakpoint"]
                    print(f"Checking {bp.name} module on {pos}: {bp.callback.__class__.__qualname__}")
                    check_gradient(bp.callback)
        recon_opt.step()
        if signal[0] + signal[1] < 2:
            recon_scheduler.step(recon_loss)
        self.untoggle_optimizer(recon_opt)
        # Only step the LR Scheduler when its missing a modality 
        
        return recon_loss, unc_loss
        
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Tạm thời tắt reconstruction để đánh giá uncertainty
        loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["trace"].trace["signal"]
        # Classify metrics to signal 
        signal_str = f"{signal[0]}{signal[1]}"
        metric_idx = int(signal_str, 2)
        # update and log metrics
        recon_loss, unc_loss = self.manual_optimize(recon, unc, loss, batch_idx=batch_idx)
        self.train_loss[metric_idx](loss.mean())
        self.log(f"train/loss_{signal_str}" , 
                 self.train_loss[metric_idx], 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True)
        
        self.train_recon_loss[metric_idx](recon["recon_loss"].mean())
        self.log(f"train/loss_recon_{signal_str}", 
                    self.train_recon_loss[metric_idx], 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)
        
        self.train_unc_loss[metric_idx](recon["unc_loss"].mean())
        self.log(f"train/loss_recon_unc_{signal_str}", 
                    self.train_unc_loss[metric_idx], 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)

        # Phase 1: Not propagating uncertainty of deficit inputs
        if self.current_epoch >= self.hparams.epoch_phase:
            self.train_nll[metric_idx](unc["loss"].mean())
            self.log(f"train/loss_unc_{signal_str}",
                    self.train_nll[metric_idx],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True)
            # EKF diagnostics: mean = magnitude, std = per-sample spread.
            # std ~ 0 means the chain collapses sigma_pred to a constant.
            sps = unc["sigma_pred_sq"].detach().flatten()
            self.log(f"train/sigma_pred_sq_mean_{signal_str}", sps.mean(),
                    on_step=True, on_epoch=True)
            self.log(f"train/sigma_pred_sq_std_{signal_str}", sps.std(),
                    on_step=True, on_epoch=True)
            # self.log(f"train/sigma_pred_sq_min_{signal_str}", sps.min(),
            #         on_step=True, on_epoch=True)
            # self.log(f"train/sigma_pred_sq_max_{signal_str}", sps.max(),
            #         on_step=True, on_epoch=True)
            # return loss or backpropagation will fail, focus on uncertainty loss only
            return recon["unc_loss"].mean()+ recon["recon_loss"].mean() + unc['loss'].mean()
        else:
            return recon["unc_loss"].mean() + recon["recon_loss"].mean()

    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ) -> None:
        return super().optimizer_step(
                                        epoch,
                                        batch_idx,
                                        optimizer,
                                        optimizer_closure,
                                    )
        
    def on_validation_start(self) -> None:
        self.controller.eval()
        self.ekf_net.eval()
        # print("Breakpoints are set to evaluation !!!")
        super().on_validation_start()
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        # Cached files
        with torch.enable_grad():
            loss, logits, _, recon, unc = self.model_step(batch, bp_signal=(1, 1))
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"
        metric_idx = int(signal_str, 2)
        self.val_loss[metric_idx](loss)
        self.log(f"val/loss", 
                 self.val_loss[metric_idx], 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True)

        self.val_recon_loss[metric_idx](recon["recon_loss"].mean())
        self.log(f"val/loss_recon_{signal_str}", 
                    self.val_recon_loss[metric_idx], 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)
        
        self.val_unc_loss[metric_idx](recon["unc_loss"].mean())
        self.log(f"val/loss_recon_unc_{signal_str}", 
                    self.val_unc_loss[metric_idx], 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)
        
        if self.current_epoch >= self.hparams.epoch_phase:
            self.val_nll[metric_idx](unc["loss"].mean())
            self.log(f"val/loss_unc_{signal_str}",
                    self.val_nll[metric_idx],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True)

            sps = unc["sigma_pred_sq"].detach().flatten()
            self.log(f"val/sigma_pred_sq_mean_{signal_str}", sps.mean(),
                    on_step=True, on_epoch=True)
            self.log(f"val/sigma_pred_sq_std_{signal_str}", sps.std(),
                    on_step=True, on_epoch=True)
            self.log(f"val/sigma_pred_sq_min_{signal_str}", sps.min(),
                    on_step=True, on_epoch=True)
            self.log(f"val/sigma_pred_sq_max_{signal_str}", sps.max(),
                    on_step=True, on_epoch=True)
        return (loss, logits, recon, unc)
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        score = min([nll.compute() for nll in self.val_nll])  # get current val acc
        self.val_nll_best(score)  # update best so far val acc
        # log `val_id_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_nll_best", self.val_nll_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_start(self):
        print("Testing and Ablation study on epoch", self.current_epoch)
        return super().on_test_epoch_start()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.enable_grad():
            loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"
        metric_idx = int(signal_str, 2)
        # update and log metrics
        self.test_loss[metric_idx](loss)
        self.log(f"test/loss", 
                 self.test_loss[metric_idx], 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True)

        self.test_recon_loss[metric_idx](recon["recon_loss"].mean())
        self.log(f"test/loss_recon_{signal_str}",
                    self.test_recon_loss[metric_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True)

        self.test_unc_loss[metric_idx](recon["unc_loss"].mean())
        self.log(f"test/loss_recon_unc_{signal_str}",
                    self.test_unc_loss[metric_idx],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True)

        self.test_nll[metric_idx](unc["loss"].mean())
        self.log(f"test/loss_unc_{signal_str}",
                self.test_nll[metric_idx],
                on_step=True,
                on_epoch=True,
                prog_bar=True)

        sps = unc["sigma_pred_sq"].detach().flatten()
        self.log(f"test/sigma_pred_sq_mean_{signal_str}", sps.mean(),
                 on_step=True, on_epoch=True)
        self.log(f"test/sigma_pred_sq_std_{signal_str}", sps.std(),
                 on_step=True, on_epoch=True)
        # self.log(f"test/sigma_pred_sq_min_{signal_str}", sps.min(),
        #          on_step=True, on_epoch=True)
        # self.log(f"test/sigma_pred_sq_max_{signal_str}", sps.max(),
        #          on_step=True, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        print("Parsing results !!!!!")
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        bp_parameters = []
        for item in self.controller.breakpoints:
            bp = item["breakpoint"]
            print(f"Assigning {bp.name} breakpoints to Optimizer for update")
            bp_parameters = bp_parameters + list(bp.callback.parameters())
        bp_optimizer = self.hparams.optimizer(params=bp_parameters)
        # Loss also has learnable calibration params
        ekf_parameters = self.ekf_net.get_parameters()
        ekf_optimizer = self.hparams.optimizer(params=ekf_parameters)
        if self.hparams.scheduler is not None:
            bp_scheduler = self.hparams.scheduler(optimizer=bp_optimizer)
            ekf_scheduler = self.hparams.scheduler(optimizer=ekf_optimizer)
            # scheduler_template = {
            #         "scheduler": None,
            #         "monitor": "val/loss_unc_11",
            #         "interval": "epoch",
            #         "frequency": 1,
            #     }
            # bp_scheduler = {"scheduler": bp_scheduler,
            #         "monitor": "val/loss_recon_10",
            #         "interval": "epoch",
            #         "frequency": 1,}
            # ekf_scheduler = {"scheduler": ekf_scheduler,
            #         "monitor": "val/loss_unc_11",
            #         "interval": "epoch",
            #         "frequency": 1,}
            return [bp_optimizer, ekf_optimizer], [bp_scheduler, ekf_scheduler]
        return {"optimizer": bp_optimizer}   

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        lit_state_dict: dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        filtered_dict = {k: v for k, v in lit_state_dict.items() if not k.startswith("recon_bp")}
        # print(list(filtered_dict.keys()))
        return filtered_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Counterpart to ``state_dict``: tolerate the deliberately-missing
        ``recon_bp.*`` keys (the breakpoint reconstructor is persisted to
        ``controller_cache.pth`` separately and restored in ``configure_model``)
        while keeping strict checks for every other key.
        """
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            missing = [k for k in result.missing_keys if not k.startswith("recon_bp")]
            unexpected = [k for k in result.unexpected_keys if not k.startswith("recon_bp")]
            if missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {type(self).__name__}:\n"
                    f"\tMissing key(s): {missing}\n"
                    f"\tUnexpected key(s): {unexpected}"
                )
        return result

    def on_save_checkpoint(self, checkpoint):
        self.controller.save(self.hparams.controller_cache_path, use_torch=True)
        pass

    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)
        
if __name__ == "__main__":
    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, DictConfig
    from functools import partial
 
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train_ekf_hook.yaml")
    def main(cfg: DictConfig) -> None: 
        plugin_cfg = cfg.plugins
        print("Initializing model")
        net = torch.load(cfg.plugins.model_checkpoint, weights_only=False).cuda()
        net.eval()
        net.requires_grad_(True)
        controller = BreakpointController.__init_dict__(net, cfg.plugins)
        controller.cuda()
        print(list(Breakpoint.list_of_breakpoints.keys()))
        # model: LightningModule = hydra.utils.instantiate(cfg.model)
        # model = model(net = net, controller = controller)
        # checkpoint = dict()
        model = LightningModule.load_from_checkpoint(r"logs/train/runs/2026-05-12_18-58-17/checkpoints/epoch_000.ckpt", weights_only=False)

    main()

    
from typing import Any, Dict, Tuple, Callable, Sequence
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
from src.plugins.ekf_propagation import full_ekf_propagation, make_reconstructor_fn, make_predictor_fn

import functools
torch.serialization.add_safe_globals([functools.partial])

def check_gradient(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # Get a summary metric to avoid console flooding
                grad_norm = param.grad.norm().item()
                print(f"Layer: {name: <30} | Gradient Norm: {grad_norm:.6f}")
            else:
                print(f"Layer: {name: <30} | Gradient: NONE")
        else:
            print(f"Layer: {name: <30} | Gradient: NOT SET")

class HuberLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred, target):
        l1_norm = torch.abs(target - pred)
        if l1_norm < self.threshold:
            return 0.5 * (l1_norm ** 2).mean()
        else:
            return (self.threshold * (l1_norm - self.threshold)).mean()

class ModelEKFManifoldModule(LightningModule):
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
                 sigma_z_mode: str = "mc",
                 src_dataset: Dataset | None = None
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["retcon_criterion", "unc_criterion", "net", "controller", "ekf_net"])
        self.net = net
        self.controller = controller
        self.ekf_net = ekf_net
        self.recon_bp = Breakpoint.get_by_name(recon_bp)
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.test_recon_loss = MeanMetric()

        self.train_unc_loss = MeanMetric()
        self.val_unc_loss = MeanMetric()
        self.test_unc_loss = MeanMetric()

        self.train_nll = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()

        self.val_nll_best = MinMetric()

        self.criterion = torch.nn.MSELoss(reduction="none")
        self.recon_criterion = recon_criterion
        self.unc_criterion = unc_criterion

        
        enc1 = self.net.x1_encoder
        enc2 = self.net.x2_encoder

        self.ekf_net = self.ekf_net(self.recon_bp.callback, self.net.head)
        # Per-sample SD-setting input-shift covariance provider.
        # Fits N(mu_A, Sigma_A) on source latents once at init; computes
        # Sigma_z(z) = (d_M^2(z) / d_z) * Sigma_A per batch at training time.
        self.sigma_z_provider = SDSigmaZ(
            encoder1=enc1,
            encoder2=enc2,
            dataset=src_dataset,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def configure_model(self):
        if self.controller is None:
            self.controller = BreakpointController.load_from_checkpoint(self.net, self.hparams.controller_cache_path)
        return super().configure_model()
    
    def on_fit_start(self):
        return super().on_fit_start()
        
    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
            Perform forward on hooked model
        """
        return self.net(*xs)

    def on_train_start(self):
        # Prevent training on training phase
        self.controller.train()
        return super().on_train_start()
    
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

        xs, y, zs = batch
        # xs = [x.unsqueeze(1) for x in xs]
        # y = y.cuda().unsqueeze(1)

        # Set kwargs for breakpoints, use cache if available
        if "bp_signal" in kwargs.keys():
            bp_signal = kwargs["bp_signal"]
        else:
            mask_index = np.random.choice(3, 1, p= (1 - self.hparams.mask_rate, 
                                                self.hparams.mask_rate / 2,
                                                self.hparams.mask_rate / 2))[0]
            bp_signal = [1, 1]
            if mask_index > 0: 
                bp_signal[mask_index - 1] = 0
        
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

        # EKF Propagation Reversed Jacobian = Latent-dim x forwards (I don't know, it just very expensive)
        z = torch.cat(srcs, dim=-1).detach()
        sigma_z = self.sigma_z_provider(z)  # (B, d_z, d_z)
        inv_alpha, beta, sigma_pred_sq = self.ekf_net(z, sigma_z, logits, signal=sigs)
        ekf_nll = self.unc_criterion(y_true=y, y_hat=logits, mu=logits, inv_alpha=inv_alpha, beta=beta)

        return loss, logits, y, \
                {"srcs": srcs, "recon_loss": recon_loss, "unc_loss": recon_unc_loss, "trace": recon_trace, "signal": bp_signal}, \
                {"var": bayescap_variance_1d(inv_alpha, beta), "loss": ekf_nll["loss"], "sigma_pred_sq": sigma_pred_sq.detach()}
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Tạm thời tắt reconstruction để đánh giá uncertainty
        loss, logits, y, recon, unc = self.model_step(batch, kwargs={"bp_signal": (1, 1)})
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"
        # update and log metrics
        self.train_loss(loss.mean())
        self.log(f"train/loss", 
                 self.train_loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True)
        
        self.train_recon_loss(recon["recon_loss"].mean())
        self.log(f"train/loss_recon_{signal_str}", 
                    self.train_recon_loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)
        
        self.train_unc_loss(recon["unc_loss"].mean())
        self.log(f"train/loss_recon_unc_{signal_str}", 
                    self.train_unc_loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)

        # Phase 1: Not propagating uncertainty of deficit inputs
        if self.current_epoch < self.hparams.epoch_phase and sum(signal) < 2:
            unc['loss'] *= 0
        
        self.train_nll(unc["loss"].mean())
        self.log(f"train/loss_unc_{signal_str}", 
                self.train_nll, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True)
        # return loss or backpropagation will fail, focus on uncertainty loss only

        # EKF diagnostics: mean = magnitude, std = per-sample spread.
        # std ~ 0 means the chain collapses sigma_pred to a constant.
        sps = unc["sigma_pred_sq"].detach().flatten()
        self.log(f"train/sigma_pred_sq_mean_{signal_str}", sps.mean(),
                 on_step=True, on_epoch=True)
        self.log(f"train/sigma_pred_sq_std_{signal_str}", sps.std(),
                 on_step=True, on_epoch=True)
        self.log(f"train/sigma_pred_sq_min_{signal_str}", sps.min(),
                 on_step=True, on_epoch=True)
        self.log(f"train/sigma_pred_sq_max_{signal_str}", sps.max(),
                 on_step=True, on_epoch=True)
        # return loss or backpropagation will fail, focus on uncertainty loss only

        return recon["unc_loss"].mean() + unc['loss'].mean()
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ) -> None:
        # Only check once
        if batch_idx == 1 and self.current_epoch == 0:
            # Check gradient at step
            print(f"Checking gradient for frozen model {self.net.__class__.__qualname__}")
            check_gradient(self.net)
            for item in self.controller.breakpoints:
                pos, bp = item['position'], item["breakpoint"]
                print(f"Checking {bp.name} module on {pos}: {bp.callback.__class__.__qualname__}")
                check_gradient(bp.callback)
            print(f"Checking gradient for Loss's Calibrator  {self.unc_criterion.__class__.__qualname__}")
            check_gradient(self.unc_criterion)
            print(f"Checking gradient for Loss's Calibrator  {self.ekf_net.__class__.__qualname__}")
            check_gradient(self.ekf_net)
            
        return super().optimizer_step(
                                        epoch,
                                        batch_idx,
                                        optimizer,
                                        optimizer_closure,
                                    )
        
    def on_validation_start(self) -> None:
        self.controller.eval()
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
            loss, logits, _, recon, unc = self.model_step(batch, kwargs={"bp_signal": (1, 1)})
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"
        self.val_loss(loss)
        self.log(f"val/loss", 
                 self.val_loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True)

        self.val_recon_loss(recon["recon_loss"].mean())
        self.log(f"val/loss_recon_{signal_str}", 
                    self.val_recon_loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)
        
        self.val_unc_loss(recon["unc_loss"].mean())
        self.log(f"val/loss_recon_unc_{signal_str}", 
                    self.val_unc_loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)
        

        self.val_nll(unc["loss"].mean())
        self.log(f"val/loss_unc_{signal_str}", 
                self.val_nll, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True)
        return (loss, logits, recon, unc)
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        score = self.val_nll.compute()  # get current val acc
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
        # update and log metrics
        self.test_loss(loss)
        self.log(f"test/loss", 
                 self.test_loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True)

        self.test_recon_loss(recon["recon_loss"].mean())
        self.log(f"train/loss_recon_{signal_str}", 
                    self.test_recon_loss, 
                    on_step=False, 
                    on_epoch=True, 
                    prog_bar=True)
        
        self.test_unc_loss(recon["unc_loss"].mean())
        self.log(f"test/loss_unc_{signal_str}", 
                    self.test_unc_loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True)

        
        self.test_nll(unc["loss"].mean())
        self.log(f"test/loss_unc_{signal_str}", 
                self.test_nll, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True)

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

        parameters = []
        for item in self.controller.breakpoints:
            bp = item["breakpoint"]
            print(f"Assigning {bp.name} breakpoints to Optimizer for update")
            parameters = parameters + list(bp.callback.parameters())

        # Loss also has learnable calibration params
        parameters += list(self.unc_criterion.parameters())
        parameters += list(self.ekf_net.inv_alpha_net.parameters())
        parameters += list(self.ekf_net.beta_net.parameters())
        optimizer = self.hparams.optimizer(params=parameters)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_unc_11",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}      

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        lit_state_dict: dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        filtered_dict = {k: v for k, v in lit_state_dict.items() if not k.startswith("recon_bp")}
        # print(list(filtered_dict.keys()))
        return filtered_dict

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

    
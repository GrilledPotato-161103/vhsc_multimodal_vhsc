from typing import Any, Dict, Tuple, Callable
from collections import defaultdict
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.plugins.hook import BreakpointController, Breakpoint
from src.plugins.head.bayescap import BayesCap1DLoss, bayescap_variance_1d

import functools
torch.serialization.add_safe_globals([functools.partial])

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

class ModelInjectModule(LightningModule):
    def __init__(self, 
                 net: nn.Module,
                 recon_bp: str,
                 unc_bp: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 controller: BreakpointController | None = None,
                 compile: bool = False,
                 recon_criterion: nn.Module | Callable | None = nn.MSELoss(),
                 unc_criterion: nn.Module | Callable | None = nn.MSELoss(),
                 epoch_phase: int = 20,
                 mask_rate: float = 0.3,
                 eta: float = 0.1,
                 n_jumps: int = 8
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["retcon_criterion", "unc_criterion", "net", "controller"])
        self.net = net
        self.controller = controller
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.test_recon_loss = MeanMetric()

        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()
        self.test_acc = MeanMetric()

        self.val_recon_best = MaxMetric()

        self.criterion = torch.nn.MSELoss()
        self.recon_criterion = recon_criterion
        self.unc_criterion = unc_criterion
        
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """
            Perform forward on hooked model
        """
        (x1, x2) = x
        return self.net(x1, x2)

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
        (x1, x2), y = batch
        x1 = x1.cuda()
        x2 = x2.cuda()
        y = y.cuda().unsqueeze(1)

        # Set kwargs for breakpoints, use cache if available
        if "bp_signal" in kwargs.keys():
            bp_signal = kwargs["bp_signal"]
        else:
            mask_index = np.random.choice(3, 1, p=(1 - self.hparams.mask_rate, 
                                                self.hparams.mask_rate / 2,
                                                self.hparams.mask_rate / 2))[0]
            bp_signal = [1, 1]
            if mask_index > 0: 
                bp_signal[mask_index - 1] = 0
        
        recon_bp = Breakpoint.get_by_name(self.hparams.recon_bp)
        recon_bp.kwargs = tuple(bp_signal)
        # print(recon_bp.kwargs)
        logits = self.forward((x1, x2)).unsqueeze(1)
        loss = self.criterion(logits, y)
        recon_trace = recon_bp.trace
        sigs = recon_trace.trace["signal"]
        recs = recon_trace.trace["reconstructed"]
        srcs = recon_trace.trace["input"]
        devs = recon_trace.trace["dev"]
        dists = recon_trace.trace["distance"]
        recon_loss = 0
        for sig, rec, src, dev, dist in zip(sigs, recs, srcs[::-1], devs, dists): 
            if sig == 0: 
                continue
            recon_loss += self.recon_criterion(rec, src) + self.recon_criterion(dev, dist) 
        
        unc_trace = Breakpoint.get_by_name(self.hparams.unc_bp).trace
        
        (mu, alpha, beta) = unc_trace.trace["output"]
        variance = bayescap_variance_1d(alpha, beta, target_dim=1, eps=1e-6)
        unc_loss = self.unc_criterion(mu, alpha, beta, logits, y)
        unc_loss = (unc_loss["loss"] + unc_loss["identity_loss"] + unc_loss["nll_loss"]) / 3

        return loss, logits, y, {"loss": recon_loss, "trace": recon_trace}, {"mu": mu, "var": variance, "loss": unc_loss}
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"
        # update and log metrics
        self.train_loss(loss)
        self.log(f"train/loss", 
                 self.train_loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True)

        self.train_recon_loss(recon["loss"])
        self.log(f"train/loss_recon_{signal_str}", 
                    self.train_recon_loss, 
                    on_step=False, 
                    on_epoch=True, 
                    prog_bar=True)
        
        
        self.train_acc(unc["loss"])
        self.log(f"train/loss_unc_{signal_str}", 
                self.train_acc, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True)
        
        # Phase 1: Not propagating uncertainty of deficit inputs
        if self.current_epoch < self.hparams.epoch_phase and sum(signal) < 2: 
            unc["loss"] *= 0

        # return loss or backpropagation will fail
        return loss + recon["loss"] + unc["loss"]

    def on_validation_start(self) -> None:
        self.controller.eval()
        print("Breakpoints are set to evaluation !!!")
        super().on_validation_start()
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        (x1_orig, x2_orig), y = batch
        kwargs = dict()
        # 1. BẮT BUỘC: Ép mở lại tính toán gradient trong Validation
        result = defaultdict(list)
        with torch.enable_grad():
            
            # Tạo bản sao của x1, x2 để không làm hỏng dữ liệu gốc
            # và bật requires_grad=True để theo dõi gradient
            x1 = x1_orig.clone().detach().requires_grad_(True)
            x2 = x2_orig.clone().detach().requires_grad_(True)
            # Sử dụng loss như 1 thang đo cho OOD
            # 2. Vòng lặp tấn công PGD (Gradient Ascent)
            for _ in range(self.hparams.n_jumps):
                # Xóa gradient cũ (nếu có) trước mỗi bước tính toán
                if x1.grad is not None: x1.grad.zero_()
                if x2.grad is not None: x2.grad.zero_()
                
                # Forward pass để tính loss
                loss, logits, y, recon, unc = self.model_step(((x1, x2), y), kwargs)
                
                signal = recon["trace"].trace["signal"]
                kwargs["bp_signal"] = signal
                # Lan truyền ngược để trích xuất gradient
                loss.backward()
                # Normalize gradient để trích xuất pha only
                grad_norm = torch.sqrt(x1.grad ** 2 + x2.grad ** 2)
                x1_jump = x1.grad.sign() / grad_norm
                x2_jump = x2.grad.sign() / grad_norm
                # Cập nhật vào bảng kết quả để đưa ra callback visualize
                result["losses"].append(loss.clone().detach())
                result["postion"].append([x1.clone().detach().item(), x2.clone().detach().item()])
                result["direction"].append([x1_jump.clone().detach().item(), x2_jump.clone().detach().item()])
                result["intensity"].append(grad_norm)
                result["uncertainty"].append(unc)
                # 3. Cập nhật dữ liệu x1, x2 để TĂNG loss
                # Thao tác này phải nằm trong no_grad để không bị theo dõi vào đồ thị
                with torch.no_grad():
                    # Đẩy lên dốc (Gradient Ascent)
                    x1_new = x1 + self.hparams.eta * x1_jump # [cite: 92]
                    x2_new = x2 + self.hparams.eta * x2_jump
                    # (Tùy chọn) Thêm bước Projection (cắt tỉa) nếu bạn muốn giới hạn nhiễu epsilon
                    # x1_new = torch.clamp(x1_new, x1_orig - epsilon, x1_orig + epsilon)
                    # x2_new = torch.clamp(x2_new, x2_orig - epsilon, x2_orig + epsilon)
                
                # Gán lại giá trị và bật requires_grad cho bước lặp tiếp theo
                x1 = x1_new.requires_grad_(True)
                x2 = x2_new.requires_grad_(True)

        # 4. Đánh giá lại mô hình trên dữ liệu đã bị tấn công (Adversarial Data)
        # Giờ x1, x2 đã trở thành dữ liệu xấu, ta tắt grad để đánh giá như bình thường
        with torch.no_grad():
            loss, logits, y, recon, unc = self.model_step(((x1, x2), y), kwargs=kwargs)
            result["losses"].append(loss.clone().detach())
            # N (B, 2)
            result["postion"].append(torch.stack([x1.clone().detach(), x2.clone().detach().item()], axis=-1))
            result["direction"].append(torch.stack([x1_jump.clone().detach(), x2_jump.clone().detach().item()], axis=-1))
            result["uncertainty"].append(unc)
        result["bp_signal"] = kwargs["bp_signal"]

        # Cached files
        # loss, logits, y, recon, unc = self.model_step(batch)
        signal = recon["trace"].trace["signal"]
        signal_str = f"{signal[0]}{signal[1]}"

        # Trả result để callback nhận và digest        
        return result
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_recon_loss.compute()  # get current val acc
        self.val_recon_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_recon_best", self.val_recon_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
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

        self.test_recon_loss(recon["loss"])
        self.log(f"train/loss_recon_{signal_str}", 
                    self.test_recon_loss, 
                    on_step=False, 
                    on_epoch=True, 
                    prog_bar=True)
        
        self.test_acc(unc["loss"])
        self.log(f"test/loss_unc_{signal_str}", 
                self.test_acc, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, DictConfig
    from functools import partial
 
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def main(cfg: DictConfig) -> None: 
        plugin_cfg = cfg.plugins
        print("Initializing model")
        model = torch.load(plugin_cfg.model_checkpoint, weights_only=False).cuda()
        model.requires_grad_(False)
        datamodule = instantiate(cfg.data)
        # print(type(datamodule)
        datamodule.setup()
        loader = datamodule.val_dataloader()
        data = iter(loader)
        batch = next(data)
        controller = BreakpointController.__init_dict__(model, plugin_cfg)
        controller.cuda()
        
        # module = ModelInjectModule(net=model, 
        #                            recon_bp="reconstructor.0", 
        #                            unc_bp="uncertainty.0",
        #                            optimizer=partial(torch.optim.Adam, lr=0.001, weight_decay=0.0),
        #                            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode=min, factor=0.1, patience=5),
        #                            controller=controller,
        #                            compile=False,
        #                            recon_criterion= nn.MSELoss(),
        #                            unc_criterion=BayesCap1DLoss(
        #                                                         lambda_identity=1.0,
        #                                                         lambda_nll=0.05,
        #                                                         identity_mode="l2",   # "l1" to mimic repo
        #                                                         nll_mode="paper",     # "repo" to mimic repo
        #                                                     ),
        #                             epoch_phase=10)
        module = instantiate(cfg.model)
        module = module(net=model, controller=controller)
        loss, logits, y, recon, unc = module.model_step(batch)
        print(loss)
        print(logits)
        print(y)
        print(recon)
        print(unc)
    main()

    
from typing import List, Callable, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
# from models.SURE.modules.mmbt import (
#     UnimodalBertEncoder, Reconstructor, ReconstructUncertainty, OutputSigma2
# )

from models.components.mmbt import MMBT
from models.SURE.losses.nce_loss import NCELoss, GaussianAlignLoss, OrderedEnforceLoss, WeightedCrossEntropyLoss
from models.SURE.trainers.model_evaluation_metrics import *


class MMBTLitModule(LightningModule):
    def __init__(self, 
                    net: nn.Module,
                    dataset: str,
                    train_params: Any,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler,
                    loss_type: str ="infonce",
                    use_fast_loading: bool =False,
                    compile: bool =False,
                    criterion: nn.Module = nn.CrossEntropyLoss(),
                    reconstruct_criterion: nn.Module = GaussianAlignLoss(),
                    output_criterion: nn.Module = GaussianAlignLoss(task='classification'),
                    order_criterion: nn.Module = OrderedEnforceLoss()
                    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        if self.hparams.compile:
            self.net.compile()
        
        # Criterion
        self.criterion = criterion
        self.reconstruction_criterion = reconstruct_criterion
        self.output_criterion = output_criterion
        self.order_criterion = order_criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
    
    def forward(self, x, out_uncertainty=False, save_embedding=False):
        return self.net.forward(x, 
                               out_uncertainty=out_uncertainty, 
                               save_embedding=save_embedding)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
    
    def loss_compute(self, 
                     pred,
                     target,
                     fused_uncertainty,
                     target_mask,
                     batch_repr,
                     recon_muys,
                     recon_dev,
                     temp,
                     beta,
                     use_fused_uncertainty=False):
        joint_mod_loss_sum = torch.tensor(0., device=pred.device)
        target_mask - target_mask.squeeze()
        # Mask for what
        whole_mask = torch.logical_and(target_mask[:, 0], target_mask[:, 1])
        partial_mask = torch.logical_or(target_mask[:, 0], target_mask[:, 1])
        if not use_fused_uncertainty:
            if torch.any(whole_mask):
                for mod in range(len(recon_muys)):
                    joint_mod_loss_sum += self.reconstruct_criterion(batch_repr[mod][whole_mask], 
                                                                     recon_dev[mod][whole_mask], 
                                                                     recon_dev[mod][whole_mask], 
                                                                     mod)

                joint_mod_loss_sum /= len(recon_muys)
                loss = torch.mean(joint_mod_loss_sum)
            else:
                loss = joint_mod_loss_sum
        else:
            reconstruct_loss = self.criterion(pred[partial_mask], target[partial_mask])
            try:
                reconstruct_loss.backward(retain_graph=True)
                # calculate the propagation uncertainty
                propagate_uncertainty = torch.zeros_like(fused_uncertainty)
                for mod in range(len(recon_muys)):
                    mask_i = torch.logical_and(~target_mask[:, mod], partial_mask)
                    # print(reconstruct_muys[mod].grad[mask_i].norm())
                    propagate_uncertainty[mask_i] = ((recon_muys[mod].grad[mask_i])**2 * recon_dev[mod][mask_i]).sum(dim=-1, keepdim=True)
                    recon_muys[mod].grad = None # clear the gradient
                    torch.cuda.empty_cache()
                
                # print('propagate_uncertainty: ', propagate_uncertainty.mean(), 'fused_uncertainty: ', fused_uncertainty.mean())
                fused_uncertainty += propagate_uncertainty
            except:
                pass
            joint_mod_loss_sum = self.output_criterion(target, 
                                                       pred, 
                                                       fused_uncertainty, 
                                                       -1, 
                                                       beta=beta)
            
            supervised_loss = self.criterion(pred[partial_mask], target[partial_mask])

            loss = torch.mean(supervised_loss + joint_mod_loss_sum)
        # loss = torch.mean(joint_mod_loss_sum)
        # loss = torch.mean(supervised_loss)

        if self.dataset == 'book':
            tqdm_dict = eval_book(pred, target, fused_uncertainty)

        tqdm_dict["loss"] = loss
        
        return loss, tqdm_dict
    
    def model_step(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        data, target_data, mask_data, _, _ = batch
        
        preds, \
            fused_uncertainty, \
                batch_repr, \
                    muys, \
                        dev = self.forward(data, 
                                            self.hparams.train_params.out_uncertainty)
        loss = self.loss_compute(preds,
                                 target_data,
                                 fused_uncertainty=fused_uncertainty,
                                 target_mask=mask_data,
                                 batch_repr=batch_repr,
                                 recon_muys=muys,
                                 recon_dev=dev,
                                 temp=self.hparams.train_params.temp,
                                 )
        return loss, preds, target_data
     
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        tqdm_dict = eval_book(preds, )
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
                                                            
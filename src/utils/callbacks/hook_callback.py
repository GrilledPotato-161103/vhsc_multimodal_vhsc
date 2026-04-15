import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import binned_statistic_2d

def pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    # Tính giá trị trung bình
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    # Tính độ lệch so với trung bình
    xm = x - mean_x
    ym = y - mean_y
    
    # Tính tử số (Covariance)
    numerator = torch.sum(xm * ym)
    
    # Tính mẫu số (Tích độ lệch chuẩn)
    # Thêm 1e-8 vào mẫu số để tránh lỗi chia cho 0 (Numerical stability)
    denominator = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + 1e-8
    
    return numerator / denominator

class AdversarialVizCallback(pl.Callback):
    def __init__(self, grid_size=20):
        super().__init__()
        self.grid_size = grid_size  # Độ phân giải của lưới (mesh)
        self.reset_states()

    def reset_states(self):
        self.positions = []
        self.directions = []
        self.intensities = []
        self.losses = []
        self.uncertainties = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0 or outputs is None or "postion" not in outputs:
            return
        # Lấy jump distance để tính loga của loss gain
        bp_signal = outputs["bp_signal"]
        # B, N
        losses = torch.stack(outputs["losses"], dim=1)
        # B, N, 2
        positions = torch.stack(outputs["positions"], dim=1)
        # Get loss gain
        losses_gain = torch.log(losses - losses[:, [0]])
        # N
        jumps = torch.arange(0, pl_module.hparams.n_jumps) * pl_module.hparams.eta
        # B, N
        jumps = torch.stack([jumps] * losses.shape[0], axis=0)
        # B, N, 2
        jumps_one = torch.stack([jumps, 1], dim=1)
        weights, _, _, _ = torch.linalg.lstsq(jumps_one.flatten(0), losses.flatten(0))

        # Get logarithm weight as correlation
        degree = weights[0]
        pl_module.log(f"val/loss_gain_on_{pl_module.hparams.eta:.2f}",
                        degree.detach().cpu().item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        
        variance = torch.stack([unc["var"].detach() for unc in outputs["uncertainty"]], dim=1)
        pcc = pearson_correlation(variance, losses)
        pl_module.log(f"val/loss_unc_pcc_{pl_module.hparams.eta:.2f}",
                        pcc.item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        
        self.positions.append(positions)
        self.directions.append(torch.stack(outputs["directions"], dim=1))
        self.intensities.append(torch.stack(outputs["intensity"], dim=1))
        self.losses.append(losses)
        self.uncertainties.extend(variance)

    def on_validation_epoch_end(self, trainer, pl_module):
        # B*, N, 2
        positions = torch.concatenate(self.positions, dim=0)
        directions = torch.concatenate(self.directions, dim=0)
        
        # B*, N, 1
        losses = torch.concatenate(self.losses, dim=0)
        uncertainties = torch.concatenate(self.uncertainties, dim=0)

        
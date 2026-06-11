import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import binned_statistic_2d
import wandb
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

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
    denominator = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + 1e-6
    return numerator / denominator

class AdversarialVizCallback(pl.Callback):
    def __init__(self, grid_size: int = 50, smooth: float = 2., x1_range  = [-1., 1.], x2_range = [-1., 1.]):
        super().__init__()
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.grid_size = grid_size # Độ phân giải của lưới (mesh)
        self.smooth = smooth
        self.reset_states()
        print("Visualizer Created")

    def reset_states(self):
        self.positions = []
        self.losses = []
        self.variances = []
        self.logits = []
        self.y = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # print(len(outputs))
        if pl_module.current_epoch < pl_module.hparams.epoch_phase:
            return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        loss, logits, recon, unc = outputs
        xs, y, positions = batch
        variance = unc['var']
        indices = torch.argsort(loss)
        pcc = pearson_correlation(loss[indices], variance[indices])
        pl_module.log(f"val/loss_unc_pcc",
                        pcc.item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        self.positions.append(positions)
        self.losses.append(loss)
        self.variances.append(variance)
        self.y.append(y)
        self.logits.append(logits)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        # print("Valid epoch end called", len(self.losses))
        if len(self.losses) <= 1:
            return super().on_validation_epoch_end(trainer, pl_module)
        # B*, N, 2
        positions = torch.concatenate(self.positions, dim=0).cpu().numpy().reshape(-1, 2)
        losses = torch.concatenate(self.losses, dim=0).cpu().numpy().flatten()
        variances = torch.concatenate(self.variances, dim=0).cpu().numpy().flatten()
        logits = torch.concatenate(self.logits, dim=0).cpu().numpy().flatten()
        gt = torch.concatenate(self.y, dim=0).cpu().numpy().flatten()
        x, y = positions[..., 0].flatten(), positions[..., 1].flatten()
        x_min, x_max = self.x1_range
        y_min, y_max = self.x2_range
        grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, self.grid_size), y_min:y_max:complex(0, self.grid_size)]

        # Hàm nội suy từ điểm phân tán lên lưới
        def rasterize(values):
            # Nội suy tuyến tính
            grid = griddata((x, y), values.flatten(), (grid_x, grid_y), method='linear', rescale=True, fill_value=0)
            # Xử lý các điểm NaN (ngoài rìa) bằng nearest neighbor
            nan_mask = np.isnan(grid) | np.isinf(grid) | (grid == 0)
            if np.any(nan_mask):
                grid_nearest = griddata((x, y), values.flatten(), (grid_x, grid_y), method='nearest', rescale=True)
                grid[nan_mask] = grid_nearest[nan_mask]
            
            print(grid.max(), grid.min(), grid.shape)
            return grid
        
        # Nội suy Loss và Variance
        logits_grids = rasterize(logits)
        y_grids = rasterize(y)
        logits_y_grid = np.concatenate([logits_grids, y_grids], axis=0)

        loss_grid = rasterize(losses)
        loss_grid = np.clip(loss_grid, 0, 20)

        var_grid = rasterize(variances)
        # NOTE: do not clip at a small ceiling — under OOD or head collapse
        # the variance can be much larger than 20, and clipping erases all
        # spatial structure into a uniform ceiling.
        var_grid = np.nan_to_num(var_grid, nan=0.0, posinf=1e6, neginf=0.0)

        # 4. Làm mịn và tính Covariance (Local Covariance)
        # E[L], E[V], E[L*V] thông qua Gaussian filter
        loss_smooth = gaussian_filter(loss_grid, sigma=self.smooth)
        var_smooth = gaussian_filter(var_grid, sigma=self.smooth)
        # 5. Vẽ Plotly Charts
        figs_to_log = {}

        # --- A. Quiver Plot (Trường Vector) ---
        # Lấy mẫu thưa hơn để biểu đồ không bị rối mịt mù
        # Hàm tiện ích vẽ Heatmap
        def create_heatmap(z_data, colorscale='viridis', ax = None, title=""):
            if isinstance(ax, Axes): 
                fig = None
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(z_data, ax=ax, cmap=colorscale,
                              cbar=False)
            if fig is not None:
                fig.colorbar(ax.collections[0], ax=ax, label="Value")
            ax.set_axis_off()
            ax.set_title(title)
            return fig

        # --- B. Loss Map ---
        # figs_to_log["val/Loss_Map"] = create_heatmap(loss_smooth, 'inferno')

        # --- C. Variance Map ---
        figs_to_log["val_plot/Log_Variance_Map"] = create_heatmap(np.log(var_smooth + 1e-6), "plasma", title="Log Variance Field")
        figs_to_log["val_plot/Logit_GT_Map"] = create_heatmap(logits_y_grid, 'plasma', title="Logits/GT Field")
        figs_to_log["val_plot/Log_Loss_Map"] = create_heatmap(np.log(loss_smooth + 1e-6), 'plasma', title="Log Loss Field")

        # 6. Push lên Weights & Biases
        # Đảm bảo trainer đang xài WandbLogger

        if "WandbLogger" in str(type(trainer.logger)):
            wandb_logger = trainer.logger.experiment
            log_dict = {
                name: wandb.Image(fig) for name, fig in figs_to_log.items()
            }
            log_dict["global_step"] = trainer.global_step
            log_dict["epoch"] = trainer.current_epoch
            
            wandb.log(log_dict)
        else:
            # Nếu chạy nội bộ không có wandb, lưu file html để debug
            print("Wandb Logger not found, saving HTML files instead...")
            for name, fig in figs_to_log.items():
                fig.write_html(f"{name.replace('/', '_')}_epoch_{trainer.current_epoch}.html")
        # Trả về state ban đầu
        self.reset_states()
        return super().on_validation_epoch_end(trainer, pl_module)



        
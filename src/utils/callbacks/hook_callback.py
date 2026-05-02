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
        self.directions = []
        self.intensities = []
        self.losses = []
        self.variances = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # print(f"DEBUG: Batch {batch_idx} ended.")
        # Lấy jump distance để tính loga của loss gain
        bp_signal = outputs["bp_signal"]
        # for key in outputs: 
        #     print(key, len(outputs[key]), outputs[key][0].shape if isinstance(outputs[key][0], torch.Tensor) else "")
        # import IPython; IPython.embed()
        # B*, N
        losses = torch.stack(outputs["losses"], dim=0)
        # B, N, 2
        positions = torch.stack(outputs["positions"], dim=0)
        # Get loss gain
        losses_gain = torch.log(losses - losses[:, [0]])
        # B, N
        jumps = torch.full_like(losses_gain, pl_module.hparams.eta)
        jumps[:, 0] = 0
        jumps = torch.cumsum(jumps, dim=1).to(losses_gain.device)
        # B, N, 2
        jumps_one = torch.nn.functional.pad(jumps, (0, 1), value=1)
        
        weights, _, _, _ = torch.linalg.lstsq(jumps_one.reshape(-1, 2), losses_gain.reshape(-1, 1))

        # Get logarithm weight as correlation
        degree = weights[0]
        pl_module.log(f"val/loss_gain_on_{pl_module.hparams.eta:.2f}",
                        degree.detach().cpu().item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        
        variance = torch.stack(outputs["variances"], dim=0)
        indices = torch.argsort(losses[:, -1])
        pcc = pearson_correlation(losses[indices], variance[indices])
        pl_module.log(f"val/loss_unc_pcc_{pl_module.hparams.eta:.2f}",
                        pcc.item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        
        self.positions.append(positions)
        self.directions.append(torch.stack(outputs["directions"], dim=0))
        self.intensities.append(torch.stack(outputs["intensities"], dim=0))
        self.losses.append(losses)
        self.variances.append(variance)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        # print("Valid epoch end called", len(self.losses))
        if len(self.losses) <= 1:
            return super().on_validation_epoch_end(trainer, pl_module)
        # B*, N, 2
        positions = torch.concatenate(self.positions, dim=0).cpu().numpy().reshape(-1, 2)
        directions = torch.concatenate(self.directions, dim=0).cpu().numpy().reshape(-1, 2)
        # B*, N, 1
        intensities = torch.concatenate(self.intensities, dim=0).cpu().numpy().flatten() * 5
        losses = torch.concatenate(self.losses, dim=0).cpu().numpy().flatten()
        variances = torch.concatenate(self.variances, dim=0).cpu().numpy().flatten()
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
        # 3. Tính toán và Rasterize
        # Tính vector U, V
        u = directions[:, 0] * intensities.flatten()
        v = directions[:, 1] * intensities.flatten()
        
        u_grid = rasterize(u)
        v_grid = rasterize(v)
        
        # Nội suy Loss và Variance
        loss_grid = rasterize(losses)
        loss_grid = np.clip(loss_grid, 0, 20)

        var_grid = rasterize(variances)
        var_grid = np.clip(var_grid, 0, 20)
        var_grid[np.isinf(var_grid)] = 20

        # 4. Làm mịn và tính Covariance (Local Covariance)
        # E[L], E[V], E[L*V] thông qua Gaussian filter
        loss_smooth = gaussian_filter(loss_grid, sigma=self.smooth)
        var_smooth = gaussian_filter(var_grid, sigma=self.smooth)
        loss_var_smooth = gaussian_filter(loss_grid * var_grid, sigma=self.smooth)
        
        # Cov(L, V) = E[LV] - E[L]E[V]
        cov_grid = loss_var_smooth - (loss_smooth * var_smooth)
        # 5. Vẽ Plotly Charts
        figs_to_log = {}

        # --- A. Quiver Plot (Trường Vector) ---
        # Lấy mẫu thưa hơn để biểu đồ không bị rối mịt mù
        # Hàm tiện ích vẽ Heatmap
        def create_heatmap(z_data, colorscale='viridis', ax = None):
            if isinstance(ax, Axes): 
                sns.heatmap(z_data, ax=ax, cmap=colorscale,
                              cbar=False)
                return 
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(z_data, ax=ax, cmap=colorscale,
                              cbar=False)
            fig.colorbar(ax.collections[0], ax=ax, label="Value")
            ax.set_axis_off()
            return fig

        # --- B. Loss Map ---
        # figs_to_log["val/Loss_Map"] = create_heatmap(loss_smooth, 'inferno')

        # --- C. Variance Map ---
        figs_to_log["val_plot/Log_Variance_Map"] = create_heatmap(np.log(var_smooth + 1e-6), 'plasma')
        

        # --- D. Covariance Map ---
        # Dùng màu có tính đối xứng (RdBu) vì covariance có thể âm hoặc dương
        figs_to_log["val_plot/Loss_Variance_Covariance_Map"] = create_heatmap(cov_grid, 'RdBu_r')

        fig_quiver, ax = plt.subplots(figsize=(8, 8))
        create_heatmap(np.log(loss_smooth + 1e-6), 'plasma', ax=ax)
        fig_quiver.colorbar(ax.collections[0], ax=ax, label="Value")
        slice_idx = (slice(None, None, 5), slice(None, None, 5))
        q = ax.quiver(
            (grid_y[slice_idx] - y_min) / (y_max - y_min) * self.grid_size, (grid_x[slice_idx] - x_min) / (x_max - x_min) * self.grid_size, 
            u_grid[slice_idx], v_grid[slice_idx],
            color='r',
            label="Adversarial vectors"
        )
        ax.set_title("Adversarial test")
        ax.set_axis_off()
        
        figs_to_log["val_plot/Loss_Log_Field"] = fig_quiver

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



        
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
    def __init__(self, grid_size: int = 50):
        super().__init__()
        self.grid_size = grid_size # Độ phân giải của lưới (mesh)
        self.reset_states()
        print("Visualizer Created")

    def reset_states(self):
        self.positions = []
        self.directions = []
        self.intensities = []
        self.losses = []
        self.variances = []
    
    # def on_validation_epoch_start(self, trainer, pl_module):
    #     print("Callback visited")
    #     return super().on_test_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0 or outputs is None or "postions" not in outputs:
            return
        # Lấy jump distance để tính loga của loss gain
        bp_signal = outputs["bp_signal"]
        for key in outputs: 
            print(key, len(outputs[key]), outputs[key][0].shape if isinstance(outputs[key][0], torch.Tensor) else "")
        # import IPython; IPython.embed()
        # B*, N
        losses = torch.stack(outputs["losses"], dim=0)
        # B, N, 2
        positions = torch.stack(outputs["positions"], dim=0)
        # Get loss gain
        losses_gain = torch.log(losses - losses[:, [0]])
        # N
        jumps = torch.arange(0, pl_module.hparams.n_jumps) * pl_module.hparams.eta
        # B, N
        jumps = torch.stack([jumps] * losses.shape[0], axis=0).to(losses.device)
        # B, N, 2
        jumps_one = torch.nn.functional.pad(jumps.unsqueeze(1), (0, 1), value=1)
        weights, _, _, _ = torch.linalg.lstsq(jumps_one.flatten(0), losses.flatten(0))

        # Get logarithm weight as correlation
        degree = weights[0]
        pl_module.log(f"val/loss_gain_on_{pl_module.hparams.eta:.2f}",
                        degree.detach().cpu().item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        
        variance = torch.stack(outputs["variances"], dim=0)
        pcc = pearson_correlation(variance, losses)
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
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        print("Valid epoch end called", len(self.losses))
        if len(self.losses) <= 1:
            return super().on_test_epoch_end(trainer, pl_module)
        # B*, N, 2
        positions = torch.concatenate(self.positions, dim=0).cpu().numpy().reshape(-1, 2)
        directions = torch.concatenate(self.directions, dim=0).cpu().numpy().reshape(-1, 2)
        
        # B*, N, 1
        intensities = torch.concatenate(self.intensities, dim=0).cpu().numpy().flatten()
        losses = torch.concatenate(self.losses, dim=0).cpu().numpy().flatten()
        variances = torch.concatenate(self.variances, dim=0).cpu().numpy().flatten()
        x, y = positions[:, 0]
        grid_x, grid_y = np.mgrid[:self.grid_size, :self.grid_size]

        # Hàm nội suy từ điểm phân tán lên lưới
        def rasterize(values):
            # Nội suy tuyến tính
            grid = griddata((x, y), values.flatten(), (grid_x, grid_y), method='linear')
            # Xử lý các điểm NaN (ngoài rìa) bằng nearest neighbor
            nan_mask = np.isnan(grid)
            if np.any(nan_mask):
                grid_nearest = griddata((x, y), values.flatten(), (grid_x, grid_y), method='nearest')
                grid[nan_mask] = grid_nearest[nan_mask]
            return grid
        # 3. Tính toán và Rasterize
        # Tính vector U, V
        u = directions[:, 0] * intensities.flatten()
        v = directions[:, 1] * intensities.flatten()
        
        u_grid = rasterize(u)
        v_grid = rasterize(v)
        
        # Nội suy Loss và Variance
        loss_grid = rasterize(losses)
        var_grid = rasterize(variances)

        # 4. Làm mịn và tính Covariance (Local Covariance)
        # E[L], E[V], E[L*V] thông qua Gaussian filter
        loss_smooth = gaussian_filter(loss_grid, sigma=self.smooth_sigma)
        var_smooth = gaussian_filter(var_grid, sigma=self.smooth_sigma)
        loss_var_smooth = gaussian_filter(loss_grid * var_grid, sigma=self.smooth_sigma)
        
        # Cov(L, V) = E[LV] - E[L]E[V]
        cov_grid = loss_var_smooth - (loss_smooth * var_smooth)

        # 5. Vẽ Plotly Charts
        figs_to_log = {}

        # --- A. Quiver Plot (Trường Vector) ---
        # Lấy mẫu thưa hơn để biểu đồ không bị rối mịt mù
        slice_idx = (slice(None, None, self.arrow_step), slice(None, None, self.arrow_step))
        fig_quiver = ff.create_quiver(
            grid_x[slice_idx], grid_y[slice_idx], 
            u_grid[slice_idx], v_grid[slice_idx],
            scale=0.05, arrow_scale=0.3, name='Gradient Vector'
        )
        fig_quiver.update_layout(title="Rasterized Vector Field", width=700, height=700)
        figs_to_log["Validation/Vector_Field"] = fig_quiver

        # Hàm tiện ích vẽ Heatmap
        def create_heatmap(z_data, title, colorscale='Viridis'):
            fig = go.Figure(data=go.Heatmap(
                z=z_data, x=grid_x[0, :], y=grid_y[:, 0],
                colorscale=colorscale
            ))
            fig.update_layout(title=title, width=700, height=700)
            return fig

        # --- B. Loss Map ---
        figs_to_log["val/Loss_Map"] = create_heatmap(loss_smooth, "Smoothed Loss Map", 'Inferno')

        # --- C. Variance Map ---
        figs_to_log["val/Variance_Map"] = create_heatmap(var_smooth, "Smoothed Variance Map", 'Plasma')

        # --- D. Covariance Map ---
        # Dùng màu có tính đối xứng (RdBu) vì covariance có thể âm hoặc dương
        figs_to_log["val/Covariance_Map"] = create_heatmap(cov_grid, "Local Covariance (Loss vs Variance)", 'RdBu_r')

        # 6. Push lên Weights & Biases
        # Đảm bảo trainer đang xài WandbLogger
        print(type(trainer.logger))

        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            print("Is Wandb logger")
            wandb_logger = trainer.logger.experiment
            log_dict = {
                name: wandb.Plotly(fig) for name, fig in figs_to_log.items()
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
        return super().on_test_epoch_end(trainer, pl_module)



        
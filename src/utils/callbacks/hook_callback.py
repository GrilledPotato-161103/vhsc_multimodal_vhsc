import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import binned_statistic_2d

class AdversarialVizCallback(pl.Callback):
    def __init__(self, grid_size=20):
        super().__init__()
        self.grid_size = grid_size  # Độ phân giải của lưới (mesh)
        self.reset_states()

    def reset_states(self):
        self.positions = []
        self.directions = []
        self.intensity = []
        self.losses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0 or outputs is None or "postion" not in outputs:
            return
        # Lấy jump distance để tính loga của loss gain
        jump_distance = pl_module.hparams.eta * pl_module.hparams.n_jumps
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
        
        
        pl_module.log(f"val/loss_gain_on_{jump_distance:.2f}",
                        degree.detach().cpu().item(), 
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False)
        
        

        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.x1_list:
            return

        # 1. Chuyển đổi toàn bộ dữ liệu thành mảng NumPy 1D
        x1_all = torch.cat(self.x1_list).numpy().flatten()
        x2_all = torch.cat(self.x2_list).numpy().flatten()
        u_all = torch.cat(self.u_list).numpy().flatten()
        v_all = torch.cat(self.v_list).numpy().flatten()

        if trainer.logger and hasattr(trainer.logger.experiment, "log"):
            wandb_logger = trainer.logger.experiment
            current_epoch = trainer.current_epoch

            # 2. Xây dựng không gian lưới (Mesh)
            x_bins = np.linspace(x1_all.min(), x1_all.max(), self.grid_size)
            y_bins = np.linspace(x2_all.min(), x2_all.max(), self.grid_size)

            # 3. Tổng hợp (Aggregate) Gradient trung bình vào từng ô lưới
            # Những ô không có dữ liệu sẽ trả về np.nan
            u_grid, x_edges, y_edges, _ = binned_statistic_2d(
                x1_all, x2_all, u_all, statistic='mean', bins=[x_bins, y_bins]
            )
            v_grid, _, _, _ = binned_statistic_2d(
                x1_all, x2_all, v_all, statistic='mean', bins=[x_bins, y_bins]
            )

            # Lấy tọa độ tâm của từng ô lưới để làm điểm đặt mũi tên
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            X_mesh, Y_mesh = np.meshgrid(x_centers, y_centers, indexing='ij')

            # Loại bỏ các ô lưới trống (chứa nan)
            valid_mask = ~np.isnan(u_grid)
            X_plot = X_mesh[valid_mask]
            Y_plot = Y_mesh[valid_mask]
            U_plot = u_grid[valid_mask]
            V_plot = v_grid[valid_mask]

            # 4. Trực quan hóa Vector Field
            fig_vec, ax_vec = plt.subplots(figsize=(8, 8))
            
            # Tính toán độ lớn để ánh xạ vào màu sắc (mũi tên dài/ngắn sẽ tự động do giá trị U, V quyết định)
            magnitude = np.sqrt(U_plot**2 + V_plot**2)

            # Vẽ đồ thị Quiver:
            # - Không chuẩn hóa u_norm, v_norm như trước để giữ nguyên độ dài đại diện cho độ lớn gradient.
            # - Dùng pivot='mid' để tâm mũi tên nằm đúng vào tâm ô lưới.
            quiver = ax_vec.quiver(
                X_plot, Y_plot, U_plot, V_plot, magnitude, 
                cmap='coolwarm', alpha=0.9, pivot='mid'
            )
            
            plt.colorbar(quiver, ax=ax_vec, label='Gradient Magnitude')
            ax_vec.set_title(f"Aggregated Gradient Vector Field (Epoch {current_epoch})")
            ax_vec.set_xlabel("X1")
            ax_vec.set_ylabel("X2")
            
            # Làm đẹp giao diện: Thêm lưới chìm để quan sát rõ hơn
            ax_vec.grid(True, linestyle='--', alpha=0.5)

            # 5. Log lên W&B và dọn dẹp bộ nhớ
            wandb_logger.log({
                "Validation/Aggregated_Gradient_Field": wandb.Image(fig_vec),
                "global_step": trainer.global_step
            })

            plt.close(fig_vec)

        # Reset states cho epoch tiếp theo
        self.reset_states()
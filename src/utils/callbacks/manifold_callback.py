from typing import Sequence
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.axes import Axes

def correlation_score(x: torch.Tensor, y: torch.Tensor, mode="pcc"):
    """Compute Pearson correlation coefficient between two tensors."""
    # Pearson or Spearman mode
    assert mode in ["pcc", "spearman"], "Correlation Mode must be either Spearman or Pearson"
    if mode == "pcc":
        indices = torch.argsort(x)
        x_sorted, y_sorted = x[indices], y[indices]
    else:
        x_sorted = x.argsort().argsort().float()
        y_sorted = y.argsort().argsort().float() 
    # Handle Correlation    
    mean_x = torch.mean(x_sorted)
    mean_y = torch.mean(y_sorted)
    xm = x_sorted - mean_x
    ym = y_sorted - mean_y
    numerator = torch.sum(xm * ym)
    denominator = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + 1e-6
    return numerator / denominator

def _cell_means(positions, values, n_bins):
            """Return (cell_mean_per_sample, cell_level_means, cell_ids, counts)."""
            p = positions.float()
            lo = p.amin(0); hi = p.amax(0)
            idx = ((p - lo) / (hi - lo + 1e-8) * (n_bins - 1)).long().clamp(0, n_bins - 1)
            flat = idx[:, 0] * n_bins + idx[:, 1]
            n_cells = n_bins * n_bins
            sums = torch.zeros(n_cells).scatter_add_(0, flat, values.float())
            cnts = torch.zeros(n_cells).scatter_add_(0, flat, torch.ones_like(values.float()))
            cell_mean = sums / cnts.clamp_min(1.0)
            return cell_mean[flat], cell_mean, flat, cnts

def _cell_correlation(pos, x, y, n_bins):
            _, var_cells, _, cnts = _cell_means(pos, x, n_bins)
            _, err_cells, _, _    = _cell_means(pos, y, n_bins)
            mask = cnts > 0
            return correlation_score(var_cells[mask], err_cells[mask], mode="pcc"), correlation_score(var_cells[mask], err_cells[mask], mode="spearman")

class ManifoldVizCallback(pl.Callback):
    """Validation callback that visualises the loss, variance, and logit fields
    over the latent manifold z.

    For z_dim > 2, z is projected to 2D via t-SNE before rasterisation.
    """

    def __init__(
        self,
        grid_size: int = 50,
        smooth: float = 2.0,
        z_range=[[-1, 1]],
        z_dim: int = 2,
        tsne_perplexity: float = 30.0,
        tsne_random_state: int = 42,
    ):
        super().__init__()
        if len(z_range) == 1:
            z_range = list(z_range) * z_dim

        self.z_range = z_range
        self.z_dim = z_dim
        self.grid_size = grid_size
        self.smooth = smooth
        self.tsne_perplexity = tsne_perplexity
        self.tsne_random_state = tsne_random_state
        self.reset_states()
        print("Visualizer Created")

    def reset_states(self):
        self.z_positions = []
        self.losses = []
        self.variances = []
        self.logits = []
        self.y = []

    def _project_to_2d(self, z: np.ndarray) -> np.ndarray:
        """Project latent z to 2D for visualisation.

        For z_dim == 2 the raw coordinates are returned.
        For z_dim > 2, t-SNE projects to 2D.
        """
        if self.z_dim == 2:
            return z[:, 0], z[:, 1]

        perplexity = min(self.tsne_perplexity, len(z) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.tsne_random_state,
        )
        z_2d = tsne.fit_transform(z)
        return z_2d[:, 0], z_2d[:, 1]

    def _compute_grid_bounds(self, x_2d: np.ndarray, y_2d: np.ndarray):
        """Return (x_min, x_max, y_min, y_max) for the rasterisation grid."""
        if self.z_dim == 2:
            # Use the known manifold range for 2D case
            (x_min, x_max) = self.z_range[0]
            (y_min, y_max) = self.z_range[1]
        else:
            # Derive bounds from t-SNE projection with 5 % padding
            x_min, x_max = x_2d.min(), x_2d.max()
            y_min, y_max = y_2d.min(), y_2d.max()
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            x_min -= x_pad
            x_max += x_pad
            y_min -= y_pad
            y_max += y_pad
        return x_min, x_max, y_min, y_max

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if pl_module.current_epoch < pl_module.hparams.epoch_phase:
            return super().on_validation_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
        loss, logits, recon, unc = outputs
        # ManifoldToyDataset returns (xs_noisy, y, xs, z)
        xs_noisy, y, xs_orig, z = batch
        variance = unc["var"]
        pcc = correlation_score(loss, variance, mode= "pcc")
        pl_module.log(
            "val/loss_unc_pcc_step",
            pcc.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.z_positions.append(z)
        self.losses.append(loss)
        self.variances.append(variance)
        self.y.append(y)
        self.logits.append(logits)
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self.losses) <= 1:
            return super().on_validation_epoch_end(trainer, pl_module)

        # Concatenate all batches — z has shape (N, z_dim)
        z = torch.cat(self.z_positions, dim=0)  # (N, z_dim)
        losses = torch.cat(self.losses, dim=0).flatten()
        variances = torch.cat(self.variances, dim=0).flatten()
        logits = torch.cat(self.logits, dim=0).flatten()
        gt = torch.cat(self.y, dim=0).flatten()
        # Dataset-scoped PCC Correlation
        pcc = correlation_score(losses, variances, mode="pcc")
        spearman_cc = correlation_score(losses, variances, mode="spearman")
        pl_module.log(
            "val/loss_unc_pcc_epoch",
            pcc.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        pl_module.log(
            "val/loss_unc_spearman_epoch",
            spearman_cc.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # Project latent z → 2D (identity for z_dim == 2, t-SNE otherwise)
        x_2d, y_2d = self._project_to_2d(z)
        # Build rasterisation grid
        x_min, x_max, y_min, y_max = self._compute_grid_bounds(x_2d, y_2d)
        grid_x, grid_y = np.mgrid[
            x_min : x_max : complex(0, self.grid_size),
            y_min : y_max : complex(0, self.grid_size),
        ]

        def rasterize(values):
            grid = griddata(
                (x_2d, y_2d),
                values.flatten(),
                (grid_x, grid_y),
                method="linear",
                rescale=True,
                fill_value=0,
            )
            nan_mask = np.isnan(grid) | np.isinf(grid) | (grid == 0)
            if np.any(nan_mask):
                grid_nearest = griddata(
                    (x_2d, y_2d),
                    values.flatten(),
                    (grid_x, grid_y),
                    method="nearest",
                    rescale=True,
                )
                grid[nan_mask] = grid_nearest[nan_mask]
            return grid

        # Interpolate fields onto the grid
        logits_grid = rasterize(logits.cpu().numpy())
        y_grid = rasterize(gt.cpu().numpy())
        logits_y_grid = np.concatenate([logits_grid, y_grid], axis=0)

        loss_grid = rasterize(losses.cpu().numpy())
        loss_grid = np.clip(loss_grid, 0, 20)

        var_grid = rasterize(variances.cpu().numpy())
        var_grid = np.nan_to_num(var_grid, nan=0.0, posinf=1e6, neginf=0.0)

        # Gaussian smoothing for cleaner visualisation
        loss_smooth = gaussian_filter(loss_grid, sigma=self.smooth)
        var_smooth = gaussian_filter(var_grid, sigma=self.smooth)

        figs_to_log = {}

        def create_heatmap(z_data, colorscale="viridis", ax=None, title=""):
            if isinstance(ax, Axes):
                fig = None
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(z_data, ax=ax, cmap=colorscale, cbar=False)
            if fig is not None:
                fig.colorbar(ax.collections[0], ax=ax, label="Value")
            ax.set_axis_off()
            ax.set_title(title)
            return fig

        figs_to_log["val_plot/Log_Variance_Map"] = create_heatmap(
            np.log(var_smooth + 1e-6), "plasma", title="Log Variance Field"
        )
        figs_to_log["val_plot/Logit_GT_Map"] = create_heatmap(
            logits_y_grid, "plasma", title="Logits/GT Field"
        )
        figs_to_log["val_plot/Log_Loss_Map"] = create_heatmap(
            np.log(loss_smooth + 1e-6), "plasma", title="Log Loss Field"
        )

        # Log to Weights & Biases or save HTML fallback
        if "WandbLogger" in str(type(trainer.logger)):
            wandb_logger = trainer.logger.experiment
            log_dict = {
                name: wandb.Image(fig) for name, fig in figs_to_log.items()
            }
            log_dict["global_step"] = trainer.global_step
            log_dict["epoch"] = trainer.current_epoch
            wandb.log(log_dict)
        else:
            print("Wandb Logger not found, saving HTML files instead...")
            for name, fig in figs_to_log.items():
                fig.write_html(
                    f"{name.replace('/', '_')}_epoch_{trainer.current_epoch}.html"
                )
        self.reset_states()
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if pl_module.current_epoch < pl_module.hparams.epoch_phase:
            return super().on_test_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
        loss, logits, recon, unc = outputs
        # ManifoldToyDataset returns (xs_noisy, y, xs, z)
        xs_noisy, y, xs_orig, z = batch
        variance = unc["var"]
        pcc = correlation_score(loss, variance, mode= "pcc")
        pl_module.log(
            "test/loss_unc_pcc_step",
            pcc.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.z_positions.append(z)
        self.losses.append(loss)
        self.variances.append(variance)
        self.y.append(y)
        self.logits.append(logits)
        return super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
    
    def on_test_epoch_end(self, trainer, pl_module):
        if len(self.losses) <= 1:
            return super().on_test_epoch_end(trainer, pl_module)

        # Concatenate all batches — z has shape (N, z_dim)
        z = torch.cat(self.z_positions, dim=0)  # (N, z_dim)
        losses = torch.cat(self.losses, dim=0).flatten()
        variances = torch.cat(self.variances, dim=0).flatten()
        logits = torch.cat(self.logits, dim=0).flatten()
        gt = torch.cat(self.y, dim=0).flatten()
        # Dataset-scoped PCC Correlation
        pcc = correlation_score(losses, variances, mode="pcc")
        spearman_cc = correlation_score(losses, variances, mode="spearman")
        pl_module.log(
            "test/loss_unc_pcc_epoch",
            pcc.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        pl_module.log(
            "test/loss_unc_spearman_epoch",
            spearman_cc.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # Project latent z → 2D (identity for z_dim == 2, t-SNE otherwise)
        x_2d, y_2d = self._project_to_2d(z.cpu().numpy())
        # Build rasterisation grid
        x_min, x_max, y_min, y_max = self._compute_grid_bounds(x_2d, y_2d)
        grid_x, grid_y = np.mgrid[
            x_min : x_max : complex(0, self.grid_size),
            y_min : y_max : complex(0, self.grid_size),
        ]

        def rasterize(values):
            grid = griddata(
                (x_2d, y_2d),
                values.flatten(),
                (grid_x, grid_y),
                method="linear",
                rescale=True,
                fill_value=0,
            )
            nan_mask = np.isnan(grid) | np.isinf(grid) | (grid == 0)
            if np.any(nan_mask):
                grid_nearest = griddata(
                    (x_2d, y_2d),
                    values.flatten(),
                    (grid_x, grid_y),
                    method="nearest",
                    rescale=True,
                )
                grid[nan_mask] = grid_nearest[nan_mask]
            return grid

        # Interpolate fields onto the grid
        logits_grid = rasterize(logits.cpu().numpy())
        y_grid = rasterize(gt.cpu().numpy())
        logits_y_grid = np.concatenate([logits_grid, y_grid], axis=0)

        loss_grid = rasterize(losses.cpu().numpy())
        loss_grid = np.clip(loss_grid, 0, 20)

        var_grid = rasterize(variances.cpu().numpy())
        var_grid = np.nan_to_num(var_grid, nan=0.0, posinf=1e6, neginf=0.0)

        # Gaussian smoothing for cleaner visualisation
        loss_smooth = gaussian_filter(loss_grid, sigma=self.smooth)
        var_smooth = gaussian_filter(var_grid, sigma=self.smooth)

        figs_to_log = {}

        def create_heatmap(z_data, colorscale="viridis", ax=None, title=""):
            if isinstance(ax, Axes):
                fig = None
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(z_data, ax=ax, cmap=colorscale, cbar=False)
            if fig is not None:
                fig.colorbar(ax.collections[0], ax=ax, label="Value")
            ax.set_axis_off()
            ax.set_title(title)
            return fig

        figs_to_log["val_plot/Log_Variance_Map"] = create_heatmap(
            np.log(var_smooth + 1e-6), "plasma", title="Log Variance Field"
        )
        figs_to_log["val_plot/Logit_GT_Map"] = create_heatmap(
            logits_y_grid, "plasma", title="Logits/GT Field"
        )
        figs_to_log["val_plot/Log_Loss_Map"] = create_heatmap(
            np.log(loss_smooth + 1e-6), "plasma", title="Log Loss Field"
        )

        # Log to Weights & Biases or save HTML fallback
        if "WandbLogger" in str(type(trainer.logger)):
            wandb_logger = trainer.logger.experiment
            log_dict = {
                name: wandb.Image(fig) for name, fig in figs_to_log.items()
            }
            log_dict["global_step"] = trainer.global_step
            log_dict["epoch"] = trainer.current_epoch
            wandb.log(log_dict)
        else:
            print("Wandb Logger not found, saving HTML files instead...")
            for name, fig in figs_to_log.items():
                fig.write_html(
                    f"{name.replace('/', '_')}_epoch_{trainer.current_epoch}.html"
                )
        self.reset_states()
        return super().on_test_epoch_end(trainer, pl_module)
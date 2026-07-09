"""Validation callback for witnessing the shared embedding space quality.

Collects embeddings from all modalities during validation and produces
visual diagnostics: modality agreement matrix, t-SNE projections, and
reconstruction quality (when decoders are available).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
from torch import Tensor

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from lightning import Callback, LightningModule, Trainer


class SharedEmbeddingCallback(Callback):
    """Visualise the quality of the learned shared embedding space.

    At each validation epoch end this callback:
    1. Computes a modality-agreement matrix (pairwise cosine similarity).
    2. Projects shared embeddings to 2D via t-SNE, coloured by target y.
    3. Projects per-modality embeddings to 2D to check alignment.
    4. Tracks per-modality reconstruction MSE (if decoders exist).
    """

    def __init__(
        self,
        tsne_perplexity: float = 30.0,
        tsne_random_state: int = 42,
        log_interval_epochs: int = 1,
        n_tsne_samples: int = 1000,
    ) -> None:
        super().__init__()
        self.tsne_perplexity = tsne_perplexity
        self.tsne_random_state = tsne_random_state
        self.log_interval_epochs = log_interval_epochs
        self.n_tsne_samples = n_tsne_samples
        self._reset_states()

    def _reset_states(self) -> None:
        self._shared_embs: List[Tensor] = []
        self._modality_embs: List[List[Tensor]] = []  # [modality_idx][batch]
        self._y_values: List[Tensor] = []
        self._has_decoders: bool = False

    # ------------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        xs_noisy, y, _xs_clean, _z = batch

        # Re-run forward to get embeddings (use clean xs if available)
        modalities = [xs_noisy[:, i, :] for i in range(xs_noisy.shape[1])]
        if hasattr(pl_module, 'net'):
            shared, per_mod_embs = pl_module.net(modalities)
        else:
            return

        self._shared_embs.append(shared.detach().cpu())
        self._y_values.append(y.detach().cpu())

        if not self._modality_embs:
            self._modality_embs = [[] for _ in range(len(per_mod_embs))]
        for i, emb in enumerate(per_mod_embs):
            self._modality_embs[i].append(emb.detach().cpu())

        if hasattr(pl_module.net, 'decoders') and pl_module.net.decoders is not None:
            self._has_decoders = True

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.log_interval_epochs != 0:
            self._reset_states()
            return

        if len(self._shared_embs) == 0:
            self._reset_states()
            return

        # Concatenate
        shared = torch.cat(self._shared_embs, dim=0).numpy()  # (N, D)
        y_vals = torch.cat(self._y_values, dim=0).flatten().numpy()  # (N,)

        mod_embs_np = []
        for i in range(len(self._modality_embs)):
            if self._modality_embs[i]:
                mod_embs_np.append(
                    torch.cat(self._modality_embs[i], dim=0).numpy()
                )
            else:
                mod_embs_np.append(np.array([]))

        figs: dict[str, plt.Figure] = {}

        # 1. Modality agreement matrix
        fig_agree = self._plot_agreement_matrix(mod_embs_np)
        if fig_agree:
            figs["val_plot/Modality_Agreement"] = fig_agree

        # 2. t-SNE of shared embeddings (subsample if large)
        fig_tsne = self._plot_tsne_shared(shared, y_vals, pl_module.current_epoch)
        if fig_tsne:
            figs["val_plot/Shared_Embedding_tSNE"] = fig_tsne

        # 3. t-SNE of modality-specific embeddings (alignment check)
        fig_align = self._plot_tsne_alignment(mod_embs_np, y_vals, pl_module.current_epoch)
        if fig_align:
            figs["val_plot/Modality_Alignment_tSNE"] = fig_align

        # Log
        self._log_figures(trainer, figs)

        self._reset_states()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_agreement_matrix(
        self, mod_embs: List[np.ndarray]
    ) -> Optional[plt.Figure]:
        """n_modals × n_modals heatmap of mean pairwise cosine similarity."""
        n = len(mod_embs)
        if n < 2:
            return None

        agreement = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if mod_embs[i].size == 0 or mod_embs[j].size == 0:
                    agreement[i, j] = np.nan
                    continue
                e_i = mod_embs[i] / (np.linalg.norm(mod_embs[i], axis=-1, keepdims=True) + 1e-8)
                e_j = mod_embs[j] / (np.linalg.norm(mod_embs[j], axis=-1, keepdims=True) + 1e-8)
                agreement[i, j] = (e_i * e_j).sum(axis=-1).mean()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            agreement,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            xticklabels=[f"M{i}" for i in range(n)],
            yticklabels=[f"M{i}" for i in range(n)],
        )
        ax.set_title("Modality Agreement (Cosine Similarity)")
        fig.tight_layout()
        return fig

    def _plot_tsne_shared(
        self, shared: np.ndarray, y: np.ndarray, epoch: int
    ) -> Optional[plt.Figure]:
        """t-SNE of shared embeddings, coloured by regression target y."""
        N = shared.shape[0]
        if N < 5 or shared.shape[1] <= 2:
            return None

        # Subsample for speed
        if N > self.n_tsne_samples:
            idx = np.random.default_rng(self.tsne_random_state).choice(
                N, self.n_tsne_samples, replace=False
            )
            shared = shared[idx]
            y = y[idx]

        perplexity = min(self.tsne_perplexity, len(shared) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.tsne_random_state,
        )
        emb_2d = tsne.fit_transform(shared)

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=y, cmap="plasma", s=5, alpha=0.6,
        )
        ax.set_title(f"Shared Embedding t-SNE (epoch {epoch})")
        ax.set_axis_off()
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("y (normalised)")
        fig.tight_layout()
        return fig

    def _plot_tsne_alignment(
        self, mod_embs: List[np.ndarray], y: np.ndarray, epoch: int
    ) -> Optional[plt.Figure]:
        """t-SNE showing all modality embeddings.  Same sample across
        modalities should land at the same point if alignment is good."""
        n = len(mod_embs)
        if n < 2:
            return None

        # Subsample
        N = mod_embs[0].shape[0]
        if N > self.n_tsne_samples:
            idx = np.random.default_rng(self.tsne_random_state).choice(
                N, self.n_tsne_samples, replace=False
            )
            y_sub = y[idx]
            mod_embs = [embs[idx] for embs in mod_embs]
        else:
            y_sub = y

        # Concatenate all modality embeddings for joint t-SNE
        all_embs = np.concatenate(mod_embs, axis=0)  # (n_modals * N_sub, D)
        if all_embs.shape[0] < 5 or all_embs.shape[1] <= 2:
            return None

        perplexity = min(self.tsne_perplexity, all_embs.shape[0] - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.tsne_random_state,
        )
        all_2d = tsne.fit_transform(all_embs)

        fig, ax = plt.subplots(figsize=(7, 6))
        markers = ['o', 's', '^', 'D', 'v']
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

        for i in range(n):
            emb_2d = all_2d[i * len(y_sub): (i + 1) * len(y_sub)]
            ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                marker=markers[i % len(markers)],
                color=colors[i],
                s=8, alpha=0.5,
                label=f"Modality {i}",
            )

        ax.set_title(f"Modality Embedding Alignment (epoch {epoch})")
        ax.legend(markerscale=2)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_figures(trainer: Trainer, figs: dict[str, plt.Figure]) -> None:
        """Log figures to WandB or fall back to console."""
        logger_name = type(trainer.logger).__name__ if trainer.logger else ""
        if "Wandb" in logger_name:
            import wandb
            log_dict = {}
            for name, fig in figs.items():
                log_dict[name] = wandb.Image(fig)
            log_dict["epoch"] = trainer.current_epoch
            log_dict["global_step"] = trainer.global_step
            trainer.logger.experiment.log(log_dict)  # type: ignore[union-attr]
        else:
            for name, fig in figs.items():
                safe_name = name.replace("/", "_")
                fpath = f"{safe_name}_epoch_{trainer.current_epoch}.png"
                fig.savefig(fpath, dpi=100)
                print(f"[SharedEmbeddingCallback] Saved {fpath}")

        # Close figures to avoid memory leaks
        for fig in figs.values():
            plt.close(fig)

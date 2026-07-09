"""Stage 1 LightningModule: shared embedding learning.

Trains per-modality encoders to produce a shared embedding space using
contrastive InfoNCE loss between modality pairs and optional reconstruction
regularisation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmetrics import MeanMetric, MinMetric

from lightning import LightningModule

from src.models.components.shared_embedding import SharedEmbeddingNet


class SharedEmbeddingModule(LightningModule):
    """Train encoders to map all modalities into a shared embedding space.

    Loss = contrastive_weight * InfoNCE + recon_weight * MSE_recon

    Expected batch: ``(xs_noisy, y, xs_clean, z)`` from ProjectedManifoldDataset.

    - xs_noisy: ``(B, n_modals, d_proj)``
    - y: ``(B, 1)``
    - xs_clean: ``(B, n_modals, d_proj)``
    - z: ``(B, emb_dim)``
    """

    def __init__(
        self,
        net: SharedEmbeddingNet,
        optimizer: Any,
        scheduler: Any | None = None,
        compile_model: bool = False,
        contrastive_temperature: float = 0.1,
        contrastive_weight: float = 1.0,
        recon_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        # Metrics
        self.train_cont_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.val_cont_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_cont_best = MinMetric()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self.hparams.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, xs: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        return self.net(xs)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    @staticmethod
    def _contrastive_loss(
        embeddings: List[Tensor],
        temperature: float,
    ) -> Tensor:
        """InfoNCE loss over all unordered modality pairs.

        For each pair (i, j), the similarity matrix ``sim_ij ∈ R^{B×B}`` is
        computed from L2-normalised embeddings.  The diagonal elements are
        positive pairs; all others are negatives.  Cross-entropy is applied
        in both directions (i→j and j→i).

        Returns:
            Scalar loss averaged over all pairs and directions.
        """
        n_modals = len(embeddings)
        if n_modals < 2:
            return torch.tensor(0.0, device=embeddings[0].device)

        B = embeddings[0].shape[0]
        device = embeddings[0].device
        labels = torch.arange(B, device=device)

        total_loss = torch.tensor(0.0, device=device)
        n_comparisons = 0

        for i in range(n_modals):
            for j in range(i + 1, n_modals):
                e_i = F.normalize(embeddings[i], dim=-1)
                e_j = F.normalize(embeddings[j], dim=-1)
                sim = (e_i @ e_j.T) / temperature  # (B, B)
                total_loss += F.cross_entropy(sim, labels)
                total_loss += F.cross_entropy(sim.T, labels)
                n_comparisons += 2

        return total_loss / n_comparisons

    def _reconstruction_loss(
        self, shared: Tensor, modalities: List[Tensor]
    ) -> Tensor:
        """MSE between each decoded modality and its original observation."""
        if self.net.decoders is None:
            return torch.tensor(0.0, device=shared.device)

        loss = torch.tensor(0.0, device=shared.device)
        for i, x in enumerate(modalities):
            x_recon = self.net.decode(shared, i)
            loss += F.mse_loss(x_recon, x)
        return loss / len(modalities)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def model_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Dict[str, Any]:
        xs_noisy, _y, _xs_clean, _z = batch
        # xs_noisy: (B, n_modals, d_proj) → list of (B, d_proj)
        modalities = [xs_noisy[:, i, :] for i in range(xs_noisy.shape[1])]

        shared, embeddings = self.forward(modalities)

        cont_loss = self._contrastive_loss(
            embeddings, self.hparams.contrastive_temperature
        )
        recon_loss = self._reconstruction_loss(shared, modalities)

        total = (
            self.hparams.contrastive_weight * cont_loss
            + self.hparams.recon_weight * recon_loss
        )

        return {
            "loss": total,
            "cont_loss": cont_loss,
            "recon_loss": recon_loss,
            "shared": shared.detach(),
            "embeddings": [e.detach() for e in embeddings],
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        out = self.model_step(batch)
        self.train_cont_loss(out["cont_loss"])
        self.train_recon_loss(out["recon_loss"])

        self.log("train/loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/cont_loss", self.train_cont_loss, on_step=True, on_epoch=True)
        self.log("train/recon_loss", self.train_recon_loss, on_step=True, on_epoch=True)
        return out["loss"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        out = self.model_step(batch)
        self.val_cont_loss(out["cont_loss"])
        self.val_recon_loss(out["recon_loss"])

        self.log("val/loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cont_loss", self.val_cont_loss, on_step=False, on_epoch=True)
        self.log("val/recon_loss", self.val_recon_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        score = self.val_cont_loss.compute()
        self.val_cont_best(score)
        self.log("val/cont_best", self.val_cont_best.compute(), sync_dist=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        out = self.model_step(batch)
        self.log("test/loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cont_loss", out["cont_loss"], on_step=False, on_epoch=True)
        self.log("test/recon_loss", out["recon_loss"], on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/cont_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        # Also save just the net for easy Stage 2 loading
        cache_path = self.hparams.get("cache_path", "checkpoints/stage1_net.pth")
        import os
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(self.net, cache_path)

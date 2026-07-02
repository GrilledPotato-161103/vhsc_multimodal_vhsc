"""Multi-modal JEPA wrapper with hook-compatible module tree.

Provides a clean ``nn.Module`` tree for attaching
:class:`~src.plugins.hook_dag.BreakpointController` hooks to a frozen
Neuro-JEPA ViT backbone and cross-attention classifier.

The wrapper mirrors the architecture of
:class:`~src.models.components.toy.MultiModalRegressor`: per-modality
encoder wrappers feed into a shared fusion head.

Module tree
-----------
::

    MultiModalJEPARegressor
    ├── backbone (VisionTransformer)              # Shared frozen ViT
    ├── encoders (nn.ModuleList)                  # Per-modality hook targets
    │   ├── 0 (ModalExtractor)                    # "encoders.0" after → mod-0 ViT features
    │   └── 1 (ModalExtractor)                    # "encoders.1" after → mod-1 ViT features
    └── classifier (MultiModalLateFusion)          # Cross-attn fusion (from cross_attn.py)
        ├── proj1, proj2 (ProjectionHead)
        ├── cross_attn_1to2, cross_attn_2to1 (CrossAttention)
        ├── norm1-4 (LayerNorm)
        ├── gate (Sequential) or fusion (Linear)
        └── classifier (nn.Linear → num_classes)

Pretrained loading
------------------
Backbone weights are loaded from HuggingFace Hub (``NYUMedML/Neuro-JEPA``)
or a local checkpoint via :func:`init_utils.load_backbone_from_hf`.
The classifier is built from scratch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Make Neuro-JEPA importable
_NEUROJEPA_ROOT = Path(__file__).resolve().parents[3] / "submodules" / "Neuro-JEPA" / "src"
if str(_NEUROJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEUROJEPA_ROOT))


# ---------------------------------------------------------------------------
# Pass-through module for per-modality hook attachment
# ---------------------------------------------------------------------------


class ModalExtractor(nn.Module):
    """Identity pass-through whose only purpose is to serve as a named
    hook-attachment target inside ``encoders`` ModuleList.

    Placed after the shared ViT backbone so breakpoints can capture
    per-modality token features independently.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Modality-flexible fusion: mean aggregation after per-modality projection
# ---------------------------------------------------------------------------

class ModalityFusion(nn.Module):
    """Fusion head that accepts an arbitrary subset of modalities.

    Each present modality is projected through a shared-structure linear
    block, then all projected features are **mean-aggregated** and passed
    through a classifier.  Missing modalities are simply skipped.

    Parameters
    ----------
    embed_dim : int
        ViT token embedding dimension (768 for vit_base).
    fusion_dim : int
        Dimension after per-modality projection (default 256).
    num_classes : int
        Number of output classes.
    modality_keys : Sequence[str]
        Fixed modality names, e.g. ``["t1", "t2", "flair"]``.
    dropout : float
        Dropout after projection activation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        fusion_dim: int = 256,
        num_classes: int = 2,
        modality_keys: Sequence[str] = ("t1", "t2", "flair"),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_keys = list(modality_keys)
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

        # Per-modality projection (same architecture, independent weights)
        self.projections = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(embed_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for key in self.modality_keys
        })

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor | None],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fuse an arbitrary subset of modality features.

        Parameters
        ----------
        features : dict
            Maps modality key → ``[B, N, embed_dim]`` token grid, or ``None``
            if that modality is missing.  Token grids are mean-pooled to
            ``[B, embed_dim]`` before projection.

        Returns
        -------
        logits : [B, num_classes]
        info : dict with keys ``"fused"``, ``"present"``, ``"projected"``
        """
        projected = []
        present = []

        for key in self.modality_keys:
            f = features.get(key)
            if f is None:
                continue
            # Mean-pool token grid → single vector per sample
            if f.dim() == 3:
                f = f.mean(dim=1)  # [B, N, D] → [B, D]
            projected.append(self.projections[key](f))
            present.append(key)

        if not projected:
            raise ValueError(
                f"At least one modality must be present. "
                f"Got all None for keys {self.modality_keys}."
            )

        # Mean aggregation across present modalities
        fused = torch.stack(projected, dim=0).mean(dim=0)  # [B, fusion_dim]
        logits = self.head(fused)

        return logits, {
            "fused": fused,
            "present": present,
            "projected": projected,
        }
# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class MultiModalJEPARegressor(nn.Module):
    """Multi-modal JEPA regressor / classifier with hook-compatible module tree.

    Supports a fixed set of modalities (default ``["t1", "t2", "flair"]``)
    with **arbitrary subsets present at inference time**.  Missing modalities
    are skipped; fusion is mean-aggregation of per-modality projections.

    Parameters
    ----------
    backbone : VisionTransformer
        Pretrained Neuro-JEPA ViT (shared, frozen).
    classifier : ModalityFusion or MultiModalLateFusion
        Fusion head.  If a legacy 2-modality ``MultiModalLateFusion`` is
        passed, the wrapper falls back to the old ``feats[0], feats[1]``
        forward path for backward compatibility.
    modality_keys : Sequence[str]
        Fixed modality names, e.g. ``["t1", "t2", "flair"]``.
    image_size : tuple
        Spatial input size ``(D, H, W)`` expected by the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        modality_keys: Sequence[str] = ("t1", "t2", "flair"),
        image_size: Tuple[int, int, int] = (96, 108, 96),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.modality_keys = list(modality_keys)
        self.n_modals = len(self.modality_keys)
        self.image_size = image_size

        # Per-modality identity wrappers — hook targets for BreakpointController
        self.encoders = nn.ModuleList([
            ModalExtractor() for _ in range(self.n_modals)
        ])

        # Freeze backbone (hooks train separately)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Detect fusion type for backward-compat routing
        self._use_legacy_fusion = not isinstance(classifier, ModalityFusion)

        # Infer num_classes
        if hasattr(self.classifier, "head") and isinstance(self.classifier.head, nn.Sequential):
            last = self.classifier.head[-1]
            self._num_classes = last.out_features if isinstance(last, nn.Linear) else 2
        elif hasattr(self.classifier, "classifier") and isinstance(self.classifier.classifier, nn.Linear):
            self._num_classes = self.classifier.classifier.out_features
        else:
            self._num_classes = 2

    @property
    def num_classes(self) -> int:
        return self._num_classes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: Dict[str, torch.Tensor | None] | List[torch.Tensor] | Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        images : dict or list/tuple
            If dict: maps modality key → ``[B,C,D,H,W]`` tensor or ``None``.
            ``None`` values are skipped (modality missing).
            If list/tuple: tensors in ``modality_keys`` order (backward compat).

        Returns
        -------
        logits : ``[B, num_classes]``
        """
        # Normalize to dict form
        if isinstance(images, dict):
            image_dict = images
        else:
            image_list = list(images)
            image_dict = {
                k: image_list[i] if i < len(image_list) else None
                for i, k in enumerate(self.modality_keys)
            }

        # Extract ViT features per modality (skip None / missing)
        feats: Dict[str, torch.Tensor | None] = {}
        for i, key in enumerate(self.modality_keys):
            img = image_dict.get(key)
            if img is None:
                feats[key] = None
                continue
            # Ensure 5D
            if img.dim() == 4:
                img = img.unsqueeze(2)  # [B,C,H,W] → [B,C,1,H,W]
            with torch.no_grad():
                f: torch.Tensor = self.backbone(img)
                if isinstance(f, tuple):
                    f = f[0]  # (tokens, moe_scores) → tokens
            # Pass through modal wrapper (hook target)
            f = self.encoders[i](f)
            feats[key] = f

        # Fuse
        if self._use_legacy_fusion:
            # Legacy 2-modality path
            present = [feats[k] for k in self.modality_keys if feats[k] is not None]
            logits: torch.Tensor = self.classifier(present[0], present[1])
        else:
            logits, _ = self.classifier(feats)

        return logits


# ---------------------------------------------------------------------------
# Factory — load from HuggingFace Hub or local checkpoint
# ---------------------------------------------------------------------------


def _download_hf_checkpoint(
    repo_id: str,
    filename: str = "model.safetensors",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = True,
    subfolder: str | None = None,
) -> str:
    """Download a file from HuggingFace Hub using the standalone API.

    Avoids importing ``neurojepa.utils.init_utils`` which triggers the
    ``torchmetrics → transformers`` import chain and has a strict
    ``huggingface-hub<1.0`` version constraint.
    """
    from huggingface_hub import hf_hub_download

    download_kwargs: Dict[str, Any] = {"repo_id": repo_id, "filename": filename}
    if revision is not None:
        download_kwargs["revision"] = revision
    if cache_dir is not None:
        download_kwargs["cache_dir"] = cache_dir
    if token is not None and token is not True:
        download_kwargs["token"] = token
    if subfolder is not None:
        download_kwargs["subfolder"] = subfolder

    return hf_hub_download(**download_kwargs)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip known wrapper prefixes from checkpoint state dict keys.

    Replicates the logic from
    ``neurojepa.utils.init_utils._clean_backbone_state_dict`` without
    importing that module.
    """
    prefix_order = (
        "student.vision_encoder.",
        "student.encoder.",
        "vision_encoder.",
        "target_encoder.",
        "encoder.",
        "model.",
        "module.",
        "_orig_mod.",
        "backbone.",
    )
    cleaned: Dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        clean_name = name.replace("_checkpoint_wrapped_module.", "")
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefix_order:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
                    stripped = True
        cleaned[clean_name] = value
    return cleaned


def _extract_state_dict(
    checkpoint: Dict[str, Any],
    checkpoint_key: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Extract backbone state dict from a checkpoint dict.

    Replicates ``neurojepa.utils.init_utils._extract_backbone_state_dict``.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict.")
    if checkpoint_key is not None:
        if checkpoint_key not in checkpoint:
            raise ValueError(f"Checkpoint missing key '{checkpoint_key}'.")
        return checkpoint[checkpoint_key]
    for key in ("encoder", "target_encoder", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _load_checkpoint_file(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint file (safetensors or torch .pt/.pth)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=device)
    return torch.load(path, map_location=device, weights_only=False)


def build_jepa_regressor(
    model_name_or_path: str = "NYUMedML/Neuro-JEPA",
    device: torch.device | str = "cpu",
    modality_keys: Sequence[str] = ("t1", "t2", "flair"),
    image_size: Tuple[int, int, int] = (96, 108, 96),
    num_classes: int = 2,
    fusion_dim: int = 256,
    freeze_backbone: bool = True,
    use_legacy_fusion: bool = False,
    hf_token: str | bool | None = True,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    **backbone_overrides,
) -> MultiModalJEPARegressor:
    """Build a :class:`MultiModalJEPARegressor` from a pretrained checkpoint.

    Downloads the ViT backbone from HuggingFace Hub
    (default: ``NYUMedML/Neuro-JEPA``) or loads from a local path.
    The fusion head is built from scratch.

    By default uses the new :class:`ModalityFusion` (mean-aggregation of
    per-modality linear projections) which supports arbitrary modality
    subsets.  Pass ``use_legacy_fusion=True`` for the old 2-modality
    ``MultiModalLateFusion`` (bidirectional cross-attention + gated fusion).

    Parameters
    ----------
    model_name_or_path : str
        HF Hub repo ID or local checkpoint file/directory path.
    device : str or torch.device
    modality_keys : Sequence[str]
        Fixed modality names (default ``["t1", "t2", "flair"]``).
    image_size : tuple
        ``(D, H, W)`` spatial input size.
    num_classes : int
    fusion_dim : int
        Hidden dim of per-modality projections (new fusion only).
    freeze_backbone : bool
    use_legacy_fusion : bool
        If True, use the old 2-modality ``MultiModalLateFusion``.
    hf_token / hf_revision / hf_cache_dir
        Passed to HF Hub download.

    Returns
    -------
    MultiModalJEPARegressor
    """
    import neurojepa.models.vision_transformer as vit

    is_local = os.path.exists(model_name_or_path)
    is_hf = not is_local and "/" in model_name_or_path

    # --- Resolve checkpoint file ---
    if is_hf:
        for fname in ("model.safetensors", "pytorch_model.bin"):
            try:
                ckpt_path = _download_hf_checkpoint(
                    repo_id=model_name_or_path,
                    filename=fname,
                    revision=hf_revision,
                    cache_dir=hf_cache_dir,
                    token=hf_token,
                    subfolder=None,
                )
                break
            except Exception:
                ckpt_path = None
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Could not download checkpoint from HF Hub: {model_name_or_path}"
            )
    elif is_local:
        ckpt_path = model_name_or_path
        if os.path.isdir(ckpt_path):
            for fname in ("model.safetensors", "pytorch_model.bin"):
                candidate = os.path.join(ckpt_path, fname)
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break
    else:
        raise ValueError(
            f"Cannot resolve model: {model_name_or_path}. "
            "Use a HF Hub ID like 'NYUMedML/Neuro-JEPA' or a local file path."
        )

    print(f"Loading checkpoint from: {ckpt_path}")

    # --- Load and clean state dict ---
    checkpoint = _load_checkpoint_file(ckpt_path, device=str(device))
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _clean_state_dict(state_dict)

    # Infer embed_dim from position embedding
    embed_dim = 768  # default vit_base
    for k, v in state_dict.items():
        if "pos_embed" in k and v.dim() >= 3:
            embed_dim = v.shape[-1]
            break
    print(f"Inferred embed_dim={embed_dim}")

    # --- Build backbone ---
    backbone = vit.vit_base(
        img_size=image_size,
        patch_size=(12, 12, 12),
        in_chans=1,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_rope=True,
        use_activation_checkpointing=True,
        **backbone_overrides,
    )
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"Backbone load_state_dict: {msg}")
    backbone.to(device)
    del checkpoint, state_dict

    # --- Build classifier / fusion ---
    if use_legacy_fusion:
        from neurojepa.models.cross_attn import MultiModalLateFusion

        classifier = MultiModalLateFusion(
            embed_dim=embed_dim,
            proj_dim=512,
            num_heads=8,
            num_tokens=32,
            num_classes=num_classes,
            fusion_type="gate",
        ).to(device)
    else:
        classifier = ModalityFusion(
            embed_dim=embed_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            modality_keys=modality_keys,
            dropout=0.1,
        ).to(device)

    if freeze_backbone:
        backbone.requires_grad_(False)
        backbone.eval()

    wrapper = MultiModalJEPARegressor(
        backbone=backbone,
        classifier=classifier,
        modality_keys=modality_keys,
        image_size=image_size,
    )
    wrapper.to(device)
    return wrapper


# ---------------------------------------------------------------------------
# Thin wrapper around the official Neuro-JEPA loading API
# ---------------------------------------------------------------------------

def load_backbone(
    model_name_or_path: str = "NYUMedML/Neuro-JEPA",
    device: str | torch.device = "cpu",
    **kwargs,
) -> nn.Module:
    """Load a pretrained Neuro-JEPA ViT backbone using the official API.

    This wraps ``neurojepa.utils.init_utils.load_backbone_from_hf``, which
    reads ``config.json`` from the HF repo to auto-detect architecture
    (vit_base / vit_large) and MoE settings.

    Parameters
    ----------
    model_name_or_path : str
        HF Hub repo ID (``"NYUMedML/Neuro-JEPA"``) or local checkpoint path.
    device : str or torch.device
    **kwargs
        Forwarded to ``load_backbone_from_hf`` (revision, cache_dir, token, etc.).

    Returns
    -------
    VisionTransformer
        Bare ViT backbone — no ModalExtractor wrappers, no classifier.
    """
    from neurojepa.utils.init_utils import load_backbone_from_hf

    return load_backbone_from_hf(
        model_name_or_path=model_name_or_path,
        device=str(device),
        **kwargs,
    )

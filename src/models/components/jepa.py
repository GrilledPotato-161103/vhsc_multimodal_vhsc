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
# Main wrapper
# ---------------------------------------------------------------------------


class MultiModalJEPARegressor(nn.Module):
    """Multi-modal JEPA regressor / classifier with hook-compatible module tree.

    Parameters
    ----------
    backbone:
        Pretrained Neuro-JEPA VisionTransformer (shared, frozen).
    classifier:
        Cross-attention ``MultiModalLateFusion`` from
        ``neurojepa.models.cross_attn``.
    modality_keys:
        Ordered modality names, e.g. ``["t1w", "t2w"]``.
    image_size:
        Spatial input size ``(D, H, W)`` expected by the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        modality_keys: Sequence[str] = ("mod_0", "mod_1"),
        image_size: Tuple[int, int, int] = (96, 108, 96),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.modality_keys = list(modality_keys)
        self.n_modals = len(self.modality_keys)
        self.image_size = image_size

        # Per-modality identity wrappers — hook targets for BreakpointController.
        # Hooking "encoders.0" (after) captures modality-0 ViT token features,
        # "encoders.1" (after) captures modality-1 ViT token features.
        self.encoders = nn.ModuleList([
            ModalExtractor() for _ in range(self.n_modals)
        ])

        # Freeze backbone (hooks train separately)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Infer num_classes from classifier
        if hasattr(self.classifier, "classifier") and isinstance(self.classifier.classifier, nn.Linear):
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
        images: Dict[str, torch.Tensor] | List[torch.Tensor] | Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward pass returning classification logits.

        Each modality is processed through the shared backbone independently
        so that per-modality breakpoints on ``encoders.*`` fire separately.

        Parameters
        ----------
        images:
            Either a dict mapping modality key → ``[B, C, D, H, W]`` tensor,
            or a list/tuple of tensors in ``modality_keys`` order.

        Returns
        -------
        logits:
            ``[B, num_classes]``.
        """
        if isinstance(images, dict):
            image_list = [images[k] for k in self.modality_keys]
        else:
            image_list = list(images)

        # Extract ViT features per modality
        feats: List[torch.Tensor] = []
        for i, img in enumerate(image_list):
            # Ensure 5D
            if img.dim() == 4:
                img = img.unsqueeze(2)  # [B,C,H,W] → [B,C,1,H,W]
            with torch.no_grad():
                f: torch.Tensor = self.backbone(img)
                if isinstance(f, tuple):
                    f = f[0]  # (tokens, moe_scores) → tokens
            # Pass through modal wrapper (hook target)
            f = self.encoders[i](f)
            feats.append(f)

        # Fuse via cross-attention classifier
        logits: torch.Tensor = self.classifier(feats[0], feats[1])
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
    modality_keys: Sequence[str] = ("t1w", "t2w"),
    image_size: Tuple[int, int, int] = (96, 108, 96),
    num_classes: int = 2,
    freeze_backbone: bool = True,
    hf_token: str | bool | None = True,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    **backbone_overrides,
) -> MultiModalJEPARegressor:
    """Build a :class:`MultiModalJEPARegressor` from a pretrained checkpoint.

    Downloads the ViT backbone from HuggingFace Hub
    (default: ``NYUMedML/Neuro-JEPA``) or loads from a local path.
    The classifier head is built from scratch.

    This function **does not** import ``neurojepa.utils.init_utils``
    (which triggers ``torchmetrics → transformers`` and fails on
    ``huggingface-hub>=1.0``).  Instead it uses ``huggingface_hub``
    directly and replicates the key state-dict cleaning logic inline.

    Parameters
    ----------
    model_name_or_path:
        HF Hub repo ID or local checkpoint file/directory path.
    device:
        Target device (cpu or cuda).
    modality_keys:
        Ordered modality names.
    image_size:
        ``(D, H, W)`` spatial input size.
    num_classes:
        Number of output classes.
    freeze_backbone:
        If True, freeze ViT backbone parameters.
    hf_token:
        HF token.  ``True`` = use cached login; string = explicit token;
        ``None`` = no auth.
    hf_revision:
        Optional HF branch/commit.
    hf_cache_dir:
        Optional HF cache directory.

    Returns
    -------
    MultiModalJEPARegressor
    """
    import neurojepa.models.vision_transformer as vit
    from neurojepa.models.cross_attn import MultiModalLateFusion

    is_local = os.path.exists(model_name_or_path)
    is_hf = not is_local and "/" in model_name_or_path

    # --- Resolve checkpoint file ---
    if is_hf:
        # Try model.safetensors first, then pytorch_model.bin
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
        # If it's a directory, look for model.safetensors or pytorch_model.bin
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

    # --- Build classifier ---
    classifier = MultiModalLateFusion(
        embed_dim=embed_dim,
        proj_dim=512,
        num_heads=8,
        num_tokens=32,
        num_classes=num_classes,
        fusion_type="gate",
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

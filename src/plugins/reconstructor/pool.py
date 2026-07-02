"""Pre/post-processing functions for ViT token-grid features.

The Neuro-JEPA ViT backbone outputs [B, N_tokens, embed_dim] token grids.
These pooling adapters reduce token sequences to flat latent vectors before
feeding reconstructors or downstream consumers.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from src.plugins.var import BreakpointContext, BreakpointOutput


class MeanPoolCollectedFn(nn.Module):
    """``pre_fn`` that mean-pools each collected tensor along dim=1 (token axis).

    ViT encoder outputs are [B, N, D] token grids.  This collapses N→1 via
    mean pooling and packs the resulting [B, D] vectors into ``ctx.inputs``
    as a tuple, matching the format expected by
    :class:`~src.plugins.reconstructor.linear.BilinearReconstructor`.
    """

    def forward(self, ctx: BreakpointContext) -> BreakpointContext:
        tensors = [
            v.mean(dim=1) for _, v in sorted(ctx.collected.items())
            if isinstance(v, torch.Tensor)
        ]
        if tensors:
            ctx.inputs = tuple(tensors)
        return ctx


class MeanPoolInputFn(nn.Module):
    """``pre_fn`` that mean-pools ``ctx.inputs[0]`` along dim=1 (token axis).

    Use when a breakpoint's ``data_sources`` is ``["input"]`` and the hook
    data is a token grid that must be collapsed before the callback runs.
    """

    def forward(self, ctx: BreakpointContext) -> BreakpointContext:
        if ctx.inputs is not None and len(ctx.inputs) > 0:
            x = ctx.inputs[0]
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                ctx.inputs = (x.mean(dim=1),)
        return ctx


# ---------------------------------------------------------------------------
# Multimodal missing-modality imputation via mean aggregation
# ---------------------------------------------------------------------------


class MapModalityCollectedFn(nn.Module):
    """``pre_fn`` that maps source BP names to modality keys and mean-pools.

    Source breakpoints (e.g. ``src_t1.0``) push ViT token grids
    ``[B, N, 768]`` to the reconstructor's buffer.  This function:

    1. Strips the ``.N`` index suffix from each source BP name.
    2. Maps the base name to a modality key via ``key_map``.
    3. Mean-pools each token grid → ``[B, 768]`` flat latent.
    4. Packages results as ``ctx.inputs = ({mod_key: latent},)``.

    Missing modalities (source BP never fired because input was ``None``)
    are simply absent from the dict.

    Parameters
    ----------
    key_map : Dict[str, str]
        Mapping from source BP base names to modality keys, e.g.
        ``{"src_t1": "t1", "src_t2": "t2", "src_flair": "flair"}``.
    """

    def __init__(self, key_map: Dict[str, str] | None = None):
        super().__init__()
        self.key_map = dict(key_map) if key_map else {}

    def forward(self, ctx: BreakpointContext) -> BreakpointContext:
        pooled: Dict[str, torch.Tensor] = {}
        for bp_name, tensor in ctx.collected.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            base = bp_name.split(".")[0]
            modality = self.key_map.get(base, base)
            # Mean-pool token grid [B, N, D] → [B, D]
            if tensor.dim() == 3:
                tensor = tensor.mean(dim=1)
            pooled[modality] = tensor
        ctx.inputs = (pooled,)
        return ctx


class MeanImputeReconstructor(nn.Module):
    """Fill missing modalities via mean aggregation of available latents.

    Expects ``ctx.inputs[0]`` to be a dict of present modality latents
    ``{key: [B, D], ...}``.  For any key in the fixed modality topology
    that is absent, imputes it as the element-wise mean of all available
    latents.

    Returns a ``BreakpointOutput`` whose ``.output`` is a 1-tuple
    ``(complete_dict,)`` — compatible with ``mutate=true`` on a
    ``before`` breakpoint.

    Parameters
    ----------
    modality_keys : Sequence[str]
        Fixed modality topology, e.g. ``["t1", "t2", "flair"]``.
    """

    def __init__(self, modality_keys: Sequence[str] = ("t1", "t2", "flair")):
        super().__init__()
        self.modality_keys = list(modality_keys)

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        present = ctx.inputs[0] if isinstance(ctx.inputs, tuple) else ctx.inputs

        available = [v for v in present.values() if isinstance(v, torch.Tensor)]
        if not available:
            raise ValueError(
                "MeanImputeReconstructor requires at least one present modality."
            )
        mean_latent = torch.stack(available, dim=0).mean(dim=0)  # [B, D]

        complete: Dict[str, torch.Tensor] = {}
        imputed: List[str] = []
        for key in self.modality_keys:
            if key in present and present[key] is not None:
                complete[key] = present[key]
            else:
                complete[key] = mean_latent
                imputed.append(key)

        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            context=ctx,
            output=(complete,),  # 1-tuple for before-mutate compatibility
            trace={
                "imputed": imputed,
                "present": list(present.keys()),
                "complete": list(complete.keys()),
            },
        )

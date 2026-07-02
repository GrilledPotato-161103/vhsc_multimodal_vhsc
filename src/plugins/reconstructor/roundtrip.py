"""Round-trip hook components for in-place effect feedback DAG.

These components implement the "round-trip" pattern where encoder outputs
are captured, processed through a pipeline (reconstructor, uncertainty
estimator), and the processed results are written back to shared state —
creating the effect of in-place replacement at the encoder output level.

Architecture
------------
::

    encoders.0 ─► src_enc0 (StateWriterHook)  ─► collector._buffer
    encoders.1 ─► src_enc1 (StateWriterHook)  ─► collector._buffer
                                                   │
                          ┌────────────────────────┘
                          ▼
                   collector (RoundTripCollector)
                     │
                     ├─ reconstructor processes both latents
                     ├─ overwrites controller.state for masked modalities
                     └─ pushes result to uncertainty (optional)
                          │
                          ▼
                   head_mediator (StateReadbackFn)
                     │
                     └─ reads processed state, mutate=true → head receives corrected latents
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.plugins.var import BreakpointContext, BreakpointOutput


class StateWriterHook(nn.Module):
    """Source BP callback: writes encoder output to controller.state, passes through.

    Use as the ``callback`` on source breakpoints (e.g. ``encoders.0``,
    ``encoders.1``, ...).  Writes the encoder's output to ``ctx.state[key]``
    so that downstream consumers (``RoundTripCollector``, ``StateReadbackFn``)
    can access it.

    The output is passed through unchanged so the DAG can push it to
    downstream breakpoints via ``data_sinks``.

    Parameters
    ----------
    state_key:
        Key in ``controller.state`` to write to, e.g. ``"z0"``.
    """

    def __init__(self, state_key: str):
        super().__init__()
        self.state_key = state_key

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        ctx.state[self.state_key] = ctx.output
        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            context=ctx,
            output=ctx.output,
            trace={"state_key": self.state_key},
        )


class RoundTripCollector(nn.Module):
    """Terminal collector: processes collected latents and feeds back to state.

    Wraps a reconstructor.  After reconstruction:

    - For **masked** modalities (``signal[i] == 0``): overwrites
      ``controller.state[key]`` with the reconstructed latent.
    - For **available** modalities (``signal[i] == 1``): keeps the original.

    The reconstructor receives ``ctx.inputs`` prepared by a ``pre_fn``
    (e.g. ``ToListCollectedFn`` which packs collected tensors into a tuple).

    Parameters
    ----------
    reconstructor:
        An ``nn.Module`` whose ``forward(ctx)`` returns a ``BreakpointOutput``
        with ``.output = (rec_0, rec_1, ...)``.
    state_keys:
        Ordered list of state keys matching the modalities, e.g.
        ``["z0", "z1"]``.
    """

    def __init__(self, reconstructor: nn.Module, state_keys: List[str]):
        super().__init__()
        self.reconstructor = reconstructor
        self.state_keys = state_keys

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        rec_result = self.reconstructor(ctx)
        recs = rec_result.output  # (rec_0, rec_1, ...)
        signal = ctx.bp_kwargs    # (p0, p1, ...) from random masking

        for i, key in enumerate(self.state_keys):
            if signal[i] == 0:
                ctx.state[key] = recs[i]

        return rec_result


class StateReadbackFn(nn.Module):
    """Head mediator: reads processed latents from state, replaces head input.

    Placed as a **before**-hook on the head module with ``mutate=true``.
    Ignores the original head input (which may contain stale encoder outputs
    from the Python list comprehension) and instead reads from
    ``controller.state``.

    Parameters
    ----------
    state_keys:
        Ordered list of state keys, e.g. ``["z0", "z1"]``.
    combine:
        How to aggregate latents before passing to the head:

        - ``"sum"``: ``torch.stack(latents).sum(dim=0)`` (default, matches
          ``MultiModalRegressor``)
        - ``"cat"``: ``torch.cat(latents, dim=-1)``
        - ``"none"``: return the tuple of latents as-is.
    """

    def __init__(self, state_keys: List[str], combine: str = "sum"):
        super().__init__()
        self.state_keys = state_keys
        self.combine = combine

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        latents = [ctx.state[k] for k in self.state_keys]

        if self.combine == "sum":
            z = torch.stack(latents).sum(dim=0)
        elif self.combine == "cat":
            z = torch.cat(latents, dim=-1)
        else:
            return BreakpointOutput(
                fn_name=self.forward.__qualname__,
                context=ctx,
                output=tuple(latents),
                trace={"latents": latents},
                valid=True,
            )

        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            context=ctx,
            output=(z,),
            trace={"latents": latents},
            valid=True,
        )

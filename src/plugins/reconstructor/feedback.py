"""Feedback callbacks for DAG-native in-place replacement.

Two classes:

- ``MutatorCallback``: on encoder modules with ``mutate=True``.  In prefill
  phase, passes through the original encoder output.  In mutate phase, reads
  processed values from its own ``_buffer`` (populated by reconstructor via
  ``data_sinks``) and emits them — true in-place replacement.

- ``FeedbackReconstructor``: wraps a reconstructor (e.g. BilinearReconstructor).
  In prefill phase, runs reconstruction and pushes results to downstream
  mutator BPs via ``data_sinks``.  In mutate phase, passes through — does
  NOT overwrite mutator buffers that were populated during prefill.
"""

from __future__ import annotations

import torch.nn as nn

from src.plugins.var import BreakpointContext, BreakpointOutput


class MutatorCallback(nn.Module):
    """Standalone breakpoint callback that applies or passes through based on phase.

    Placed on encoder modules with ``mutate=True``.  Reads from its own
    ``_buffer`` (populated by reconstructor via ``data_sinks``).

    The reconstructor pushes a tuple ``(rec_0, rec_1)`` to each mutator's
    buffer.  ``index`` selects which element this mutator should emit
    (0 for the first encoder's reconstruction, 1 for the second's).

    No ``state_key`` needed — the reconstructor's ``data_sinks`` declaration
    routes data directly into this breakpoint's ``_buffer``.
    """

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        phase = ctx.state.get("_phase", "default")
        if phase == "mutate":
            for key, value in ctx.collected.items():
                # value is (rec_0, rec_1) tuple from reconstructor
                if isinstance(value, tuple):
                    return BreakpointOutput(
                        fn_name=self.forward.__qualname__,
                        context=ctx,
                        output=value[self.index],
                        trace={"source": "feedback", "from": key, "index": self.index},
                    )
                return BreakpointOutput(
                    fn_name=self.forward.__qualname__,
                    context=ctx,
                    output=value,
                    trace={"source": "feedback", "from": key},
                )
        # prefill or default: pass through original encoder output
        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            context=ctx,
            output=ctx.output,
            trace={"source": "encoder"},
        )


class FeedbackReconstructor(nn.Module):
    """Phase-aware reconstructor wrapper.

    In **prefill** phase: runs the wrapped reconstructor, returns its result.
    The DAG's ``run_after()`` pushes ``result.output`` to ``data_sinks``
    (mutator BPs' ``_buffer``).

    In **mutate** phase: passes through.  Returns ``ctx.inputs`` unchanged so
    the DAG pushes original encoder outputs to sinks — this is harmless
    because mutator BPs consume their ``_buffer`` before the reconstructor
    fires on ``encoders.1`` after-hook (mutator_enc1 fires before
    reconstructor in the hook chain).
    """

    def __init__(self, reconstructor: nn.Module):
        super().__init__()
        self.reconstructor = reconstructor

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        phase = ctx.state.get("_phase", "default")
        if phase == "mutate":
            return BreakpointOutput(
                fn_name=self.forward.__qualname__,
                context=ctx,
                output=ctx.inputs,
                trace={"phase": "mutate"},
            )
        return self.reconstructor(ctx)

# True Feedback Hook DAG — Distributing Processing Output Back to Source Hooks

## Problem Statement

The user wants processing breakpoint output (e.g., reconstructor) to be distributed
**back to the source breakpoints** on the encoders, so the source breakpoints themselves
emit the processed values. This matters because the model's `forward()` contains
**non-Module computation** between encoders and head that hooks cannot intercept:

```python
# MultiModalRegressor.forward()
def forward(self, xs):
    zs = [self.encoders[i](x) for i, x in enumerate(xs)]  # list comprehension
    z = torch.stack(zs).sum(dim=0)                          # non-Module ops
    return self.head(z).squeeze(-1)
```

If `src_enc0` (after-hook on `encoders.0`) emits the **reconstructed** `z0` instead
of the original, then `zs[0]` gets the processed value, `torch.stack(zs).sum(dim=0)`
operates on processed latents, and the entire downstream computation — including
non-Module operations — uses the corrected values.

This is **true in-place replacement** at the encoder output level, as opposed to
"faking" it by mutating head input (which misses `torch.stack`, `sum`, and any
other non-Module computation between encoders and head).

---

## The Fundamental Constraint

### Temporal ordering

```
Time ───────────────────────────────────────────────────────►

encoders.0(x0)          encoders.1(x1)           head(z)
│                        │                        │
├─ src_enc0 fires        ├─ src_enc1 fires        ├─ head runs
│  emits z0               │  emits z1              │
│  zs[0] = z0 ✓          │  zs[1] = z1 ✓          │
│                         │                        │
│  At this point:         │  At this point:         │
│  z1 doesn't exist yet   │  Both z0 and z1         │
│  → reconstructor        │  exist                  │
│    can't run            │  → reconstructor CAN    │
│                         │    run here              │
│                         │  → BUT zs[0] is already │
│                         │    fixed in the list     │
```

**The reconstructor needs both `z0` and `z1`, but `src_enc0` must emit its value
before `encoders.1` runs.** This is a fundamental temporal constraint — no hook
mechanism can retroactively change what `encoders.0` returned after `encoders.1`
hasn't even executed yet.

### Why hooks alone can't solve it

PyTorch forward hooks fire synchronously during module execution. The hook on
`encoders.0` fires when `encoders.0(x0)` is called. At that moment, `encoders.1`
hasn't run, `z1` doesn't exist, and the reconstructor cannot produce a result.

Any solution must either:
1. Accept **temporal decoupling** (use reconstruction from a previous forward pass), or
2. **Restructure the forward pass** (two passes: collect then replay), or
3. **Modify the model** (add a shared buffer that encoders write to and head reads from).

---

## Architecture Brainstorm

### Approach A: Delayed Feedback (Zero Code Changes to Existing Files)

**Concept**: Source BPs emit cached reconstruction from the **previous** training step.

```
Forward pass N:
  src_enc0: checks controller.state["z0_recon"]
    → pass 0: empty → emits original z0 (pass through)
    → pass ≥1: contains reconstruction from pass N-1 → emits z0_recon ✓
  src_enc1: same logic
  reconstructor (after encoders.1, or before head):
    collects z0, z1, reconstructs
    stores z0_recon, z1_recon in controller.state  ← for pass N+1

Forward pass N+1:
  src_enc0: finds z0_recon from pass N → emits reconstructed value ✓
  src_enc1: finds z1_recon from pass N → emits reconstructed value ✓
```

**New files needed** (no edits to existing code):

| File | Purpose |
|---|---|
| `src/plugins/reconstructor/feedback.py` | `DeferredSourceCallback` — source BP callback with state-cache check |
| `configs/plugins/hook_dag_feedback.yaml` | Plugin config with `DeferredSourceCallback` on source BPs |

**`DeferredSourceCallback` sketch**:

```python
class DeferredSourceCallback(nn.Module):
    """Source BP that emits cached reconstruction when available.

    On the first forward pass (cold cache), passes through the original
    encoder output.  After the reconstructor has populated
    controller.state, subsequent passes emit the reconstructed value.
    """

    def __init__(self, state_key: str):
        super().__init__()
        self.state_key = state_key

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        recon = ctx.state.get(f"{self.state_key}_recon")
        if recon is not None:
            return BreakpointOutput(
                fn_name=self.forward.__qualname__,
                output=recon,
                trace={"source": "cache", "state_key": self.state_key},
            )
        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            output=ctx.output,
            trace={"source": "encoder", "state_key": self.state_key},
        )
```

**Reconstructor callback** (runs after both encoders, writes to state):

```python
class FeedbackCollector(nn.Module):
    """Collects encoder outputs, reconstructs, caches for next pass."""

    def __init__(self, reconstructor: nn.Module, state_keys: List[str]):
        super().__init__()
        self.reconstructor = reconstructor
        self.state_keys = state_keys

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        result = self.reconstructor(ctx)
        recs = result.output  # (rec_0, rec_1)
        for i, key in enumerate(self.state_keys):
            ctx.state[f"{key}_recon"] = recs[i]
        return result
```

**Key properties**:

| Dimension | Assessment |
|---|---|
| Model-agnostic? | Yes — works with any `nn.Module` tree, any `forward()` |
| Code edits needed? | **None** — new files only |
| Extra compute? | None (same forward pass count) |
| Staleness | 1 training step |
| Hook DAG cycles? | No — feedback via `controller.state`, not DAG edges |
| Training dynamic impact | Mild — equivalent to momentum encoder staleness in BYOL/JEPA |

**Staleness analysis**: The one-step delay means the reconstruction used at step N
was computed from encoder outputs at step N-1. For stochastic gradient training
with small learning rates, the encoder outputs change slowly, making the staleness
acceptable. This is the same principle used in momentum encoders (BYOL, JEPA) where
the target encoder lags behind the online encoder.

---

### Approach B: Two-Phase Forward (Exact Replacement)

**Concept**: The training loop runs two forward passes per batch — one to collect
encoder outputs, one to replay with reconstructed values.

```python
def model_step(self, batch):
    xs, y = batch

    # Phase 1: Collect encoder outputs (no_grad)
    self.controller.state["phase"] = "collect"
    with torch.no_grad():
        _ = self.forward(xs)  # source BPs write z0, z1 to state

    # Reconstruct between phases
    z0, z1 = self.controller.state["z0"], self.controller.state["z1"]
    z0_rec, z1_rec = reconstructor(z0, z1, signal)
    self.controller.state.update({"z0_recon": z0_rec, "z1_recon": z1_rec})

    # Phase 2: Replay with reconstructed values
    self.controller.state["phase"] = "replay"
    output = self.forward(xs)  # source BPs emit z0_recon, z1_recon
    # zs[0] = z0_recon ✓, zs[1] = z1_recon ✓, all non-Module ops use processed values
```

**Source BP callback**:

```python
class PhasedSourceCallback(nn.Module):
    """Source BP that switches behavior based on controller.state["phase"]."""

    def __init__(self, state_key: str):
        super().__init__()
        self.state_key = state_key

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        if ctx.state.get("phase") == "collect":
            ctx.state[self.state_key] = ctx.output   # store for reconstruction
            return BreakpointOutput(output=ctx.output)
        elif ctx.state.get("phase") == "replay":
            recon = ctx.state[f"{self.state_key}_recon"]
            return BreakpointOutput(output=recon)
        return BreakpointOutput(output=ctx.output)   # default: pass through
```

**Key properties**:

| Dimension | Assessment |
|---|---|
| Model-agnostic? | Yes |
| Code edits needed? | Training loop change (two forward calls per step) |
| Extra compute? | **2× encoder forward** per batch |
| Staleness | None — exact replacement |
| Hook DAG cycles? | No |
| Correctness | Highest — same-batch reconstruction |

---

### Approach C: Model Modification (Shared Latent Buffer)

**Concept**: Add a `LatentBuffer` module between encoders and head. Encoders write
to the buffer, the reconstructor can modify the buffer, and the head reads from it.

```python
class LatentBuffer(nn.Module):
    """Shared mutable buffer between encoders and head."""
    def __init__(self, n_modals: int):
        super().__init__()
        self._data: List[Optional[torch.Tensor]] = [None] * n_modals

    def write(self, idx: int, value: torch.Tensor):
        self._data[idx] = value

    def read(self, idx: int) -> torch.Tensor:
        return self._data[idx]

class MultiModalRegressorWithBuffer(nn.Module):
    def __init__(self, ...):
        ...
        self.buffer = LatentBuffer(n_modals=2)

    def forward(self, xs):
        for i, x in enumerate(xs):
            z = self.encoders[i](x)
            self.buffer.write(i, z)  # encoder writes to buffer

        # Breakpoint on encoders.1 can modify buffer via controller.state ref

        zs = [self.buffer.read(i) for i in range(len(xs))]
        z = torch.stack(zs).sum(dim=0)
        return self.head(z).squeeze(-1)
```

**Key properties**:

| Dimension | Assessment |
|---|---|
| Model-agnostic? | **No** — requires model-specific changes |
| Code edits needed? | Model class must be changed |
| Extra compute? | None |
| Staleness | None |
| Correctness | Highest |

---

### Approach D: Forward Patch (Monkey-Patch `forward`)

**Concept**: Dynamically replace the model's `forward` method at runtime to insert
a buffer between encoder outputs and head input.

```python
def install_feedback_forward(model, controller):
    original_forward = model.forward

    def patched_forward(self, xs):
        zs = []
        for i, x in enumerate(xs):
            z = self.encoders[i](x)
            # Source BP writes to state (via hook)
            # Check for reconstructed value
            recon = controller.state.get(f"z{i}_recon")
            zs.append(recon if recon is not None else z)
        z = torch.stack(zs).sum(dim=0)
        return self.head(z).squeeze(-1)

    model.forward = patched_forward.__get__(model, type(model))
```

**Key properties**:

| Dimension | Assessment |
|---|---|
| Model-agnostic? | No — assumes list-comprehension pattern |
| Fragile? | Yes — breaks if model structure changes |
| Extra compute? | None |

Not recommended except for rapid prototyping.

---

## Recommendation

### For toy/manifold dataset: Approach B (Two-Phase Forward)

The encoders are small MLPs (a few linear layers). Doubling their forward cost is
negligible. The two-phase approach gives **exact, same-batch replacement** with
clean semantics and no staleness concerns.

Implementation outline:
1. New file: `src/plugins/reconstructor/feedback.py` with `PhasedSourceCallback`
2. Modified training loop in a new LightningModule or a thin wrapper
3. Source BPs use `PhasedSourceCallback`
4. Reconstructor writes to `controller.state` between phases

### For large models (JEPA): Approach A (Delayed Feedback)

When encoder forward is expensive (ViT on 3D MRI), doubling compute is unacceptable.
Delayed feedback gives the same architecture at 1× cost, with one-step staleness
that is well-studied and accepted in self-supervised learning (momentum encoders).

### Decision matrix

| Criterion | A: Delayed | B: Two-Phase | C: Model Mod | D: Monkey-Patch |
|---|---|---|---|---|
| Model-agnostic | Yes | Yes | No | No |
| No code edits | Yes | Training loop only | No | No |
| Compute cost | 1× | 2× | 1× | 1× |
| Staleness | 1 step | None | None | None |
| Production-ready | Yes | Small models only | Yes | No |
| Implementation effort | Low | Low | Medium | Low |

---

## Implementation Plan (Approach A — Delayed Feedback)

### New files only

```
src/plugins/reconstructor/feedback.py          # DeferredSourceCallback, FeedbackCollector
configs/plugins/hook_dag_feedback.yaml          # Breakpoint config with deferred feedback
```

### `src/plugins/reconstructor/feedback.py`

```python
"""Feedback callbacks for true in-place replacement via deferred state.

These callbacks enable a "round-trip" pattern where processing breakpoint
output is distributed back to source breakpoints via controller.state,
creating true in-place replacement at the encoder output level.

Two classes:

- DeferredSourceCallback: source BP that checks controller.state for a
  cached reconstruction.  On cold cache, passes through the original encoder
  output.  The reconstructor populates the cache for the next forward pass.

- FeedbackCollector: wraps a reconstructor.  After reconstruction, writes
  the result to controller.state so DeferredSourceCallback can consume it.

Key invariant: _buffer is NOT reset between forward passes (reset() is only
called by BreakpointController.clear(), not during normal forward).  So the
feedback loop is:
  pass N: src → original, recon → state[key] = rec
  pass N+1: src → state[key] (from pass N), recon → state[key] = rec (updated)
"""

from __future__ import annotations

from typing import List

import torch.nn as nn

from src.plugins.var import BreakpointContext, BreakpointOutput


class DeferredSourceCallback(nn.Module):
    """Source BP callback with state-cached reconstruction support.

    On the first forward pass (cold cache), passes through the original
    encoder output unchanged.  After a FeedbackCollector has populated
    ``controller.state[f"{state_key}_recon"]``, subsequent forward passes
    emit the reconstructed value instead.

    This enables true in-place replacement: the source breakpoint itself
    emits the processed value, so ALL downstream computation — including
    non-Module operations in the model's forward() like torch.stack, sum,
    list comprehensions — operates on the processed latent.

    Parameters
    ----------
    state_key:
        Key prefix in controller.state.  The callback reads from
        ``{state_key}_recon`` (set by FeedbackCollector) and falls back
        to the original encoder output if not found.
    """

    def __init__(self, state_key: str):
        super().__init__()
        self.state_key = state_key

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        recon = ctx.state.get(f"{self.state_key}_recon")
        if recon is not None:
            return BreakpointOutput(
                fn_name=self.forward.__qualname__,
                output=recon,
                trace={"source": "cache", "state_key": self.state_key},
            )
        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            output=ctx.output,
            trace={"source": "encoder", "state_key": self.state_key},
        )


class FeedbackCollector(nn.Module):
    """Collector callback: reconstructs and caches result for next pass.

    Wraps a reconstructor (e.g., BilinearReconstructor).  After the
    reconstructor runs, writes each reconstructed latent to
    ``controller.state[f"{key}_recon"]`` so that DeferredSourceCallback
    instances on source breakpoints can consume them on the next forward pass.

    The reconstructor's original BreakpointOutput is returned unchanged,
    so downstream breakpoints (e.g., uncertainty) can read the trace
    via DAG data_sources as usual.

    Parameters
    ----------
    reconstructor:
        nn.Module whose forward(ctx) → BreakpointOutput with
        .output = (rec_0, rec_1, ...).
    state_keys:
        Ordered list of state key prefixes, e.g. ["z0", "z1"].
        Results are stored at "{key}_recon".
    """

    def __init__(self, reconstructor: nn.Module, state_keys: List[str]):
        super().__init__()
        self.reconstructor = reconstructor
        self.state_keys = state_keys

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        result = self.reconstructor(ctx)
        recs = result.output
        for i, key in enumerate(self.state_keys):
            ctx.state[f"{key}_recon"] = recs[i]
        return result
```

### `configs/plugins/hook_dag_feedback.yaml`

```yaml
# True-feedback hook DAG: source BPs emit cached reconstruction from
# previous forward pass, achieving true in-place replacement at the
# encoder output level.
#
# Data flow (per forward pass):
#   src_enc0 → reconstructor (via DAG _buffer, current-pass data)
#   src_enc1 → reconstructor (via DAG _buffer, current-pass data)
#   reconstructor → controller.state (feedback, for NEXT pass)
#   controller.state → src_enc0 (next pass: emits cached reconstruction)
#   controller.state → src_enc1 (next pass: emits cached reconstruction)
#
# No DAG cycles — the feedback edge goes through controller.state,
# not through data_sources/data_sinks.

target: MultiModalRegressor
model_checkpoint: data/checkpoints/checkpoint_multimodal_manifold.pth
plugins_checkpoint: null

breakpoints:

  # ── Source BPs: emit cached reconstruction when available ──

  - layer_name: encoders.0
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_enc0
      callback:
        _target_: src.plugins.reconstructor.feedback.DeferredSourceCallback
        state_key: z0
    pos: after

  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_enc1
      callback:
        _target_: src.plugins.reconstructor.feedback.DeferredSourceCallback
        state_key: z1
    pos: after

  # ── Reconstructor: runs after both encoders, caches for next pass ──
  #
  # data_sources: [src_enc0.0, src_enc1.0] — collects via DAG.
  # callback: FeedbackCollector — wraps BilinearReconstructor,
  #           writes reconstructed latents to controller.state.

  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: reconstructor
      data_sources: [src_enc0.0, src_enc1.0]
      pre_fn:
        _target_: src.plugins.hook_dag.ToListCollectedFn
      callback:
        _target_: src.plugins.reconstructor.feedback.FeedbackCollector
        reconstructor:
          _target_: src.plugins.reconstructor.linear.BilinearReconstructor
          d_1: 16
          d_2: 16
          hidden_dims: [32, 16, 16, 16]
          activation: silu
          norm: group
          dropout: 0.3
          order: adn
          concat: false
          dist:
            _target_: torch.nn.MSELoss
            reduction: none
        state_keys: [z0, z1]
      mutate: false
      valid: true
    pos: after

  # ── Uncertainty: BayesCap on final head output ──

  - layer_name: head
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: uncertainty
      data_sources: [input]
      callback:
        _target_: src.plugins.head.bayescap.BayesCap1D
        input_dim: 1
        output_dim: 1
        hidden_dims: [8, 8, 4]
        per_dim_uncertainty: False
        dropout: 0.3
        norm: batch
        activation: silu
      mutate: false
      valid: true
    pos: after
```

### Execution trace (first two forward passes)

```
=== Pass 0 (cold cache) ===

signal = (0, 1)  # T1w masked

1. encoders.0(x0) → z0_orig [B,16]
   └─ src_enc0 (after):
        DeferredSourceCallback: state["z0_recon"] is None → emit z0_orig
        → DAG pushes z0_orig to reconstructor._buffer["src_enc0.0"]

2. encoders.1(x1) → z1 [B,16]
   └─ src_enc1 (after):
        DeferredSourceCallback: state["z1_recon"] is None → emit z1
        → DAG pushes z1 to reconstructor._buffer["src_enc1.0"]

   └─ reconstructor (after):
        FeedbackCollector:
          BilinearReconstructor(signal=(0,1)):
            rec_0 = ln21(z1)   # reconstruct z0 from z1
            rec_1 = z1
          state["z0_recon"] = rec_0  ← cached for pass 1
          state["z1_recon"] = rec_1  ← cached for pass 1

   zs = [z0_orig, z1]  ← zs[0] is original (cold cache)
   z = z0_orig + z1
   head(z) → prediction

=== Pass 1 (warm cache) ===

signal = (0, 1)  # T1w masked again

1. encoders.0(x0) → z0_orig
   └─ src_enc0:
        DeferredSourceCallback: state["z0_recon"] = rec_0 from pass 0 → emit rec_0 ✓
        → DAG pushes rec_0 to reconstructor._buffer["src_enc0.0"]

2. encoders.1(x1) → z1
   └─ src_enc1:
        DeferredSourceCallback: state["z1_recon"] = rec_1 from pass 0 → emit rec_1 (≈ z1)
        → reconstructor updates cache

   zs = [rec_0, rec_1]  ← zs[0] is RECONSTRUCTED ✓
   z = rec_0 + rec_1    ← torch.stack + sum operate on processed values ✓
   head(z) → prediction
```

### Model-agnostic guarantee

This approach works with **any** model structure because:

1. **No assumption about `forward()`**: Source BPs change what `encoders[i](x)` returns.
   Whatever the model does with that return value — list comprehension, `torch.stack`,
   `torch.cat`, loops, conditionals — it receives the processed latent.

2. **No assumption about Module tree**: The only requirement is that encoders are
   `nn.Module` instances accessible via `named_modules()`. This is true for all
   PyTorch models.

3. **No DAG cycle**: Feedback goes through `controller.state`, not through
   `data_sources`/`data_sinks`. The DAG remains acyclic (src → recon only).

4. **No `reset()` interference**: `Breakpoint.reset()` clears `_buffer` but is only
   called in `BreakpointController.clear()` (hook removal), never during normal
   forward. `controller.state` is never reset between forward passes.

### Cold-start handling

The first forward pass always uses original encoder outputs (no cache available).
For training, this means:
- Batch 0: original encoder outputs (reconstructor warms the cache)
- Batch 1+: reconstructed encoder outputs (cache is warm)

This is acceptable because:
- One batch of non-reconstructed outputs has negligible impact on training
- For validation/testing, the reconstructor is in eval mode and cache is warm
  after the first training step

---

## Implementation Plan (Approach B — Two-Phase Forward)

If exact same-batch replacement is preferred (for toy manifold), the two-phase
approach requires only training-loop changes:

### New files

```
src/models/hook_modules/feedback_module.py       # LightningModule with two-phase model_step
```

### `model_step` sketch

```python
def model_step(self, batch, **kwargs):
    xs, y, _, _ = batch
    signal = self._random_mask()

    # Phase 1: collect encoder outputs (no_grad)
    self.controller.state["phase"] = "collect"
    with torch.no_grad():
        _ = self.forward(torch.split(xs, 1, dim=1))

    # Phase 2: reconstruct between passes
    z0 = self.controller.state["z0"]
    z1 = self.controller.state["z1"]
    # ... run reconstructor on (z0, z1, signal) ...
    self.controller.state["z0_recon"] = z0_rec
    self.controller.state["z1_recon"] = z1_rec
    self.controller.state["phase"] = "replay"

    # Phase 3: replay with reconstructed values
    logits = self.forward(torch.split(xs, 1, dim=1)).unsqueeze(1)

    # Loss computation as usual...
```

**Key trade-off**: 2× encoder compute per batch. For toy manifold (small MLPs),
this is negligible. For JEPA ViT, this doubles a forward pass that may already
take seconds.

---

## Optimizing the Processor Hook Module

In Approach A (Delayed Feedback), the reconstructor runs as an after-hook on
`encoders.1` — it sits **in the critical path** of every forward pass:

```
encoders.0 → src_enc0 → encoders.1 → src_enc1 → [RECONSTRUCTOR] → head → uncertainty
                                                  ↑
                                                  adds latency to every pass
```

The reconstructor's output is only consumed on the **next** pass (by source BPs via
`controller.state`). This raises the question: can we avoid paying the reconstructor
cost on every forward pass?

### Strategy 1: Signal-Conditional Skip

The simplest optimization. When `signal == (1,1)` (both modalities present, no
masking), the reconstructor has nothing to reconstruct — all cross-modal mappings
(`ln12`, `ln21`) are skipped. But the **deviation heads** (`dev1`, `dev2`) still
run, computing `dist(rec, src)` for loss even though no reconstruction happened.

Add an early-exit guard:

```python
class SkippableBilinearReconstructor(BilinearReconstructor):
    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        (p1, p2) = ctx.bp_kwargs
        if p1 == 1 and p2 == 1:
            # Both modalities present — skip all compute
            return BreakpointOutput(
                fn_name=self.forward.__qualname__,
                context=ctx,
                output=ctx.inputs,
                trace={
                    "signal": (1, 1),
                    "input": ctx.inputs,
                    "reconstructed": ctx.inputs,
                    "distance": (torch.zeros(1), torch.zeros(1)),
                    "dev": (torch.zeros(1), torch.zeros(1)),
                },
            )
        return super().forward(ctx)
```

**Impact**: With `mask_rate=0.3`, the probability of `(1,1)` is `1 - mask_rate = 0.7`.
So **70% of forward passes skip the reconstructor entirely**. Source BPs still
emit from cache (which is at most 1 step stale from the last masked pass).

**Trade-off**: the reconstructor trains on fewer examples (only masked passes),
but those are exactly the examples where reconstruction matters.

### Strategy 2: Strided Cache Update

Run the reconstructor every **K** steps instead of every step. The cache is
K steps stale instead of 1, but the reconstructor cost is amortized by factor K.

```python
class StridedFeedbackCollector(FeedbackCollector):
    def __init__(self, reconstructor, state_keys, stride: int = 4):
        super().__init__(reconstructor, state_keys)
        self.stride = stride
        self._step_counter = 0

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        self._step_counter += 1
        if self._step_counter % self.stride == 0:
            return super().forward(ctx)  # run reconstructor, update cache
        # Skip: return a dummy trace, cache unchanged
        return BreakpointOutput(
            fn_name=self.forward.__qualname__,
            output=ctx.inputs,
            trace={},
        )
```

**Impact with stride=4**: 4× reduction in reconstructor forward/backward cost.
Cache staleness goes from 1→4 steps.

**When this works well**: the data distribution changes slowly (small learning
rate, large datasets). The encoder outputs at step N and step N+4 are similar,
so a 4-step-stale reconstruction is nearly as good as a 1-step-stale one.

### Strategy 3: Async Decoupling (Background Reconstructor)

Take the reconstructor **entirely off the critical path**. It runs in a
background thread/process, pulling from a queue of encoder outputs and pushing
reconstructions to a shared cache.

```
Main thread (every step):               Background thread (every K ms):
  src_enc0 → emit from cache              queue.pop() → (z0, z1, signal)
  src_enc1 → emit from cache              reconstructor(z0, z1, signal)
  head → prediction → loss → backward     cache.update(z0_rec, z1_rec)
  queue.push(z0, z1, signal)              (no_grad or separate optimizer)
```

```python
class AsyncFeedbackCollector(nn.Module):
    """Pushes encoder outputs to a background queue instead of running inline.

    A separate worker thread runs the reconstructor and updates the shared
    cache (controller.state). The main forward pass never waits for the
    reconstructor.
    """

    def __init__(self, reconstructor, state_keys, max_queue_size=16):
        super().__init__()
        self.reconstructor = reconstructor
        self.state_keys = state_keys
        self._queue = collections.deque(maxlen=max_queue_size)
        self._worker = None

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        # Non-blocking: push to queue, return immediately
        zs = list(ctx.inputs)
        signal = ctx.bp_kwargs
        try:
            self._queue.append((zs, signal, ctx.state))
        except Exception:
            pass  # queue full, drop oldest
        return BreakpointOutput(output=ctx.inputs, trace={})

    def _worker_loop(self):
        while True:
            zs, signal, state = self._queue.popleft()  # blocks
            # Run reconstructor
            recs = self._reconstruct(zs, signal)
            # Update shared cache (controller.state is thread-safe in CPython)
            for i, key in enumerate(self.state_keys):
                state[f"{key}_recon"] = recs[i]
```

**Caveats**:
- `controller.state` is a plain dict — GIL makes this safe in CPython but not
  in multiprocessing contexts (DDP). For DDP, use `torch.multiprocessing` with
  shared memory or a dedicated CUDA stream.
- The reconstructor needs its own optimizer if it's trained (not just inference).
- Adds complexity (thread management, queue sizing, graceful shutdown).

**When to use**: large models (JEPA ViT) where reconstructor cost rivals encoder
cost. The async approach decouples the two, giving near-zero overhead on the
main forward pass.

### Strategy 4: Epoch-Phased Execution

Train the reconstructor in dedicated epochs, freeze it during others:

```
Epochs 0..N_warmup:
  - Reconstructor: OFF (source BPs pass through originals)
  - Train: head only (baseline regression)
  - Purpose: encoders produce meaningful latents first

Epochs N_warmup..N_warmup+N_recon:
  - Reconstructor: ON (runs every step, updates cache)
  - Train: head + reconstructor
  - Purpose: reconstructor learns cross-modal mappings

Epochs N_warmup+N_recon..:
  - Reconstructor: ON but frozen (eval mode)
  - Source BPs emit from frozen reconstructor cache
  - Train: uncertainty head only
  - Purpose: uncertainty calibration on stable reconstructions
```

**Impact**: While the reconstructor is frozen, no backward pass through it —
forward only, which is cheaper and can run in `torch.no_grad()`.

### Strategy comparison

| Strategy | Reconstructor compute | Cache staleness | Implementation | Best for |
|---|---|---|---|---|
| Baseline (every step) | 1× per step | 1 step | Trivial | Default |
| Signal-conditional skip | 0.3× per step (mask_rate=0.3) | 1 step (when masked) | `SkippableBilinearReconstructor` | Any model |
| Strided (K=4) | 0.25× per step | K steps | `StridedFeedbackCollector` | Large datasets, small LR |
| Async decoupling | ~0× (background) | Variable | Queue + worker thread | JEPA ViT |
| Epoch-phased | 0× when frozen | Frozen cache | Training schedule | Production pipelines |

### Recommended pipeline

For the toy manifold dataset:

```
1. Start with Signal-Conditional Skip (Strategy 1)
   - 3 lines of code change to BilinearReconstructor
   - 70% reduction in reconstructor compute at mask_rate=0.3

2. If more speed needed: add Strided update (Strategy 2)
   - Stride=4: total reconstructor cost drops to 0.075×
   - Cache is 1-4 steps stale, acceptable for small-LR training
```

For JEPA brain age (large model):

```
1. Start with Signal-Conditional Skip (Strategy 1) — baseline optimization
2. Evaluate Async Decoupling (Strategy 3) if reconstructor is a bottleneck
3. Use Epoch-Phased (Strategy 4) for final production: train reconstructor
   once, freeze it, run uncertainty calibration on stable cache
```

### Why the reconstructor "breaks the pass"

The reconstructor breaks the forward pass in two senses:

1. **Latency**: it sits between the last encoder and the head, adding compute
   to the critical path. Every millisecond spent in reconstructor is a
   millisecond the head and loss computation are delayed.

2. **Gradient flow**: the reconstructor is trainable (unlike the frozen
   encoders). Its backward pass competes for GPU memory and compute with the
   head and uncertainty modules. In a peak-memory regime, the reconstructor's
   activations and gradients may force a smaller batch size.

The optimization strategies above address both concerns by reducing how often
the reconstructor runs (Strategies 1, 2, 4) or by moving it off the critical
path entirely (Strategy 3).

---

## Next Steps

1. Choose between Approach A (delayed feedback, 1× compute) and Approach B
   (two-phase, exact replacement)
2. Choose an optimization strategy for the reconstructor (start with Strategy 1)
3. For Approach A: create `src/plugins/reconstructor/feedback.py` and
   `configs/plugins/hook_dag_feedback.yaml`
4. For Approach B: create `src/models/hook_modules/feedback_module.py`
5. Neither approach requires any edits to existing files — `hook_dag.py`,
   `toy.py`, `var.py`, and model components remain unchanged

The author's recommendation: **Approach A + Strategy 1 for production (JEPA),
Approach B for prototyping (toy manifold)**.

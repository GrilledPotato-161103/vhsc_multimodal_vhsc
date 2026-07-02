# Backprop-Preserving In-Place Hook DAG

## Problem

In-place replacement via delayed feedback detaches the reconstructor from the
computational graph. The reconstructor receives zero gradient from the main task
loss because cached values are from a previous forward pass. The two-phase
(prefill → mutate) approach preserves gradient flow by keeping all values in a
single computational graph across both forward passes.

## Architecture

```
            ┌─ Prefill Phase ───────────────────────────┐
            │  encoders.0 → src_enc0 → recon._buffer     │
            │  encoders.1 → src_enc1 → recon._buffer     │
            │  reconstructor runs, pushes (rec_0, rec_1) │
            │     → mutator_enc0._buffer                 │
            │     → mutator_enc1._buffer                 │
            │  Mutator BPs pass through (phase=prefill)  │
            └────────────────────────────────────────────┘
                               │
            ┌─ Mutate Phase ────────────────────────────┐
            │  mutator_enc0 reads _buffer → emits rec_0  │
            │  mutator_enc1 reads _buffer → emits rec_1  │
            │  zs = [rec_0, rec_1] → sum → head → loss  │
            │  reconstructor passes through (no overwrite)│
            └────────────────────────────────────────────┘
                               │
            ∂loss → head → rec_0 → reconstructor params ✓
                       → rec_1 → z1 (prefill) → encoders.1 ✓
```

**Key insight**: `rec_0` and `rec_1` are pushed to mutator buffers during
prefill WITHOUT `.detach()`. In the mutate phase, mutator BPs read these
still-attached tensors and emit them. Backpropagation traces through the
mutate phase tensors back to the prefill-phase reconstructor and encoders.

## Data Flow (no `controller.state` for data)

All data flows through the native `_buffer` mechanism. `controller.state`
only tracks `_phase` (`"prefill"`, `"mutate"`, `"default"`).

| Breakpoint | Pushes to | Reads from |
|---|---|---|
| `src_enc0` | `reconstructor._buffer["src_enc0.0"]` | encoder hook output |
| `src_enc1` | `reconstructor._buffer["src_enc1.0"]` | encoder hook output |
| `reconstructor` | `mutator_enc0._buffer["reconstructor.0"]`, `mutator_enc1._buffer["reconstructor.0"]` | `_buffer["src_enc0.0"]`, `_buffer["src_enc1.0"]` |
| `mutator_enc0` | (mutate return → module output, index 0) | `_buffer["reconstructor.0"]` |
| `mutator_enc1` | (mutate return → module output, index 1) | `_buffer["reconstructor.0"]` |

---

## Code Changes

### 1. `src/plugins/hook_dag.py` — three additive changes

#### 1a. `Breakpoint.__init__` — new `data_sinks` parameter

```python
def __init__(
    self,
    name: str,
    callback: Optional[...] = None,
    mutate: bool = False,
    valid: bool = False,
    kwargs: dict = dict(),
    data_sources: List[str] | None = None,
    data_sinks: List[str] | None = None,          # NEW
    pre_fn: Optional[...] = None,
    post_fn: Optional[...] = None,
):
    ...
    self.data_sinks: List[Breakpoint] = []         # resolved by wire()
    self._declared_sinks: List[str] = list(data_sinks) if data_sinks else []  # NEW
```

A breakpoint can now declare `data_sinks` in config to establish DAG edges
in the forward direction (producer → consumer). Combined with `data_sources`
(consumer → producer), edges can be declared bidirectionally.

#### 1b. `BreakpointController.wire()` — resolve explicit `data_sinks`

After resolving `data_sources → data_sinks`, the wire method now also walks
each breakpoint's `_declared_sinks` and resolves them to actual Breakpoint
objects, populating both `bp.data_sinks` (forward edge) and
`downstream.data_sources` (reverse edge for traceability). Duplicate
prevention ensures clean wiring.

#### 1c. `BreakpointController.phase()` — context manager

```python
@contextmanager
def phase(self, name: str):
    """Set inference phase: "prefill", "mutate", or "default"."""
    old = self.state.get("_phase", "default")
    self.state["_phase"] = name
    try:
        yield
    finally:
        self.state["_phase"] = old
```

### 2. `src/plugins/reconstructor/feedback.py` — new file

#### `MutatorCallback(nn.Module)`

```python
class MutatorCallback(nn.Module):
    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        phase = ctx.state.get("_phase", "default")
        if phase == "mutate":
            for key, value in ctx.collected.items():
                if isinstance(value, tuple):
                    return BreakpointOutput(
                        output=value[self.index],  # pick rec_0 or rec_1
                        trace={"source": "feedback", "from": key, "index": self.index},
                    )
                return BreakpointOutput(
                    output=value,
                    trace={"source": "feedback", "from": key},
                )
        # prefill or default: pass through
        return BreakpointOutput(
            output=ctx.output,
            trace={"source": "encoder"},
        )
```

Placed on encoder modules with `mutate=true`. In prefill, passes through the
original encoder output. In mutate, reads from `ctx.collected` (its own
`_buffer`, populated by the reconstructor during prefill) and emits the
processed value. The `index` parameter selects which element of the
reconstructor's output tuple `(rec_0, rec_1)` this mutator should emit.

#### `FeedbackReconstructor(nn.Module)`

```python
class FeedbackReconstructor(nn.Module):
    def __init__(self, reconstructor: nn.Module):
        super().__init__()
        self.reconstructor = reconstructor

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        phase = ctx.state.get("_phase", "default")
        if phase == "mutate":
            # Pass through — don't overwrite mutator buffers
            return BreakpointOutput(
                output=ctx.inputs, trace={"phase": "mutate"},
            )
        # Prefill: run the wrapped reconstructor
        return self.reconstructor(ctx)
```

Wraps any reconstructor (e.g., `BilinearReconstructor`). Phase-aware behavior
ensures reconstruction output only flows during prefill.

### 3. `src/models/hook_modules/feedback_module.py` — new file

#### `FeedbackInjectModule(LightningModule)` — two-phase training

The key difference from the existing single-pass `ModelInjectModule` is the
`model_step` method which runs the model forward **twice**:

```python
def model_step(self, batch, **kwargs):
    xs, y, _, _ = batch
    signal = kwargs.get("bp_signal", self._random_mask())

    recon_bp = Breakpoint.get_by_name(self.hparams.recon_bp)
    recon_bp.kwargs = tuple(signal)

    # Phase 1: Prefill — collect + reconstruct
    with self.controller.phase("prefill"):
        _ = self.forward(torch.split(xs, 1, dim=1))

    # Phase 2: Mutate — emit processed values
    with self.controller.phase("mutate"):
        logits = self.forward(torch.split(xs, 1, dim=1)).unsqueeze(1)

    loss = self.criterion(logits, y)
    recon_loss, recon_unc_loss = self._compute_recon_loss(recon_bp.trace)
    unc = self._compute_unc_loss(logits, y)
    return loss, logits, y, {"recon_loss": recon_loss, "unc_loss": recon_unc_loss, "trace": recon_bp.trace}, unc
```

The rest of the module mirrors `ModelInjectModule` (training_step,
validation_step, configure_optimizers with breakpoint callback parameters).

---

## Config: `configs/plugins/hook_dag_feedback.yaml`

```yaml
target: MultiModalRegressor
model_checkpoint: data/checkpoints/checkpoint_multimodal_manifold.pth

breakpoints:

  # Source BPs: capture encoder outputs, push to reconstructor
  - layer_name: encoders.0
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_enc0
      data_sinks: [reconstructor.0]
    pos: after

  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_enc1
      data_sinks: [reconstructor.0]
    pos: after

  # Mutator BPs: mutate=true with index
  - layer_name: encoders.0
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: mutator_enc0
      callback:
        _target_: src.plugins.reconstructor.feedback.MutatorCallback
        index: 0
      mutate: true
    pos: after

  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: mutator_enc1
      callback:
        _target_: src.plugins.reconstructor.feedback.MutatorCallback
        index: 1
      mutate: true
    pos: after

  # Reconstructor: data_sources + data_sinks
  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: reconstructor
      data_sources: [src_enc0.0, src_enc1.0]
      data_sinks: [mutator_enc0.0, mutator_enc1.0]
      pre_fn:
        _target_: src.plugins.hook_dag.ToListCollectedFn
      callback:
        _target_: src.plugins.reconstructor.feedback.FeedbackReconstructor
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
      mutate: false
      valid: true
    pos: after

  # Uncertainty: BayesCap on head output
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
        per_dim_uncertainty: false
        dropout: 0.3
        norm: batch
        activation: silu
      mutate: false
      valid: true
    pos: after
```

### DAG Edge Summary

```
src_enc0 ──(data_sinks)──► reconstructor
src_enc1 ──(data_sinks)──► reconstructor
reconstructor ──(data_sinks)──► mutator_enc0
reconstructor ──(data_sinks)──► mutator_enc1
mutator_enc0/1: terminal (no outgoing edges)
uncertainty ←──(data_sources: input)── head output
```

### Hook Registration Order

On `encoders.0` (after-hooks, in registration order):
1. `src_enc0` (callback=None) — push z0 to reconstructor, pass-through
2. `mutator_enc0` (MutatorCallback, index=0, mutate=true)

On `encoders.1` (after-hooks, in registration order):
1. `src_enc1` (callback=None) — push z1 to reconstructor, pass-through
2. `mutator_enc1` (MutatorCallback, index=1, mutate=true)
3. `reconstructor` (FeedbackReconstructor, after mutator_enc1) — reconstruct
   and push to mutator buffers (prefill), pass-through (mutate)

Mutator on encoders.1 fires BEFORE reconstructor, so in mutate phase it reads
the prefill buffer before reconstructor could overwrite it.

---

## Application: Toy Manifold Model

### Model: `MultiModalRegressor`

```python
# src/models/components/toy.py
class MultiModalRegressor(nn.Module):
    def forward(self, xs: Sequence[Tensor]) -> torch.Tensor:
        zs = [self.encoders[i](x) for i, x in enumerate(xs)]  # list comprehension
        z = torch.stack(zs).sum(dim=0)                         # non-Module op
        return self.head(z).squeeze(-1)
```

The list comprehension captures `zs[i]` as the return value of `encoders[i](x)`.
With `mutate=true` on mutator BPs, the module output IS replaced, so `zs[i]`
gets the reconstructed value — true in-place replacement through ALL non-Module
computation.

### Training Entry Point

```bash
python src/train_hook_dag.py --config-name train_hook_dag_feedback
```

Training config: `configs/train_hook_dag_feedback.yaml` — extends
`train_hook_dag.yaml` with the feedback plugins config and W&B logging to
the `VHSC_Feedback` project.

### Expected Checkpoint

`data/checkpoints/checkpoint_multimodal_manifold.pth` — pretrained
`MultiModalRegressor` checkpoint (the same one used by the existing
single-pass pipeline).

### Gradient Flow (verified)

| Signal | rec_0 | rec_1 | encoders.0 grad | encoders.1 grad | reconstructor grad |
|---|---|---|---|---|---|
| (1,1) | z0 | z1 | yes | yes | no (pass-through) |
| (1,0) | ln21(z1) | z1 | yes (via ln21) | yes (direct) | yes (ln21 params) |
| (0,1) | z0 | ln12(z0) | yes (direct) | no | yes (ln12 params) |

With `signal=(1,0)` and two-phase training, the gradient path is:
```
loss → head → zs=[ln21(z1), z1] → ln21 params (2.89 grad norm) ✓
                                 → z1 (prefill) → encoders.1 params (10.35 grad norm) ✓
```

---

## Application: Neuro-JEPA Pretrained Model

### Model: `MultiModalJEPARegressor`

```python
# src/models/components/jepa.py
class MultiModalJEPARegressor(nn.Module):
    backbone: VisionTransformer      # Shared frozen ViT-base (768-dim, 12 layers)
    encoders: ModuleList             # [ModalExtractor, ModalExtractor] — hook targets
    classifier: MultiModalLateFusion # Cross-attention fusion → brain age

    def forward(self, images: Dict[str, Tensor] | List[Tensor]) -> Tensor:
        feats = []
        for i, img in enumerate(image_list):
            with torch.no_grad():
                f = self.backbone(img)     # [B, N, 768] ViT token grid
            f = self.encoders[i](f)        # Hook target — "encoders.0" / "encoders.1"
            feats.append(f)
        return self.classifier(feats[0], feats[1])  # Cross-attn → [B, num_classes]
```

### Two-Phase JEPA Config

Create `configs/plugins/hook_dag_jepa_feedback.yaml`:

```yaml
target: MultiModalJEPARegressor
model_checkpoint: data/checkpoints/jepa/jepa_pretrained.pth

breakpoints:

  # Source BPs — capture ViT token grids [B, N, 768]
  - layer_name: encoders.0
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_enc0
      data_sinks: [reconstructor.0]
    pos: after

  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_enc1
      data_sinks: [reconstructor.0]
    pos: after

  # Mutator BPs — replace Vit token grids with reconstructed latents
  - layer_name: encoders.0
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: mutator_enc0
      callback:
        _target_: src.plugins.reconstructor.feedback.MutatorCallback
        index: 0
      mutate: true
    pos: after

  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: mutator_enc1
      callback:
        _target_: src.plugins.reconstructor.feedback.MutatorCallback
        index: 1
      mutate: true
    pos: after

  # Reconstructor — mean-pool token grids [B,N,768]→[B,768], then reconstruct
  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: reconstructor
      data_sources: [src_enc0.0, src_enc1.0]
      data_sinks: [mutator_enc0.0, mutator_enc1.0]
      pre_fn:
        _target_: src.plugins.reconstructor.pool.MeanPoolCollectedFn
      callback:
        _target_: src.plugins.reconstructor.feedback.FeedbackReconstructor
        reconstructor:
          _target_: src.plugins.reconstructor.linear.BilinearReconstructor
          d_1: 768
          d_2: 768
          hidden_dims: [512, 256, 128]
          activation: silu
          norm: layer
          dropout: 0.3
          order: adn
          concat: false
          dist:
            _target_: torch.nn.MSELoss
            reduction: none
      mutate: false
      valid: true
    pos: after

  # Uncertainty on classifier output
  - layer_name: classifier
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: uncertainty
      data_sources: [input]
      callback:
        _target_: src.plugins.head.bayescap.BayesCap1D
        input_dim: 1
        output_dim: 1
        hidden_dims: [32, 16, 8]
        per_dim_uncertainty: false
        dropout: 0.3
        norm: batch
        activation: silu
      mutate: false
      valid: true
    pos: after
```

**Key difference from manifold model**: The reconstructor is placed **after**
`encoders.1` (not before `classifier`) because:
1. Mutator BPs need to replace encoder outputs BEFORE the classifier runs
2. Mutator on `encoders.1` fires before reconstructor (correct ordering)
3. The classifier receives the reconstructed latents (via mutator replacement)

**`MeanPoolCollectedFn`**: Unlike the manifold model's `ToListCollectedFn`,
this pre_fn mean-pools ViT token grids `[B, N, 768]` → `[B, 768]` before
passing to the BilinearReconstructor. The reconstructor maps between modality
spaces in the 768-dim pooled latent space.

### Two-Phase JEPA Training Module

```python
# src/models/hook_modules/jepa_feedback.py
class JEPAFeedbackModule(FeedbackInjectModule):
    """Extends FeedbackInjectModule with JEPA-specific forward pass."""

    def forward(self, images: Dict[str, Tensor] | List[Tensor]) -> Tensor:
        return self.net(images)

    def model_step(self, batch, **kwargs):
        images, y = batch
        y = y.view(-1, 1).float()
        signal = kwargs.get("bp_signal", self._random_mask())

        recon_bp = Breakpoint.get_by_name(self.hparams.recon_bp)
        recon_bp.kwargs = tuple(signal)

        # Phase 1: Prefill
        with self.controller.phase("prefill"):
            _ = self.forward(images)

        # Phase 2: Mutate
        with self.controller.phase("mutate"):
            logits = self.forward(images)

        loss = self.criterion(logits, y)
        recon_loss, recon_unc_loss = self._compute_recon_loss(recon_bp.trace)
        unc = self._compute_unc_loss(logits, y)
        return loss, logits, y, {"recon_loss": recon_loss, "unc_loss": recon_unc_loss, "trace": recon_bp.trace}, unc
```

### JEPA Gradient Flow

```
Prefill:
  z0 = backbone(t1w)  [ViT token grid, grad disabled]
  z1 = backbone(t2w)  [ViT token grid, grad disabled]
  z0_pool = mean_pool(z0)  [B, 768]
  z1_pool = mean_pool(z1)  [B, 768]
  rec_0, rec_1 = BilinearReconstructor(z0_pool, z1_pool, signal)
  → mutator_enc0._buffer["reconstructor.0"] = (rec_0, rec_1)
  → mutator_enc1._buffer["reconstructor.0"] = (rec_0, rec_1)

Mutate:
  zs[0] = mutator_enc0 emits rec_0  [with grad from prefill]
  zs[1] = mutator_enc1 emits rec_1  [with grad from prefill]
  zs = [rec_0, rec_1] → classifier(zs[0], zs[1]) → logits

Backward:
  ∂loss/∂rec_0 → ∂loss/∂ln21_params ✓
  ∂loss/∂rec_1 → ∂loss/∂z1_pool → 0 (backbone frozen) ✓
```

**Important**: The JEPA backbone is frozen (`requires_grad=False`), so only
the reconstructor and uncertainty modules receive gradients. The reconstructor
is optimized by the MAIN task loss (brain age MSE), not just auxiliary
reconstruction loss. This is the critical advantage of the two-phase approach
over the single-pass method.

---

## Pretrained Model Architecture: Full `nn.Module` Tree

### Toy Manifold Model — `MultiModalRegressor`

```
MultiModalRegressor
├── encoders: ModuleList
│   ├── 0: MLP
│   │   └── net: Sequential
│   │       ├── 0: Sequential
│   │       │   ├── 0: Linear(8 → 32)
│   │       │   ├── 1: BatchNorm1d(32)
│   │       │   └── 2: GELU
│   │       ├── 1: Sequential
│   │       │   ├── 0: Linear(32 → 32)
│   │       │   ├── 1: BatchNorm1d(32)
│   │       │   └── 2: GELU
│   │       └── 2: Linear(32 → 16)
│   └── 1: MLP                          (identical structure)
│       └── net: Sequential [...]
├── head: Sequential
│   ├── 0: MLP
│   │   └── net: Sequential
│   │       ├── 0: Sequential [Linear(16→32), BatchNorm, GELU]
│   │       ├── 1: Sequential [Linear(32→16), BatchNorm, GELU]
│   │       └── 2: Linear(16 → 16)
│   └── 1: Linear(16 → 1)
```
```
Forward:
  xs = [x0, x1]                        # split modalities
  zs = [encoders[i](x) for i, x in ...]  # parallel encoder pass
  z = torch.stack(zs).sum(dim=0)        # non-Module aggregation
  return head(z).squeeze(-1)            # [B]
```

### Neuro-JEPA Pretrained Model — `MultiModalJEPARegressor`

#### Wrapper

```
MultiModalJEPARegressor  (from src/models/components/jepa.py)
├── backbone: VisionTransformer       [frozen, requires_grad=False]
├── encoders: ModuleList
│   ├── 0: ModalExtractor             [identity pass-through]
│   └── 1: ModalExtractor             [identity pass-through]
└── classifier: MultiModalLateFusion  [trainable cross-attention fusion]
```
```
Forward:
  for i, img in enumerate([t1w, t2w]):
      f = backbone(img)               # [B, N, 768] ViT token grid
      f = encoders[i](f)              # Hook target for breakpoints
  return classifier(feats[0], feats[1])  # cross-attn → [B, num_classes]
```

#### ViT Backbone — `VisionTransformer`

```
VisionTransformer  (vit_base, from submodules/Neuro-JEPA/src/neurojepa/models/vision_transformer.py)
│
│   Hyperparameters:
│     img_size=(96, 108, 96)   patch_size=(12, 12, 12)
│     embed_dim=768             depth=12           num_heads=12
│     mlp_ratio=4.0             use_rope=True      use_sdpa=True
│     wide_silu=True            uniform_power=True
│
│   # Patches per volume: (96/12)×(108/12)×(96/12) = 8×9×8 = 576 tokens
│
├── patch_embed: PatchEmbed3D
│   └── proj: Conv3d(1, 768, kernel=(12,12,12), stride=(12,12,12))
│       # [B, 1, 96, 108, 96] → [B, 768, 8, 9, 8] → flatten → [B, 576, 768]
│
├── blocks: ModuleList[12 × Block]
│   └── Block [i]  (i ∈ {0..11})
│       ├── norm1: LayerNorm(768, eps=1e-6)
│       ├── attn: RoPEAttention
│       ├── drop_path: DropPath(stochastic_depth_rate[i])
│       ├── norm2: LayerNorm(768, eps=1e-6)
│       └── mlp: SwiGLUFFN
│
└── norm: LayerNorm(768, eps=1e-6)
```
```
Forward:
  x = patch_embed(x)                  # Tokenize: Conv3D → [B, 576, 768]
  for blk in blocks:
      x, _ = blk(x, D_patches, H_patches, W_patches)  # Transformer layer
  x = norm(x)                          # Final LayerNorm
  return x, []                         # [B, 576, 768], no MoE scores
```

#### `Block` — Transformer Layer

```
Block  (from submodules/Neuro-JEPA/src/neurojepa/models/utils/modules.py)
│
├── norm1: LayerNorm(768)
├── attn: RoPEAttention
│   ├── qkv: Linear(768 → 2304)      # Combined Q, K, V projection
│   ├── proj: Linear(768 → 768)       # Output projection
│   ├── proj_drop: Dropout
│   ├── attn_drop: Dropout
│   └── proj_attn_gate: Linear(768 → 12)  # Head-wise gating
├── drop_path: DropPath
├── norm2: LayerNorm(768)
└── mlp: SwiGLUFFN
    ├── fc1: Linear(768 → 2048)       # SwiGLU gate branch
    ├── fc2: Linear(768 → 2048)       # SwiGLU value branch
    ├── act: SiLU                     # Applied to gate branch
    └── fc3: Linear(2048 → 768)       # Down-projection
```
```
Forward:
  y = attn(norm1(x), D_patches, H_patches, W_patches)
  x = x + drop_path(y)
  x = x + drop_path(mlp(norm2(x)))
  return x, None
```
```
SwiGLU:
  gate = fc1(x)                       # [B, N, 2048]
  value = fc2(x)                      # [B, N, 2048]
  hidden = SiLU(gate) * value         # element-wise gating
  return fc3(hidden)                  # [B, N, 768]

  # Note: wide_silu=True adjusts hidden dim:
  #   swiglu_hidden = int(2 * 3072 / 3) = 2048
  #   aligned to 8: 2048 → 2048
```

#### `RoPEAttention` — 3D Rotary Position Self-Attention

```
RoPEAttention  (from modules.py)
│
│   Head dimension partitioning (D=64 per head, 12 heads):
│     d_dim = 2 * ((64//3) // 2) = 20   # depth RoPE dims
│     h_dim = 2 * ((64//3) // 2) = 20    # height RoPE dims
│     w_dim = 2 * ((64//3) // 2) = 20    # width RoPE dims
│     remaining = 64 - 60 = 4            # static dims (no RoPE)
│
├── qkv: Linear(768 → 2304)          # 768 * 3 = 2304
├── proj: Linear(768 → 768)           # Output
├── proj_drop: Dropout(p=drop)
├── proj_attn_gate: Linear(768 → 12)  # Per-head attention gate
```
```
Forward:
  1. Compute per-token 3D positions:
       d_pos = token_id // (H_patches * W_patches)
       h_pos = (token_id % (H_patches * W_patches)) // W_patches
       w_pos = token_id % W_patches

  2. Project Q, K, V: qkv = Linear(x) → [B, N, 3*768]

  3. Apply 3D Rotary Position Encoding:
       q_d = rotate(q[:, :d_dim], pos=d_pos)     # depth axis
       k_d = rotate(k[:, :d_dim], pos=d_pos)
       q_h = rotate(q[:, d_dim:d_dim+h_dim], pos=h_pos)  # height axis
       k_h = rotate(k[:, d_dim:d_dim+h_dim], pos=h_pos)
       q_w = rotate(q[:, ...:w_dim], pos=w_pos)  # width axis
       k_w = rotate(k[:, ...:w_dim], pos=w_pos)
       # Remaining 4 dims are static (no RoPE)

  4. Scaled dot-product attention (SDPA):
       x = F.scaled_dot_product_attention(q, k, v)

  5. Head-wise gating:
       gate = sigmoid(proj_attn_gate(x))         # [B, N, 12, 1]
       x = x * gate                                # per-head modulation

  6. Output: proj(x) → [B, N, 768]
```

#### `PatchEmbed3D` — 3D Convolutional Tokenizer

```
PatchEmbed3D
└── proj: Conv3d(1, 768, kernel=(12,12,12), stride=(12,12,12))
    # Input:  [B, 1, 96, 108, 96]
    # Output: [B, 768, 8, 9, 8] → flatten(2) → [B, 768, 576] → transpose → [B, 576, 768]
    # Params: 1 * 768 * 12 * 12 * 12 = 1,327,104
```

#### Classifier — `MultiModalLateFusion`

```
MultiModalLateFusion  (from submodules/Neuro-JEPA/src/neurojepa/models/cross_attn.py)
│
├── proj1: ProjectionHead
│   ├── projection: Linear(768 → 512)   # expand + project
│   ├── gelu: GELU
│   ├── fc: Linear(512 → 512)
│   └── dropout: Dropout(0.1)
│   # Residual: output = fc(gelu(proj(x))) + proj(x)
│
├── proj2: ProjectionHead             (identical structure)
│
├── cross_attn_1to2: CrossAttention
│   ├── q_proj: Linear(512 → 512)     # Query from modality 1
│   ├── k_proj: Linear(512 → 512)     # Key from modality 2
│   ├── v_proj: Linear(512 → 512)     # Value from modality 2
│   ├── out_proj: Linear(512 → 512)
│   ├── attn_dropout: Dropout(0.1)
│   └── proj_dropout: Dropout(0.1)
│
├── cross_attn_2to1: CrossAttention   (identical, reversed)
│
├── norm1: LayerNorm(512)             # Post-attention norms for mod 1
├── norm2: LayerNorm(512)             # Post-attention norms for mod 2
├── norm3: LayerNorm(512)             # Pre-residual norms
├── norm4: LayerNorm(512)             # Pre-residual norms
│
├── gate: Sequential                   # Gated fusion
│   ├── 0: Linear(1024 → 512)
│   ├── 1: ReLU
│   ├── 2: Linear(512 → 512)
│   └── 3: Tanh
│
└── classifier: Linear(512 → num_classes)  # Brain age prediction head
```
```
Forward:
  feat1, feat2 = proj1(f1), proj2(f2)          # [B, N, 512] each

  # Bidirectional cross-attention
  f1_att = cross_attn_1to2(feat1, feat2)       # mod1 attends to mod2
  feat1_out = norm1(feat1 + norm3(f1_att))     # residual + norm
  f2_att = cross_attn_2to1(feat2, feat1)       # mod2 attends to mod1
  feat2_out = norm2(feat2 + norm4(f2_att))     # residual + norm

  # Mean pool token sequences
  f1_pool = feat1_out.mean(dim=1)              # [B, 512]
  f2_pool = feat2_out.mean(dim=1)              # [B, 512]

  # Gated fusion
  gate = tanh(ReLU(Linear(cat(f1_pool, f2_pool))))  # [B, 512]
  fused = gate * f1_pool + (1 - gate) * f2_pool     # [B, 512]

  return classifier(fused)                     # [B, num_classes]
```

#### `CrossAttention` — Modality Cross-Attention

```
CrossAttention  (from cross_attn.py, MultiModalLateFusion's internal module)
│
├── q_proj: Linear(512 → 512)         # Query from attending modality
├── k_proj: Linear(512 → 512)         # Key from attended modality
├── v_proj: Linear(512 → 512)         # Value from attended modality
├── out_proj: Linear(512 → 512)       # Output projection
├── attn_dropout: Dropout(0.1)
└── proj_dropout: Dropout(0.1)
```
```
Forward:
  Q = q_proj(query)                   # [B, N_q, 512] → multi-head
  K = k_proj(key_value)               # [B, N_kv, 512] → multi-head
  V = v_proj(key_value)               # [B, N_kv, 512] → multi-head
  attn = softmax(Q @ K^T / sqrt(d))   # [B, heads, N_q, N_kv]
  return out_proj(attn @ V)           # [B, N_q, 512]
```

### Parameter Counts

| Component | Parameters | Trainable |
|---|---|---|
| **ViT Backbone** | | |
| PatchEmbed3D | 1,327,104 | frozen |
| RoPEAttention × 12 | 28,311,552 | frozen |
| SwiGLUFFN × 12 | 56,623,104 | frozen |
| LayerNorms + other | ~553,000 | frozen |
| *Backbone subtotal* | *~86.8M* | *0* |
| **Classifier** | | |
| ProjectionHead × 2 | 1,574,912 | yes |
| CrossAttention × 2 | 4,198,400 | yes |
| LayerNorm × 4 | 4,096 | yes |
| Gate (Sequential) | 788,480 | yes |
| classifier (Linear) | ~513 / ~1,025 | yes |
| *Classifier subtotal* | *~6.6M* | *~6.6M* |
| **Total** | **~93.4M** | **~6.6M** |

### Data Flow Through Breakpoints

```
Input: {t1w: [B,1,96,108,96], t2w: [B,1,96,108,96]}

MultiModalJEPARegressor.forward():
│
├─ [1] backbone(t1w) ──► f1 [B,576,768]  (frozen, torch.no_grad)
│   └─ encoders[0](f1)                     HOOK TARGET: encoders.0 (after)
│                                          src_enc0 captures f1
│                                          mutator_enc0 may replace f1
│
├─ [2] backbone(t2w) ──► f2 [B,576,768]  (frozen, torch.no_grad)
│   └─ encoders[1](f2)                     HOOK TARGET: encoders.1 (after)
│                                          src_enc1 captures f2
│                                          mutator_enc1 may replace f2
│
├─ [3] classifier(f1, f2)
│   ├─ proj1(f1) → [B,576,512]
│   ├─ proj2(f2) → [B,576,512]
│   ├─ cross_attn_1to2 + cross_attn_2to1
│   ├─ mean pool → [B,512], [B,512]
│   ├─ gate fusion → [B,512]
│   └─ classifier(fused) → [B, num_classes]   HOOK TARGET: classifier (after)
│                                               uncertainty BP reads logits
```

With two-phase feedback on the JEPA model, the reconstructor mean-pools the
ViT token grids before reconstruction (using `MeanPoolCollectedFn`), and
mutator BPs replace the token grids with pooled reconstructed latents.
The classifier expects `[B, N, D]` token sequences, so the reconstructed
latents must be reshaped or the classifier must accept pooled inputs — in
practice, the mutator replaces the token grid with a pooled latent that the
projection heads can still process (since `ProjectionHead` applies `Linear`
over the last dimension regardless of sequence length).

---

## Running

### Manifold Toy Model

```bash
# Checkpoint must exist at data/checkpoints/checkpoint_multimodal_manifold.pth

# Train with two-phase feedback
python src/train_hook_dag.py --config-name train_hook_dag_feedback

# Run gradient flow validation
python tests/test_feedback_gradient.py
```

### JEPA Pretrained Model

```bash
# First download the checkpoint
python scripts/download_jepa_checkpoint.py

# Train with two-phase feedback (requires hook_dag_jepa_feedback.yaml config)
python src/train_hook_dag_jepa_brain_age.py --config-name train_hook_dag_jepa_feedback
```

---

## Test Results

```
Test 1: Gradient flow through reconstructor ........... PASS
  - reconstructor.ln21 params: grad norm = 2.891278
  - encoders.1 params: grad norm = 10.354498
  - mutator_enc0 emits feedback value from 'reconstructor.0' (index=0)

Test 2: Phase context manager isolation ............... PASS
  - prefill/mutate/default phases correctly set and restored
  - Nested phase contexts work correctly

Test 3: MutatorCallback phase behavior ................ PASS
  - Prefill and mutate produce different outputs
  - True in-place replacement confirmed

All tests passed.
```

---

## Files Reference

| File | Role |
|---|---|
| `src/plugins/hook_dag.py` | Core DAG framework (+3 additive changes) |
| `src/plugins/reconstructor/feedback.py` | `MutatorCallback`, `FeedbackReconstructor` |
| `src/plugins/reconstructor/linear.py` | `BilinearReconstructor` (unchanged) |
| `src/plugins/reconstructor/pool.py` | `MeanPoolCollectedFn` (for JEPA token pooling) |
| `src/models/hook_modules/feedback_module.py` | `FeedbackInjectModule` (two-phase training) |
| `configs/plugins/hook_dag_feedback.yaml` | Config for manifold model (6 BPs) |
| `configs/train_hook_dag_feedback.yaml` | Training config for manifold model |
| `tests/test_feedback_gradient.py` | Gradient flow validation (3 tests) |
| `src/models/components/jepa.py` | `MultiModalJEPARegressor` (unchanged) |
| `src/models/components/toy.py` | `MultiModalRegressor` (unchanged) |
| `claude/BACKPROP_HOOK_INPLACE.md` | This document |

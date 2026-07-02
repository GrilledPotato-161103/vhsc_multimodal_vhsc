# Neuro-JEPA Brain Age Estimation — Hook DAG Implementation

## Table of Contents

1. [Neuro-JEPA Architecture](#1 Need to re-read architecture)
2. [Pretrained Checkpoint Structure](#2-pretrained-checkpoint-structure)
3. [MultiModalJEPARegressor Module Tree](#3-multimodaljeparegressor-module-tree)
4. [Breakpoint Hook DAG System](#4-breakpoint-hook-dag-system)
5. [Breakpoint Placement on JEPA](#5-breakpoint-placement-on-jepa)
6. [Reconstructor](#6-reconstructor)
7. [Uncertainty Estimator (BayesCap)](#7-uncertainty-estimator-bayescap)
8. [Brain Age Estimation Pipeline](#8-brain-age-estimation-pipeline)
9. [Data Flow Through Breakpoints](#9-data-flow-through-breakpoints)
10. [Training Protocol](#10-training-protocol)
11. [File Reference](#11-file-reference)

---

## 1. Neuro-JEPA Architecture

Neuro-JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning framework for 3D brain MRI based on the V-JEPA2 architecture from Meta. It uses a Vision Transformer (ViT) backbone pretrained via masked prediction: given visible patches (context), the model predicts latent representations of masked patches (targets) in embedding space.

### 1.1 ViT Backbone (`VisionTransformer`)

```
VisionTransformer
├── patch_embed: PatchEmbed3D        # Conv3d tokenizer
│   └── proj: Conv3d(in=1, out=768, kernel=(12,12,12), stride=(12,12,12))
├── blocks: ModuleList[Block × 12]   # Transformer blocks with RoPE
│   ├── blocks.0
│   │   ├── norm1: LayerNorm(768)
│   │   ├── attn: RoPEAttention     # 3D rotary position encoding
│   │   │   ├── qkv: Linear(768→2304)
│   │   │   ├── proj: Linear(768→768)
│   │   │   └── proj_attn_gate: Linear(768→12)  # head gating
│   │   ├── drop_path: DropPath
│   │   ├── norm2: LayerNorm(768)
│   │   └── mlp: SwiGLUFFN          # SwiGLU feedforward
│   │       ├── fc1: Linear(768→2048)
│   │       ├── fc2: Linear(768→2048)
│   │       └── fc3: Linear(2048→768)
│   ├── blocks.1 ... blocks.11      # Same structure
├── norm: LayerNorm(768)            # Final normalization
```

**Key parameters** (vit_base for brain MRI):
| Parameter | Value |
|---|---|
| `img_size` | (96, 108, 96) — D×H×W |
| `patch_size` | (12, 12, 12) |
| `num_patches` | 8×9×8 = **576 tokens** |
| `embed_dim` | 768 |
| `depth` | 12 blocks |
| `num_heads` | 12 (64 dim/head) |
| `mlp_ratio` | 4.0 (SwiGLU hidden 2048) |
| `in_chans` | 1 (single-channel MRI) |
| Position encoding | **RoPE** (3D rotary: depth + height + width axes) |
| `use_sdpa` | True (Flash Attention via PyTorch SDPA) |

### 1.2 JEPA Pretraining Objective

The pretraining uses a **teacher-student** setup with momentum encoder:
- **Context encoder** (student): processes visible patches → latent representations
- **Target encoder** (teacher): processes masked patches → target latents (no gradient)
- **Predictor**: small transformer that predicts target latents from context latents + mask token positions

The loss is the L2 distance between predicted and target latent representations, computed only on masked positions.

### 1.3 Cross-Attention Classifier (`MultiModalLateFusion`)

For downstream tasks, a late-fusion cross-attention classifier sits on top of the frozen ViT:

```
MultiModalLateFusion (embed_dim=768, proj_dim=512, num_heads=8)
├── proj1: ProjectionHead(768→512→512)      # Modality 1 projection
│   ├── projection: Linear(768→512)
│   ├── fc: Linear(512→512)
│   └── residual connection
├── proj2: ProjectionHead(768→512→512)      # Modality 2 projection
├── cross_attn_1to2: CrossAttention(512, 8 heads)  # T1w attends to T2w
│   ├── q_proj, k_proj, v_proj, out_proj
├── cross_attn_2to1: CrossAttention(512, 8 heads)  # T2w attends to T1w
├── norm1-4: LayerNorm(512) × 4
├── gate: Sequential(                        # Gated fusion
│   ├── Linear(1024→512)
│   ├── ReLU
│   ├── Linear(512→512)
│   └── Tanh
├── classifier: Linear(512→1)               # Brain age regression
```

**Fusion flow**:
```
feat1 [B,576,768] ─► proj1 ─► [B,576,512] ─► cross_attn_1to2(q=feat1, kv=feat2) ─► +residual+norm ─► mean_pool ─► [B,512]
feat2 [B,576,768] ─► proj2 ─► [B,576,512] ─► cross_attn_2to1(q=feat2, kv=feat1) ─► +residual+norm ─► mean_pool ─► [B,512]
                                                                                                   │
                                                                                          gate(cat)├──► classifier ─► brain_age [B,1]
```

---

## 2. Pretrained Checkpoint Structure

### 2.1 Source

The backbone is downloaded from HuggingFace Hub: **`NYUMedML/Neuro-JEPA`**

- Files: `model.safetensors` or `pytorch_model.bin`
- The checkpoint contains the full JEPA training state (encoder + target_encoder + predictor)
- Only the encoder portion is extracted for downstream use

### 2.2 State Dict Cleaning

The raw checkpoint has wrapper prefixes that must be stripped:

```
Raw key:  student.vision_encoder.blocks.0.norm1.weight
Cleaned:  blocks.0.norm1.weight

Raw key:  target_encoder.blocks.0.attn.qkv.weight
Cleaned:  blocks.0.attn.qkv.weight
```

Prefixes removed (in order): `_checkpoint_wrapped_module.`, `student.vision_encoder.`, `student.encoder.`, `vision_encoder.`, `target_encoder.`, `encoder.`, `model.`, `module.`, `_orig_mod.`, `backbone.`

### 2.3 Saved Regressor Format

The full `MultiModalJEPARegressor` is saved locally at `data/checkpoints/jepa/jepa_pretrained.pth` via `torch.save()`. This pickle includes:
- `backbone` (VisionTransformer) — frozen, pretrained weights
- `classifier` (MultiModalLateFusion) — randomly initialized
- `encoders` (ModuleList of ModalExtractor) — identity wrappers
- `modality_keys` — `["t1w", "t2w"]`
- `image_size` — `(96, 108, 96)`

---

## 3. MultiModalJEPARegressor Module Tree

```python
MultiModalJEPARegressor(
  modality_keys=["t1w", "t2w"],
  image_size=(96, 108, 96),
  n_modals=2,
  _num_classes=1,
)
├── backbone: VisionTransformer(           # SHARED frozen ViT
│   ├── patch_embed: PatchEmbed3D          # Conv3d(1→768, k=12³, s=12³)
│   ├── blocks: ModuleList[Block × 12]     # Transformer blocks with RoPE
│   │   ├── 0: Block
│   │   │   ├── norm1: LayerNorm(768)
│   │   │   ├── attn: RoPEAttention(768, 12 heads, RoPE 3D)
│   │   │   ├── drop_path: DropPath
│   │   │   ├── norm2: LayerNorm(768)
│   │   │   └── mlp: SwiGLUFFN(768→2048→768)
│   │   ├── 1..11: Block (×11)
│   │   └── ...
│   └── norm: LayerNorm(768)
├── encoders: ModuleList[                   # Per-modality hook targets
│   ├── 0: ModalExtractor()                # encoders.0 — identity pass-through
│   └── 1: ModalExtractor()                # encoders.1 — identity pass-through
│ ]
└── classifier: MultiModalLateFusion(       # Cross-attention fusion + regression
    ├── proj1: ProjectionHead(768→512→512)
    ├── proj2: ProjectionHead(768→512→512)
    ├── cross_attn_1to2: CrossAttention(512, 8 heads)
    ├── cross_attn_2to1: CrossAttention(512, 8 heads)
    ├── norm1-4: LayerNorm(512) × 4
    ├── gate: Sequential(Linear(1024→512), ReLU, Linear(512→512), Tanh)
    └── classifier: Linear(512→1)
)
```

**All hookable named modules:**
```
backbone.patch_embed
backbone.patch_embed.proj
backbone.blocks.0           through    backbone.blocks.11
backbone.blocks.0.norm1     through    backbone.blocks.11.mlp.fc3
backbone.blocks.0.attn      through    backbone.blocks.11.attn
backbone.blocks.0.mlp       through    backbone.blocks.11.mlp
backbone.norm
encoders.0                             ← HOOK TARGET
encoders.1                             ← HOOK TARGET
classifier                             ← HOOK TARGET
classifier.proj1 / .proj2
classifier.cross_attn_1to2 / .cross_attn_2to1
classifier.norm1-4
classifier.gate / .classifier
```

---

## 4. Breakpoint Hook DAG System

### 4.1 Core Classes

**`Breakpoint`** (`src/plugins/hook_dag.py`):
- A named hook point on a specific `nn.Module` layer
- `callback`: callable that receives `BreakpointContext` and returns `BreakpointOutput`
- `mutate`: if True, replaces the module's input/output with the callback's output
- `data_sources`: list of upstream breakpoint names to collect data from
- `data_sinks`: downstream breakpoints that receive this breakpoint's data
- `pre_fn`: transforms `BreakpointContext` before callback sees it
- `post_fn`: transforms `BreakpointOutput` after callback returns
- Global registry: `Breakpoint.list_of_breakpoints: Dict[str, List[Breakpoint]]`

**`BreakpointController`** (`src/plugins/hook_dag.py`):
- Manages hook registration, DAG wiring, cycle detection, and serialization
- `add_breakpoint(root, target, bp, position)`: registers a forward hook
- `wire()`: resolves `data_sources` → `data_sinks` references, validates DAG for cycles
- `state_dict()` / `save()` / `load()`: serialization for checkpointing

**`BreakpointContext`** (`src/plugins/var.py`):
- Runtime context passed to callbacks: name, layer, position, module, inputs, output, kwargs, state, collected (upstream data)

**`BreakpointOutput`** (`src/plugins/var.py`):
- Return type: fn_name, context, output (data pushed to sinks), trace (debug), valid

### 4.2 DAG Data Flow

```
                         ┌──────────────────┐
                         │  Source BP        │
                         │  callback=None    │
                         │  data_sinks=[B,C] │
                         └──────┬───────────┘
                                │ pushes raw hook data
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌──────────┐
              │  BP B    │ │  BP C    │      ← data_sources=[src]
              │  pre_fn  │ │  pre_fn  │
              │  callback│ │  callback│
              │  mutate=T│ │  valid=T │
              └──────────┘ └──────────┘
```

### 4.3 Built-in pre_fn Utilities

| Class | File | Purpose |
|---|---|---|
| `ConcatCollectedFn` | `hook_dag.py` | Concatenate all collected tensors along dim=-1 |
| `ToListCollectedFn` | `hook_dag.py` | Pack collected tensors into tuple |
| `SumCollectedFn` | `hook_dag.py` | Element-wise sum all collected tensors |
| `MeanPoolCollectedFn` | `reconstructor/pool.py` | Mean-pool each collected tensor along dim=1 (for ViT tokens) |
| `MeanPoolInputFn` | `reconstructor/pool.py` | Mean-pool `ctx.inputs[0]` along dim=1 |
| `PreprocessCollectedFn` | `hook_dag.py` | Apply per-key processing functions to collected dict |
| `SumPostOp` | `hook_dag.py` | Post_fn: sum a list of tensors in BreakpointOutput |

---

## 5. Breakpoint Placement on JEPA

### 5.1 Breakpoint Configuration

All four breakpoints and their DAG wiring:

```yaml
# configs/plugins/hook_dag_jepa_brain_age.yaml

breakpoints:

  # SOURCE BP 1: Capture T1w ViT token features after ModalExtractor
  - layer_name: encoders.0
    bp:
      name: src_enc0           # Registered as src_enc0.0
      # callback=None → pure source: pushes raw hook data to sinks
    pos: after                  # fires after ModalExtractor.forward()

  # SOURCE BP 2: Capture T2w ViT token features after ModalExtractor
  - layer_name: encoders.1
    bp:
      name: src_enc1           # Registered as src_enc1.0
      # callback=None → pure source
    pos: after

  # RECONSTRUCTOR BP: collects encoder outputs, reconstructs missing modality
  - layer_name: classifier
    bp:
      name: reconstructor      # Registered as reconstructor.0
      data_sources: [src_enc0.0, src_enc1.0]  # DAG: reads from both sources
      pre_fn: MeanPoolCollectedFn             # [B,576,768] → [B,768]
      callback: BilinearReconstructor(d_1=768, d_2=768, ...)
      mutate: true                            # Replaces classifier input
      valid: true
    pos: before                # fires before classifier.forward()

  # UNCERTAINTY BP: reads classifier output, estimates predictive uncertainty
  - layer_name: classifier
    bp:
      name: uncertainty        # Registered as uncertainty.0
      data_sources: [input]    # DAG: reads model's classifier output
      callback: BayesCap1D(input_dim=1, output_dim=1, ...)
      mutate: false            # Observer only
      valid: true
    pos: after                 # fires after classifier.forward()
```

### 5.2 Why These Four Breakpoints?

| Breakpoint | Rationale |
|---|---|
| `encoders.0` (after) | Captures T1w ViT token features — the representation that enters the cross-attention fusion. Essential for reconstruction when T1w is masked. |
| `encoders.1` (after) | Captures T2w ViT token features — symmetric to encoders.0. |
| `classifier` (before) | Intercepts the input to cross-attention. When a modality is masked, the reconstructor replaces it with a reconstructed version from the available modality. `mutate=true` overwrites the inputs. |
| `classifier` (after) | Captures the final brain age prediction. BayesCap estimates uncertainty over this scalar output. |

### 5.3 Hook Lifecycle During Forward Pass

```
Forward pass for (t1w, t2w) batch:

1. backbone(t1w) → feat1 [B,576,768]  (frozen, no_grad)
2. encoders.0(feat1) → feat1          (identity)
   └─ HOOK: src_enc0.0.run_after()    → pushes feat1 to reconstructor._buffer["src_enc0.0"]

3. backbone(t2w) → feat2 [B,576,768]  (frozen, no_grad)
4. encoders.1(feat2) → feat2          (identity)
   └─ HOOK: src_enc1.0.run_after()    → pushes feat2 to reconstructor._buffer["src_enc1.0"]

5. ── classifier.forward() called ──
   └─ HOOK: reconstructor.0.run_before()
      ├─ _buffer = {"src_enc0.0": feat1, "src_enc1.0": feat2}
      ├─ pre_fn: mean_pool each → (feat1_pooled [B,768], feat2_pooled [B,768])
      ├─ callback: BilinearReconstructor(ctx)
      │   ├─ if signal=(1,1): pass through (both available)
      │   ├─ if signal=(0,1): reconstruct feat1 from feat2 via ln21
      │   └─ if signal=(1,0): reconstruct feat2 from feat1 via ln12
      └─ mutate=true → replaces classifier input with reconstructed latents

6. classifier(feat1_recon, feat2_recon) → brain_age [B,1]
   └─ HOOK: uncertainty.0.run_after()
      ├─ data_sources=["input"] → reads brain_age from ctx.output
      ├─ callback: BayesCap1D(brain_age)
      │   → (mu [B,1], alpha [B,1], beta [B,1])  # NIG parameters
      └─ trace stores (mu, alpha, beta) for loss computation
```

### 5.4 DAG Wiring (Internal Resolution)

When `controller.wire()` is called after all breakpoints are added:

```
src_enc0.0.data_sinks = [reconstructor.0]
src_enc1.0.data_sinks = [reconstructor.0]
reconstructor.0.data_sources = [src_enc0.0, src_enc1.0]  (resolved)
uncertainty.0.data_sources = ["input"]                     (runtime keyword, no resolution needed)
```

Cycle detection via DFS tricolor: the DAG is valid because edges only flow `src → reconstructor` and `input → uncertainty` (no back edges).

---

## 6. Reconstructor

### 6.1 BilinearReconstructor (`src/plugins/reconstructor/linear.py`)

A cross-modal reconstructor that learns bidirectional mappings between two latent spaces.

```
BilinearReconstructor(d_1=768, d_2=768, hidden_dims=[512,256,128])
├── ln12: MLP(768→[512,256,128]→768)   # T1w → T2w mapping
│   └── with residual connections
├── ln21: MLP(768→[512,256,128]→768)   # T2w → T1w mapping
├── dev1: FeedForward(1536→1536→768)    # Deviation predictor 1
├── dev2: FeedForward(1536→1536→768)    # Deviation predictor 2
└── dist: MSELoss(reduction='none')     # Distance metric
```

**Forward logic** (keyed by `ctx.bp_kwargs` signal):

| Signal `(p1, p2)` | Meaning | Action |
|---|---|---|
| `(1, 1)` | Both modalities present | `rec_2 = mod_2` (no reconstruction needed) |
| `(0, 1)` | T1w masked | `rec_2 = ln12(mod_1)` — predict T2w from T1w |
| `(1, 0)` | T2w masked | `rec_1 = ln21(mod_2)` — predict T1w from T2w |

The deviation heads (`dev1`, `dev2`) learn to predict the reconstruction error `dist(rec, src)`, which contributes to the reconstruction uncertainty loss — a self-supervised signal for how reliable each cross-modal prediction is.

**Output**: `(rec_1, rec_2)` — both tensors `[B, 768]`, replacing the classifier's input latent pair.

### 6.2 MeanPoolCollectedFn — ViT Token Aggregation

Since ViT outputs `[B, 576, 768]` token grids and `BilinearReconstructor` expects flat `[B, 768]` vectors, `MeanPoolCollectedFn` is applied as a `pre_fn`:

```
collected = {"src_enc0.0": [B,576,768], "src_enc1.0": [B,576,768]}
                                   │
                    MeanPoolCollectedFn.forward(ctx)
                                   │
         ctx.inputs = (T1w_pooled [B,768], T2w_pooled [B,768])
```

This is done **before** the reconstructor callback, keeping the reconstructor itself modality-agnostic.

### 6.3 Reconstruction Loss

Two components, computed from the reconstructor's trace after each forward pass:

1. **Reconstruction fidelity**: `MSELoss(rec, src)` — how close is the reconstructed latent to the real one?
2. **Reconstruction uncertainty**: `MSELoss(dev, dist)` — how well can the deviation head predict its own error?

Only computed for modalities where `sig == 1` (i.e., the modality IS present, meaning reconstruction is evaluated against ground truth).

---

## 7. Uncertainty Estimator (BayesCap)

### 7.1 BayesCap1D (`src/plugins/head/bayescap.py`)

A neural network that maps a point prediction to Normal-Inverse-Gamma (NIG) parameters, providing full predictive uncertainty:

```
BayesCap1D(input_dim=1, output_dim=1, hidden_dims=[32,16,8], per_dim_uncertainty=False)
├── stem: Sequential(Linear(1→32), SiLU)
├── blocks: MLP(32→[16]→8)         # Residual MLP
├── mu_head: MLP(8→[8]→1)          # Calibrated mean
├── alpha_head: MLP(8→[8]→1)       # Evidence parameter → softplus
└── beta_head: MLP(8→[8]→1)        # Scale parameter → softplus
```

**Input**: `ctx.output` = classifier's brain age prediction `[B, 1]`

**Output**: `(mu, alpha, beta)` where:
- `mu`: calibrated mean prediction `[B, 1]`
- `alpha`: inverse evidence `[B, 1]` (lower = more evidence = more certain)
- `beta`: scale parameter `[B, 1]`

**Predictive variance**: `Var = alpha² × Γ(3/β) / Γ(1/β)` — higher alpha or lower beta → wider predictive distribution.

### 7.2 BayesCap1DLoss

Two-term loss:
1. **Identity loss**: `||mu - y_hat||²` — keeps calibrated mean close to original prediction
2. **NLL loss**: Generalized Gaussian NLL: `(|mu - y| × inv_alpha)^beta - log(beta) - log(inv_alpha) + log Γ(1/beta)`

The NLL term calibrates uncertainty against ground-truth brain age.

---

## 8. Brain Age Estimation Pipeline

### 8.1 Task Definition

**Input**: Paired 3D brain MRI volumes (T1w, T2w) as 5D tensors `[B, 1, 96, 108, 96]`

**Output**: Predicted brain age (continuous scalar) + per-sample predictive uncertainty

**Metric**: MAE (years), plus NLL (uncertainty calibration quality)

### 8.2 Full Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop                                 │
│                                                                 │
│  Batch: ({t1w: [B,1,96,108,96], t2w: [B,1,96,108,96]}, age)   │
│                                                                 │
│  1. Random modality masking (p=mask_rate each modality)         │
│     ┌─ signal (0,1): T1w masked, must reconstruct from T2w     │
│     ├─ signal (1,0): T2w masked, must reconstruct from T1w     │
│     └─ signal (1,1): both present, no reconstruction needed    │
│                                                                 │
│  2. Frozen JEPA forward:                                        │
│     backbone(t1w) → feat1 [B,576,768]                           │
│     backbone(t2w) → feat2 [B,576,768]                           │
│                                                                 │
│  3. Hook DAG fires:                                             │
│     src_enc0, src_enc1 → reconstructor(mean_pool→reconstruct)  │
│     → classifier(reconstructed_latents) → brain_age            │
│     → uncertainty(brain_age) → (mu, alpha, beta)               │
│                                                                 │
│  4. Loss = MSE(age_pred, age_true)                             │
│          + recon_loss(signal)      # reconstruction fidelity   │
│          + recon_unc_loss(signal)  # deviation head            │
│          + unc_loss(mu,alpha,beta,age)  # BayesCap NLL        │
│                                                                 │
│  5. Backprop through:                                           │
│     - reconstructor.callback (BilinearReconstructor)            │
│     - uncertainty.callback (BayesCap1D)                         │
│     ✓ backbone FROZEN (no_grad, eval mode)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Phase Training Strategy

```
Epochs 0..epoch_phase-1 (default: 10):
  - When signal != (1,1): uncertainty loss *= 0  (suppressed)
  - Focus: train reconstructor to produce good latents from masked inputs
  - The reconstructor must stabilize before uncertainty estimation is useful

Epochs epoch_phase+..:
  - All losses active
  - Uncertainty head learns to calibrate based on reconstruction quality
```

---

## 9. Data Flow Through Breakpoints

### 9.1 Step-by-step with Tensor Shapes

```
Input: t1w [B,1,96,108,96], t2w [B,1,96,108,96]

Step 1: ViT Backbone (frozen, torch.no_grad)
  t1w → backbone.patch_embed(t1w)       → [B, 576, 768]
      → backbone.blocks[0..11]           → [B, 576, 768]
      → backbone.norm                    → [B, 576, 768]
  t2w → (same)                           → [B, 576, 768]

Step 2: ModalExtractors (identity)
  encoders.0(feat1) → feat1 [B, 576, 768]
    └─ [HOOK after] src_enc0.0.run_after()
       └─ pushes [B,576,768] to reconstructor._buffer["src_enc0.0"]

  encoders.1(feat2) → feat2 [B, 576, 768]
    └─ [HOOK after] src_enc1.0.run_after()
       └─ pushes [B,576,768] to reconstructor._buffer["src_enc1.0"]

Step 3: Reconstructor (before classifier)
  reconstructor._buffer = {
    "src_enc0.0": Tensor[B,576,768],
    "src_enc1.0": Tensor[B,576,768]
  }
    └─ [HOOK before] reconstructor.0.run_before()
       ├─ pre_fn: MeanPoolCollectedFn
       │   ctx.inputs = (t1w_pooled [B,768], t2w_pooled [B,768])
       ├─ callback: BilinearReconstructor(ctx)
       │   signal = ctx.bp_kwargs = (p1, p2) from random mask
       │   if p1==0: rec = ln21(t2w_pooled)  reconstruct T1w from T2w
       │   if p2==0: rec = ln12(t1w_pooled)  reconstruct T2w from T1w
       │   output = (rec_1 [B,768], rec_2 [B,768])
       │   trace = {signal, input, reconstructed, distance, dev}
       │   pushes output to uncertainty._buffer["reconstructor.0"] (no sinks use it)
       └─ mutate=true: replaces classifier input with output

Step 4: Classifier
  classifier(rec_1, rec_2) → brain_age [B,1]
    └─ [HOOK after] uncertainty.0.run_after()
       ├─ _buffer: empty (data_sources=["input"] → read from ctx.output)
       ├─ callback: BayesCap1D(ctx)
       │   ctx.output = brain_age [B,1]
       │   → mu [B,1], alpha [B,1], beta [B,1]
       └─ trace = {signal, input, output: (mu, alpha, beta)}

Step 5: Loss computation
  recon_loss = MSELoss(rec, src)  for available modalities
  unc_loss   = BayesCap1DLoss(mu, alpha, beta, logits=age_pred, y_true=age)
  total_loss = MSE(age_pred, age) + recon_loss + unc_loss
```

---

## 10. Training Protocol

### 10.1 Optimizer Configuration

**Optimized parameters** (only hook callbacks, not the backbone):
- `BilinearReconstructor` (ln12, ln21, dev1, dev2, dist)
- `BayesCap1D` (stem, blocks, mu_head, alpha_head, beta_head)

**Optimizer**: AdamW with `lr=1e-4`, `weight_decay=1e-4`

**Scheduler**: ReduceLROnPlateau, `mode=min`, `factor=0.5`, `patience=10`, monitoring `val/loss_nll_11`

### 10.2 Loss Weighting

| Loss Component | Weight | When Active |
|---|---|---|
| Regression MSE | 1.0 | Always |
| Reconstruction fidelity | 1.0 | Only for available modalities (sig=1) |
| Reconstruction uncertainty | 1.0 | Only for available modalities (sig=1) |
| BayesCap identity | 1.0 | Always |
| BayesCap NLL | 0.05 | Epochs ≥ epoch_phase (or all if signal full) |

### 10.3 Key Design Decisions

1. **Backbone completely frozen**: `requires_grad_(False)` and kept in `eval()` mode. The ViT is a fixed feature extractor.
2. **Reconstructor mutates classifier inputs**: The reconstructor physically replaces the cross-attention input latents. This means the classifier always sees "complete" inputs even when a modality was masked.
3. **Uncertainty on final output, not intermediate**: BayesCap operates on the scalar brain age prediction, not on latent features. This gives interpretable per-sample uncertainty in years.
4. **Mean-pooling for token aggregation**: ViT token grids are collapsed to single vectors before reconstruction. This is lossy but keeps the reconstructor architecture simple (MLP-based rather than transformer-based).
5. **Source breakpoints have no callback**: `src_enc0` and `src_enc1` are pure data sources (`callback=None`). They only push raw hook data to downstream consumers via `data_sinks`.

---

## 11. File Reference

### Source Files

| File | Purpose |
|---|---|
| `src/plugins/hook_dag.py` | `Breakpoint`, `BreakpointController`, DAG wiring, built-in pre/post fns |
| `src/plugins/hook.py` | Simplified Breakpoint (no DAG — deprecated in favor of hook_dag) |
| `src/plugins/var.py` | `BreakpointContext`, `BreakpointOutput`, `EndpointSpec`, formatting utilities |
| `src/plugins/reconstructor/linear.py` | `BilinearReconstructor` — cross-modal latent reconstructor |
| `src/plugins/reconstructor/identity.py` | `IdentityHook` — pass-through observer |
| `src/plugins/reconstructor/pool.py` | `MeanPoolCollectedFn`, `MeanPoolInputFn` — ViT token pooling |
| `src/plugins/head/bayescap.py` | `BayesCap1D` (uncertainty head), `BayesCap1DLoss`, `EKFGGDNLLLoss` |
| `src/plugins/head/ekf.py` | `EKFBiModalInferer` — EKF-based uncertainty propagation |
| `src/models/components/jepa.py` | `MultiModalJEPARegressor`, `ModalExtractor`, `build_jepa_regressor()` |
| `src/models/hook_modules/jepa_brain_age.py` | `JEPABrainAgeModule` — LightningModule for brain age hook training |
| `src/models/hook_modules/toy.py` | `ModelInjectModule` — reference implementation for hook training |
| `src/models/hook_modules/ekf_manifold.py` | `ModelEKFManifoldModule` — EKF variant with sigma_z |
| `src/data/jepa_brain_age_datamodule.py` | `JEPABrainAgeDataModule` + `SyntheticBrainAgeDataset` |
| `src/train_hook_dag_jepa_brain_age.py` | Training entry point for brain age estimation |
| `scripts/download_jepa_checkpoint.py` | Standalone script to download Neuro-JEPA from HF Hub |

### Config Files

| File | Purpose |
|---|---|
| `configs/train_hook_dag_jepa_brain_age.yaml` | Top-level Hydra config |
| `configs/plugins/hook_dag_jepa_brain_age.yaml` | Breakpoint DAG definition (4 breakpoints) |
| `configs/model/hook_dag_jepa_brain_age.yaml` | Model config (JEPABrainAgeModule params) |
| `configs/data/jepa_brain_age.yaml` | Data config (brain age datamodule) |
| `configs/plugins/hook_dag.yaml` | Reference: toy MultiModalRegressor hook config |
| `configs/train_hook_dag.yaml` | Reference: toy hook DAG training config |

### Neuro-JEPA Submodule

| File | Purpose |
|---|---|
| `submodules/Neuro-JEPA/src/neurojepa/models/vision_transformer.py` | `VisionTransformer` (ViT backbone) |
| `submodules/Neuro-JEPA/src/neurojepa/models/cross_attn.py` | `MultiModalLateFusion` (classifier) |
| `submodules/Neuro-JEPA/src/neurojepa/models/utils/modules.py` | `Block`, `RoPEAttention`, `SwiGLUFFN`, `MLP` |
| `submodules/Neuro-JEPA/src/neurojepa/models/utils/patch_embed.py` | `PatchEmbed3D` (Conv3d tokenizer) |
| `submodules/Neuro-JEPA/src/neurojepa/models/utils/pos_embs.py` | 3D sincos position embeddings |
| `submodules/Neuro-JEPA/src/neurojepa/models/predictor.py` | `VisionTransformerPredictor` (JEPA predictor) |
| `submodules/Neuro-JEPA/src/neurojepa/engines/pretrain.py` | JEPA pretraining loop |
| `submodules/Neuro-JEPA/src/neurojepa/loss/jepa_loss.py` | JEPA contrastive/predictive loss |

### Checkpoint Paths

| Path | Content |
|---|---|
| `data/checkpoints/jepa/jepa_pretrained.pth` | Full `MultiModalJEPARegressor` (backbone + fresh classifier) |
| `data/checkpoints/jepa/` | Downloaded from `NYUMedML/Neuro-JEPA` on HF Hub |

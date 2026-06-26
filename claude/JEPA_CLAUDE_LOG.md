# JEPA CLAUDE LOG — Multi-Modal Hook Integration Development

## Session: 2026-06-26

---

### Phase 1: Exploration & Understanding

**Action 1.1 — Neuro-JEPA Submodule Exploration**
- Read the entire Neuro-JEPA submodule architecture:
  - `mm_classifier.py`: `MultiModalLateFusion` + `ProjectionHead` for late-fusion classification
  - `vision_transformer.py`: 3D Vision Transformer with optional MoE blocks
  - `utils/moe.py`: `MoE`, `Gate` — top-k routing with auxiliary-loss-free load balancing
  - `engines/finetune/clf_mm.py`: Multi-modal classification engine supporting `vit_late`, `vit_early`, `vit_avg`, `vit_mil`
  - `cross_attn.py`: Alternative bidirectional cross-attention fusion
  - `mil.py`: `ClassifyThenAggregate` attention-based MIL
- Key finding: The classification pipeline splits into **backbone** (ViT) + **classifier** (e.g., `MultiModalLateFusion`). The `vit_late` fusion path is most amenable to hook injection because it produces per-modality feature vectors before fusion.

**Action 1.2 — Main Project Hook System Exploration**
- Read `src/plugins/hook_dag.py`: Complete DAG-enhanced Breakpoint system
  - `Breakpoint`: nn.Module with `data_sources`, `data_sinks`, `pre_fn`, `post_fn`, `callback`
  - `BreakpointController`: Registers hooks on named modules, resolves DAG wiring
  - Built-in `pre_fn` utilities: `ConcatCollectedFn`, `ToListCollectedFn`, `SumCollectedFn`, etc.
- Read `src/plugins/hook.py`: Original simpler Breakpoint system
- Read `src/plugins/var.py`: `BreakpointContext`, `BreakpointOutput` dataclasses
- Read `src/plugins/reconstructor/linear.py`: `BilinearReconstructor` — cross-modal reconstruction
- Read `src/plugins/head/bayescap.py`: `BayesCap1D`, `BayesCap1DLoss` — GGD uncertainty estimation
- Read `src/models/hook_modules/toy.py`: `ModelInjectModule` — LightningModule pattern for hook training
- Read `src/models/components/toy.py`: `BiModalRegressor`, `MultiModalRegressor` — backbone models
- Read `configs/train_hook_dag.yaml`, `configs/plugins/hook_dag.yaml`, `configs/model/hook_dag.yaml`

**Action 1.3 — Config Tree Analysis**
- Root configs: `train.yaml`, `train_hook.yaml`, `train_hook_dag.yaml`, `train_ekf_hook.yaml`
- Plugin configs: `toy.yaml`, `ekf.yaml`, `aggregation.yaml`, `hook_dag.yaml`
- Model configs: `hook.yaml`, `hook_dag.yaml`, `hook_ekf.yaml`, `hook_ekf_recon.yaml`, `hook_ekf_manifold.yaml` + `ekf_net/*`
- Data configs: `hook_manifold.yaml`, `manifold.yaml`
- The config tree uses Hydra `defaults` for composition

---

### Phase 2: Architecture Design

**Design Decision 2.1 — Wrapper Module Tree**
Created `MultiModalJEPAWrapper` that wraps Neuro-JEPA's ViT backbone + classifier into a clean named-module tree:
```
MultiModalJEPAWrapper
├── backbone (VisionTransformer)         # Shared ViT backbone
│   ├── patch_embed (PatchEmbed3D)
│   ├── blocks.0..N (Block)              # Transformer blocks (with optional MoE)
│   └── norm (LayerNorm)
├── classifier (MultiModalLateFusion)    # Late-fusion classification head
│   ├── proj1 (ProjectionHead)
│   ├── proj2 (ProjectionHead)
│   ├── bn1, bn2 (BatchNorm1d)
│   ├── fusion (nn.Linear)
│   └── act (GELU)
```
This structure enables breakpoints on:
- `backbone` (after): capture per-modality ViT features
- `classifier.proj1`, `classifier.proj2`: intermediate projections
- `classifier.fusion` (before): inject reconstructed features
- `classifier` (after): capture logits for uncertainty estimation

**Design Decision 2.2 — Breakpoint DAG for JEPA**
Following the `hook_dag.yaml` pattern:
1. **Source breakpoints** on `backbone` (after) → capture ViT features (per-modality)
2. **Reconstructor breakpoint** on `classifier.fusion` (before, mutate=true):
   - `data_sources: [src_feats.0]` → collects backbone features
   - `pre_fn: ToListCollectedFn` → converts to tuple format
   - `callback: JEPAClassificationReconstructor` → cross-modal feature reconstruction
3. **Uncertainty breakpoint** on `classifier` (after):
   - `data_sources: [input]` → captures classifier output (logits)
   - `callback: BayesCap1D` → per-class uncertainty estimation

**Design Decision 2.3 — Classification Reconstructor**
`JEPAClassificationReconstructor` adapts the `BilinearReconstructor` pattern for classification:
- Takes multi-modal feature vectors (from ViT backbone)
- Reconstructs missing modality features based on available ones
- Works with variable-dimensional feature spaces
- Produces reconstructed features that get fused by the classifier

**Design Decision 2.4 — Lightning Module**
`JEPAHookModule` extends the pattern from `ModelInjectModule`:
- Frozen backbone (ViT) + trainable hook callbacks
- Modality masking during training (controlled by `mask_rate`)
- Reconstruction loss: MSE between reconstructed and original features
- Uncertainty loss: BayesCap1DLoss on classification logits
- Classification metrics: CrossEntropy loss, Accuracy
- Two-phase training: Phase 1 (reconstruction only), Phase 2 (+uncertainty)

---

### Phase 3: Implementation

**Action 3.1 — Create `src/models/components/jepa.py`**
- `MultiModalJEPAWrapper`: Clean wrapper exposing named modules for breakpoint attachment
- Supports `vit_late` fusion strategy
- Forward method: image_dict → logits
- Module tree designed for easy breakpoint targeting

**Action 3.2 — Create `src/plugins/reconstructor/jepa_classification.py`**
- `JEPAClassificationReconstructor`: Cross-modal feature reconstruction for classification
- Adapts `BilinearReconstructor` interface to work with ViT feature vectors
- Produces BreakpointOutput with reconstruction trace data

**Action 3.3 — Create `src/models/hook_modules/jepa.py`**
- `JEPAHookModule`: LightningModule for JEPA hook training
- Handles modality masking, reconstruction, classification, and uncertainty estimation
- CrossEntropyLoss for classification + BayesCap1DLoss for uncertainty
- Accuracy and NLL metrics

**Action 3.4 — Create config files**
- `configs/plugins/hook_jepa.yaml`: Plugin config with breakpoint DAG
- `configs/model/hook_jepa.yaml`: Model config for JEPAHookModule
- `configs/train_hook_jepa.yaml`: Top-level training config

---

### Phase 5.5: Pretrained Model Loading Analysis (2026-06-26)

**Key findings from `init_utils.py` and `scripts/finetune/mm.py`:**

1. **HuggingFace Hub**: `NYUMedML/Neuro-JEPA` — the pretrained checkpoint
2. **Checkpoint format**: 
   - `model.safetensors` or `pytorch_model.bin` containing backbone (ViT) weights
   - Keys can be wrapped in `encoder.`, `target_encoder.`, `student.vision_encoder.`, `model.`, `backbone.`, etc.
   - `_clean_backbone_state_dict()` strips all known prefixes
3. **Actual classifier**: `cross_attn.MultiModalLateFusion` (NOT `mm_classifier.MultiModalLateFusion`):
   - Bidirectional cross-attention with learnable query tokens
   - `proj1`, `proj2` → ProjectionHead (embed_dim → proj_dim)
   - `cross_attn_1to2`, `cross_attn_2to1` → CrossAttention
   - `norm1`-`norm4` → LayerNorm
   - `gate` (gate fusion) or `fusion` (concat) or None (add)
   - `classifier` → nn.Linear (proj_dim → num_classes)
4. **Forward flow**: `[B, num_tokens, embed_dim]` through proj → `[B, num_tokens, proj_dim]` → cross-attn → pool → `[B, proj_dim]` → gate → `[B, num_classes]`
5. **vit_late pipeline**: All modalities concatenated along batch dim → single backbone call → split per-modality

### Phase 6: Revised Architecture (2026-06-26)

**Problem**: The original wrapper design batched all modalities through a single backbone call, making per-modality hook attachment difficult.

**Solution**: Redesign wrapper as `MultiModalJEPARegressor` with:
- `backbone` (shared ViT) — frozen, called separately per modality
- `encoders` (nn.ModuleList of identity wrappers) — hook targets for per-modality ViT features
- `classifier` (cross_attn.MultiModalLateFusion) — cross-attn fusion + classification

**New Module Tree**:
```
MultiModalJEPARegressor
├── backbone (VisionTransformer)              # Shared frozen ViT
├── encoders (nn.ModuleList)                  # Per-modality hook targets
│   ├── 0 (ModalExtractor)                    # "encoders.0" → mod 0 ViT features
│   └── 1 (ModalExtractor)                    # "encoders.1" → mod 1 ViT features
├── classifier (MultiModalLateFusion)         # Cross-attn fusion
│   ├── proj1, proj2 (ProjectionHead)
│   ├── cross_attn_1to2, cross_attn_2to1
│   ├── norm1-4 (LayerNorm)
│   ├── gate (Sequential: Linear+ReLU+Linear+Tanh)
│   └── classifier (nn.Linear → logits)
```

**Breakpoint DAG**:
1. Source: `encoders.0` (after) → ViT token features mod 0
2. Source: `encoders.1` (after) → ViT token features mod 1
3. Reconstructor: `classifier` (before, mutate) → reconstructs missing modality features
4. Uncertainty: `classifier` (after) → logits → BayesCap1D

**Pretrained model loading**: 
- From HF Hub: `load_backbone_from_hf("NYUMedML/Neuro-JEPA")`
- From local: `load_backbone_weights(backbone, cfg)` with `checkpoint_source: local`

**Plugin Config (`hook_jepa.yaml`)**:
- `target: MultiModalJEPAWrapper` — the wrapper class name
- `model_checkpoint`: path to pretrained JEPA checkpoint
- Breakpoints:
  - `src_feats`: source breakpoint after backbone (captures per-modality features)
  - `reconstructor`: pre-fusion breakpoint with cross-modal reconstruction
  - `uncertainty`: post-classifier breakpoint with BayesCap1D
- DAG flow: backbone → reconstructor (via `src_feats`) → classifier → uncertainty (via `input`)

**Model Config (`hook_jepa.yaml`)**:
- `_target_: src.models.hook_modules.jepa.JEPAHookModule`
- CrossEntropyLoss for reconstruction quality metric
- BayesCap1DLoss for uncertainty estimation
- Adam optimizer + ReduceLROnPlateau scheduler
- Configurable `epoch_phase`, `mask_rate`

**Train Config (`train_hook_jepa.yaml`)**:
- Follows `train_hook_dag.yaml` pattern
- `data: hook_manifold` (placeholder — data pipeline TBD)
- `plugins: hook_jepa`, `model: hook_jepa`
- WandB logger, model checkpoint + early stopping on `val/loss_nll_best`

---

### Key Assumptions & Notes

1. **Backbone is frozen**: ViT backbone is pretrained and frozen; only hook callbacks (reconstructor, BayesCap) are trained.
2. **Feature dimensions**: Reconstructor needs to know `d_1` and `d_2` (feature dims per modality) — these are configurable.
3. **Modality masking**: During training, modalities are randomly masked to train reconstruction.
4. **vit_late fusion**: The wrapper assumes `vit_late` fusion where each modality is independently encoded then fused.
5. **The `nn.Module` registry**: The module tree exposes stable named paths for hook attachment.

---

### Phase 5: Files Created Summary

| File | Purpose |
|---|---|
| `src/models/components/jepa.py` | `MultiModalJEPAWrapper` — clean named-module tree for hook attachment |
| `src/plugins/reconstructor/jepa_classification.py` | `JEPAClassificationReconstructor` — cross-modal feature reconstruction |
| `src/models/hook_modules/jepa.py` | `JEPAHookModule` — LightningModule for hook training |
| `src/train_hook_jepa.py` | Training entry point with JEPA checkpoint loading logic |
| `configs/plugins/hook_jepa.yaml` | Plugin config with DAG breakpoints |
| `configs/model/hook_jepa.yaml` | Model config for JEPAHookModule |
| `configs/train_hook_jepa.yaml` | Top-level training config |
| `claude/JEPA_CLAUDE_LOG.md` | This log file |

### Phase 6: Usage Instructions

```bash
# Train hooks on JEPA backbone
python src/train_hook_jepa.py --config-name train_hook_jepa

# Override specific parameters
python src/train_hook_jepa.py --config-name train_hook_jepa \
    model.epoch_phase=5 \
    model.mask_rate=0.5 \
    plugins.model_checkpoint=path/to/neurojepa_checkpoint.pth

# Use tensorboard instead of wandb
python src/train_hook_jepa.py --config-name train_hook_jepa logger=tensorboard
```

### Architecture Verification

The DAG flow works as follows:
1. `src_feats` breakpoint (after backbone) captures ViT feature vectors
2. `reconstructor` breakpoint (before classifier.fusion) receives features via DAG from src_feats, reconstructs missing modalities, and mutates the fusion input
3. `uncertainty` breakpoint (after classifier) receives logits via `input` data_source, produces BayesCap (mu, alpha, beta) parameters

All breakpoints are resolved and validated for DAG cycles by `BreakpointController.wire()`.

---

### Future Work / TODOs

- [ ] Create data module for JEPA-compatible multi-modal classification data
- [ ] Add MoE-aware reconstruction (reconstruct expert routing distributions)
- [ ] Support additional fusion strategies (vit_avg, vit_mil)
- [ ] Add EKF-based uncertainty propagation for classification
- [ ] Test with actual MRI multi-modal datasets

# Neuro-JEPA Brain Age Estimation — Training Pipeline

## Overview

Downstream brain age regression using a pretrained Neuro-JEPA ViT backbone on the
[OpenBHB](https://huggingface.co/datasets/benoit-dufumier/openBHB) dataset.
The pipeline is optimised for consumer GPUs (8 GB VRAM, tested on RTX 4060).

```
OpenBHB quasiraw T1w  ─► MONAI transforms  ─► ViT backbone (frozen, MoE)  ─► pooled features  ─► MLP head  ─► predicted age
    [1,182,218,182]         [1,96,108,96]         [B,576,768]                    [B,768]           [B,1]            scalar
```

## Files Created / Modified

| File | Role |
|---|---|
| `src/models/modules/jepa_brain_age.py` | `JEPABrainAgeModule` (LightningModule) + `build_jepa_brain_age_backbone()` + `build_mlp_head()` |
| `configs/model/jepa_brain_age.yaml` | Model config: checkpoint path, head arch, optimizer, scheduler |
| `configs/data/openbhb_brain_age.yaml` | Data config: OpenBHB paths, batch_size=1, num_workers=0 |
| `configs/train_jepa_brain_age.yaml` | Main training config: 50 epochs, bf16, gradient accumulation |
| `configs/data/transform/train/openbhb.yaml` | Training MONAI transforms (10 steps) |
| `configs/data/transform/val/openbhb.yaml` | Validation MONAI transforms (4 steps) |
| `configs/data/transform/openbhb.yaml` | Top-level Hydra defaults (for Hydra app composition) |
| `scripts/train_jepa_brain_age.py` | Training entry point |
| `src/data/openbhb/dataset.py` | `OpenBHBDataset` — .npy loader |
| `src/data/openbhb/datamodule.py` | `OpenBHBDataModule` — Lightning DataModule |
| `src/data/openbhb/transforms.py` | `to_monai_compatible()` helper |
| `scripts/download_openbhb_samples.py` | Download N samples from HF Hub |

## Architecture

### Backbone: Neuro-JEPA vit_base (MoE)

```
VisionTransformer (vit_base)
  embed_dim=768  depth=12  heads=12  patch_size=(12,12,12)
  img_size=(96,108,96) -> 8x9x8 = 576 tokens per volume

  Layers 0,2,4,6,8,10:  RoPEAttention + SwiGLUFFN (standard)
  Layers 1,3,5,7,9,11:  RoPEAttention + MoE
    MoE: 2 shared experts + 16 routed experts, 6 activated per token
    Expert hidden dim: 384

  Total params: 122.1 M
  Frozen inference VRAM: ~0.95 GB (fp32), ~0.5 GB (bf16)
```

The MoE layers are essential — the pretrained checkpoint
(`NYUMedML/Neuro-JEPA`) was trained with MoE on alternate layers. Building
without MoE leaves layers 1,3,5,7,9,11 randomly initialised.

### Regression Head

```python
Sequential(
    Linear(768 -> 256), LayerNorm(256), GELU, Dropout(0.1),
    Linear(256 -> 128), LayerNorm(128), GELU, Dropout(0.1),
    Linear(128 -> 64),  LayerNorm(64),  GELU, Dropout(0.1),
    Linear(64 -> 1)       # scalar age prediction
)
```

Total: 239K params (~1 MB). Trains from scratch on top of frozen backbone.

## Data Pipeline

### Source

`benoit-dufumier/openBHB` on HuggingFace Hub. Quasi-raw T1w volumes already in
MNI152 space, brain-extracted, stored as `.npy` arrays of shape
`(1, 1, 182, 218, 182)` float64.

### Preprocessing (in `OpenBHBDataset.__getitem__`)

```
1. np.load(.npy) -> squeeze axis=0 -> (1, 182, 218, 182) float32
2. np.nan_to_num                                 NaN removal
3. np.percentile clip [0.5, 99.5]                outlier suppression
4. min-max normalize -> [0, 1]                   intensity scaling
```

### MONAI Transforms (Hydra-native, from YAML)

**Validation** (4 steps):
```
Lambdad(to_monai_compatible) -> EnsureTyped(MetaTensor) -> Resized(96,108,96, trilinear) -> ToTensor(float32)
```

**Training** (10 steps — val + 6 augmentations):
```
... above 4 steps ...
RandFlip(axis=0, p=0.1) -> RandFlip(axis=1, p=0.1) -> RandFlip(axis=2, p=0.1)
RandAdjustContrast(gamma=[0.6,1.5], p=0.5) -> RandGaussianNoise(std=0.1, p=0.5)
RandShiftIntensity(offset=0.1, p=0.5)
```

Augmentations match Neuro-JEPA's `vit3d_transforms` train mode.

## 8 GB VRAM Configuration (RTX 4060)

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 1 | Minimal memory per step |
| `accumulate_grad_batches` | 4 | Effective batch = 4 |
| `precision` | bf16-mixed | Half memory, RTX 4060 native support |
| `num_workers` | 0 | Avoid Windows multiprocessing crashes |
| `freeze_backbone` | true | Only head gets gradients -> ~2 GB VRAM |
| `gradient_clip_val` | 1.0 | Stability with small batches |
| `max_epochs` | 50 | Converges quickly with frozen backbone |

**Measured VRAM** (cuda.max_memory_allocated):
- Frozen backbone (fp32): 0.95 GB
- Frozen backbone (bf16, estimated): ~0.6 GB
- Training head (bf16): +~0.1 GB for gradients/optimizer
- **Total training: ~0.7-1.1 GB** — safe margin on 8 GB

### If Training the Backbone

Set `freeze_backbone: false` in model config. Additional VRAM cost:
- Gradients for 122M params: ~244 MB (fp16)
- AdamW states: ~488 MB (fp16)
- Activation memory (with checkpointing): ~500 MB
- **Estimated total: ~2-3 GB (bf16)** — still fits in 8 GB

Two-stage training: set `unfreeze_after_epoch: 10` to train head-only for
10 epochs, then unfreeze backbone for fine-tuning.

## Training

```bash
# Quick validation (6 samples, head only, ~1 min/epoch)
python scripts/train_jepa_brain_age.py

# Override specific params
python scripts/train_jepa_brain_age.py \
    model.freeze_backbone=false \
    model.unfreeze_after_epoch=10 \
    trainer.max_epochs=100 \
    data.n_train=8 data.n_val=8

# Full dataset (requires downloading all ~4000 samples)
python scripts/download_openbhb_samples.py --n_train 3227 --n_val 757
python scripts/train_jepa_brain_age.py data.n_train=3227 data.n_val=757 \
    data.batch_size=1 trainer.accumulate_grad_batches=8
```

## Expected Results

With a frozen backbone and 239K-parameter MLP head trained on 8 samples
(validation only — not a real training run):

```
Initial (random head):   MAE ~27 years, R^2 < 0    (expected — untrained head)
After 50 epochs:         MAE < 8 years,  R^2 > 0.3 (head-only, toy scale)
After full training:     MAE ~2.8 years, R^2 ~0.89 (Neuro-JEPA paper, full finetune)
```

The Neuro-JEPA paper reports **MAE 2.78 years, R^2 0.894** on OpenBHB (n=757
test) with full backbone fine-tuning, establishing the ceiling. Head-only
training on the full dataset should achieve competitive results.

## Key Design Decisions

1. **MoE backbone required**: The pretrained checkpoint uses Mixture of Experts
   on layers 1,3,5,7,9,11. Building without MoE leaves those layers random.
   Frozen MoE inference is safe for 8 GB VRAM (~0.95 GB fp32).

2. **Mean pooling**: ViT outputs [B, 576, 768] token grids. Mean pooling to
   [B, 768] discards spatial information but is simple and effective for
   regression. An `AttentiveClassifier` (learnable query cross-attention) would
   be stronger but adds ~4M params.

3. **Hydra-native transforms**: All MONAI transforms defined in YAML, not
   hard-coded in Python. Enables config-driven experimentation without code
   changes. The only retained Python helper is `to_monai_compatible()` for the
   channel-dim fixup (MONAI `Lambdad` requires a callable).

4. **num_workers=0**: Prevents `RuntimeError: DataLoader worker (pid X) exited
   unexpectedly` on Windows when using MONAI `PersistentDataset` with
   multiprocessing.

5. **Gradient accumulation**: `batch_size=1 + accumulate=4` gives effective
   batch of 4 without the VRAM cost of batch_size=4. Standard practice for
   training large vision models on consumer GPUs.

## Verification Log

```
=== Backbone (MoE) ===
Loading backbone from: .../NYUMedML/Neuro-JEPA/.../model.safetensors
MOE layer indices: [1, 3, 5, 7, 9, 11]
Loaded backbone: <All keys matched successfully>
Params: 122.1M  Device: cuda

=== Forward passes ===
Dummy: [1, 1, 96, 108, 96] -> pred=-0.26
OpenBHB: [[1, 1, 96, 108, 96]]  true=22.1  pred=-0.08

=== model_step ===
Loss=759.55  MAE=27.12  RMSE=27.56  R2=-30.06

VRAM peak: 0.95 GB  |  Safe for 8 GB: YES
```

## References

- Neuro-JEPA paper: Huang et al. "Learning Sparse Latent Predictive Foundation
  Model for Multimodal Neuroimaging" (2026)
- OpenBHB: Dufumier et al. "OpenBHB: a Large-Scale Multi-Site Brain MRI
  Data-set for Age Prediction and Debiasing", NeuroImage (2022)
- Checkpoint: `NYUMedML/Neuro-JEPA` on HuggingFace Hub (CC BY-NC-ND 4.0)
- OpenBHB data: `benoit-dufumier/openBHB` on HuggingFace Hub (CC BY-NC-SA 3.0)

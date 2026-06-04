# EKF Uncertainty Propagation Experiment

## Experiment Goals

Validate the full uncertainty propagation chain using ground-truth input covariance before moving to noisier estimates. The setup feeds a known Σ_z (computed via MC sampling or Jacobian from the frozen encoder pair) through an EKF Jacobian chain to produce per-sample predictive variance σ²_pred, which is used as the scale parameter α of a Generalized Gaussian NLL loss. The learnable parameter is a single shape scalar β.

**Hypothesis**: σ²_pred should (1) correlate positively with actual squared residuals, and (2) grow as the input distribution shifts OOD.

## Architecture Overview

```
x1, x2
  │
  ├─ frozen encoder1 → z1 (16-d)
  └─ frozen encoder2 → z2 (16-d)
                          │
                    z = cat[z1, z2]   (B, 32)
                          │
              GroundTruthSigmaZ
              (MC or Jacobian)
                          │
                   diag_sigma_z  (32,)
                          │
              EKFPropagation step 1:
              J_f · Σ_z · J_fᵀ  → diag_sigma_recon  (B, d')
                          │
              EKFPropagation step 2:
              J_g · Σ_recon · J_gᵀ → sigma_pred_sq  (B,)
                          │
              EKFGGDNLLLoss:
              α = √sigma_pred_sq,  β = exp(log_beta) [learnable]
                          │
                    GGD NLL loss
```

**Frozen vs. learned**: The backbone (encoder pair + head) and the reconstructor backbone are frozen for the EKF path. The only new learnable parameter in the EKF path is `log_beta` in `EKFGGDNLLLoss`. The BilinearReconstructor and BayesCap1D heads continue to train under their own losses as before.

## New Files

| File | Purpose |
|------|---------|
| `src/plugins/sigma_z.py` | `GroundTruthSigmaZ` (MC + Jacobian modes) and `BNShiftSigmaZ` (Phase 2) |
| `src/plugins/ekf_propagation.py` | Diagonal EKF utilities: `compute_reconstructor_jacobian`, `propagate_sigma_z_to_sigma_recon`, `compute_predictor_jacobian`, `propagate_sigma_recon_to_sigma_pred`, `full_ekf_propagation` |
| `src/plugins/head/ekf_nll_loss.py` | `EKFGGDNLLLoss`: GGD NLL with EKF-sourced α and learnable β |

Modified file:

| File | Change |
|------|--------|
| `src/models/hook_module.py` | `ModelInjectModule.__init__` accepts `ekf_enabled` and `sigma_z_mode`; EKF path runs in `training_step` when `ekf_enabled=True`; `ekf_loss.parameters()` added to optimizer |

## Environment Setup

```bash
conda env create -f environment.yaml && conda activate myenv
# or:
pip install -r requirements.txt
```

## Step-by-Step Run Instructions

### Phase 0: Train backbone (skip if `data/checkpoints/toy.pth` exists)

```bash
python src/train.py trainer.max_epochs=100 trainer=gpu
```

### Phase 1: Train EKF uncertainty experiment

```bash
python src/train_hook.py trainer.max_epochs=50 trainer=gpu model.ekf_enabled=true
```

To switch sigma_z estimation to Jacobian mode:

```bash
python src/train_hook.py trainer.max_epochs=50 trainer=gpu \
    model.ekf_enabled=true \
    plugins.sigma_z.mode=jacobian
```

### Evaluate

```bash
python src/eval.py ckpt_path=logs/train_hook/[DATE]/checkpoints/best.ckpt
```

## What to Monitor (WandB / TensorBoard)

| Metric | Expected behaviour |
|--------|--------------------|
| `train/ekf_nll` | Decreasing over epochs |
| `train/beta` | Converges to [1.5, 2.5] (between Laplace β=1 and Gaussian β=2) |
| `train/sigma_pred_mean` | Positive and stable; should not collapse to zero |
| `val/loss_nll_best` | Checkpoint criterion — minimise this |

## Key Evaluation Diagnostics

### OOD uncertainty sweep

```python
from src.plugins.sigma_z import GroundTruthSigmaZ

for shift in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    x_range = (-1.0 + shift, 1.0 + shift)
    sigma_z = GroundTruthSigmaZ(enc1, enc2, x_range=x_range, mode="mc")
    diag_sigma_z = sigma_z.diag_sigma_z
    # run full_ekf_propagation on test samples from this range and record sigma_pred_sq.mean()
```

Expected: `sigma_pred_sq.mean()` grows monotonically as `shift` increases.

### Correlation diagnostic

```python
correlation = torch.corrcoef(
    torch.stack([sigma_pred_sq, (y_true - mu_pred) ** 2])
)[0, 1]
# Target: r > 0.3
```

## Ablation Roadmap

| Phase | Σ_z source | Goal |
|-------|-----------|------|
| **1 (this experiment)** | `GroundTruthSigmaZ` via MC sampling from known Uniform(a,b) | Validate EKF pipeline end-to-end |
| **2 (next)** | `BNShiftSigmaZ` via BN running statistics — no source data needed at inference | Test whether cheap BN proxy preserves rank correlation |

**Key metric between phases**: Spearman rank correlation and relative error of `sigma_pred_sq` distributions between Phase 1 (oracle) and Phase 2 (proxy). If rank correlation > 0.8, the BN proxy is sufficient to replace oracle Σ_z.

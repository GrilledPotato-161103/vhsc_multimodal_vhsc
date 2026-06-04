# SURE Project — Claude Context Export
> Paste this file into a new Claude Code session on the remote server to restore full project context.
> Generated: 2026-05-26

---

## Who I Am

AI researcher focused on LLMs, VLMs, and multimodal learning. Primary focus: medical VLMs, uncertainty estimation, distribution shift. Work scope only. Prefer concise, technical responses.

---

## Project: SURE — Stochastic Uncertainty under Representation Shift

**Core problem**: A model M_A is pretrained on source domain A, then deployed on target domain B (distribution shift + possibly missing modalities). Goal: post-hoc uncertainty estimation — convert deterministic predictions into calibrated uncertainty without retraining M_A.

**Key bottleneck**: modeling input covariance Σ_z caused by distribution shift. Once Σ_z is formalized, reconstruction uncertainty and prediction uncertainty follow by EKF propagation.

**Ablation roadmap**:
- **Phase 1 (current)**: Use ground-truth Σ_z (known because toy dataset has known distribution) → validate the full pipeline works
- **Phase 2 (next)**: Replace ground-truth Σ_z with BN-shift-score estimate (no source data needed) → compare against Phase 1 → this gap = the paper's main contribution

---

## Codebase Overview

**Framework**: PyTorch Lightning + Hydra. Toy synthetic dataset: y = (x1² + 2·x1·x2 + 2·x2)/5, x1,x2 ~ Uniform[-1,1], 256k samples.

**Git branch for experiment**: `feature/ekf-uncertainty-experiment`

### Architecture

```
BiModalRegressor (frozen backbone):
  x1 → MLP(1→16→16→16) → z1
  x2 → MLP(1→16→16→16) → z2
  concat([z1, z2]) → MLP(32→32→16→16→1) → y_hat

BilinearReconstructor (trainable plugin, hooks into backbone):
  ln12: FeedForward(16→24→16)  -- reconstruct z2 from z1
  ln21: FeedForward(16→24→16)  -- reconstruct z1 from z2

EKF chain (NEW — implemented on feature branch):
  diag_sigma_z (32,) → J_f → diag_sigma_recon (32,) → J_g → sigma_pred_sq (B,)

EKFGGDNLLLoss:
  alpha = sqrt(sigma_pred_sq)   -- from EKF, NOT from neural head
  beta  = exp(log_beta)         -- single learnable scalar
  NLL   = (|y - mu| / alpha)^beta + log(alpha) + lgamma(1/beta) - log(beta)
```

### Key source files

| File | Purpose |
|---|---|
| `src/train.py` | Train backbone (Phase 1) |
| `src/train_hook.py` | Train plugins on frozen backbone (Phase 2) |
| `src/eval.py` | Evaluate checkpoint |
| `src/models/components/toy.py` | BiModalRegressor |
| `src/models/hook_module.py` | ModelInjectModule — **modified for EKF** |
| `src/plugins/hook.py` | Breakpoint + BreakpointController |
| `src/plugins/head/bayescap.py` | Original BayesCap1D (kept for baseline) |
| `src/plugins/reconstructor/linear.py` | BilinearReconstructor |
| `src/plugins/sigma_z.py` | **NEW** — GroundTruthSigmaZ + BNShiftSigmaZ |
| `src/plugins/ekf_propagation.py` | **NEW** — diagonal EKF Jacobian utilities |
| `src/plugins/head/ekf_nll_loss.py` | **NEW** — EKFGGDNLLLoss |
| `configs/plugins/toy.yaml` | Plugin config — **has sigma_z + ekf + nll blocks** |
| `configs/train_hook.yaml` | Training config — **has ekf_enabled + sigma_z_mode** |

---

## What Was Implemented (on feature branch)

### 1. `src/plugins/sigma_z.py`

```python
class GroundTruthSigmaZ:
    # MC sampling: forward N uniform samples through frozen encoders → empirical var
    # Returns: diag_sigma_z (32,) — diagonal input variance vector
    # mode="mc" (default) or mode="jacobian" (analytical at input mean)

class BNShiftSigmaZ:
    # Per-sample BN shift score using backbone's running_mean/running_var
    # compute_shift_score(z_activations) → (B,) scalar shift score per sample
    # get_sigma_z(shift_score, d=32) → (B, 32) isotropic s(z)·I
    # Phase 2 implementation — not yet used in training loop
```

### 2. `src/plugins/ekf_propagation.py`

```python
# Key functions:
make_reconstructor_fn(reconstructor, signal)  # → pure fn (32,) → (32,) for vmap/jacrev
make_predictor_fn(head)                        # → fn (B, d') → (B,)

compute_reconstructor_jacobian(fn, z)          # → (B, 32, 32) via vmap(jacrev)
propagate_sigma_z_to_sigma_recon(J_f, diag_sigma_z)  # → (B, 32): (J_f²) @ diag_sigma_z
compute_predictor_jacobian(fn, z_recon)        # → (B, 32) gradient via autograd
propagate_sigma_recon_to_sigma_pred(J_g, diag_sigma_recon)  # → (B,): (J_g² * Σ_recon).sum

full_ekf_propagation(z, diag_sigma_z, reconstructor_fn, predictor_fn)
    # → (sigma_pred_sq (B,), diag_sigma_recon (B,32), J_f (B,32,32))
```

**EKF formulas**:
- Step 1: `diag(Σ_recon)_i = Σ_k J_f[i,k]² · σ²_z[k]`
- Step 2: `σ²_pred = Σ_i J_g[i]² · diag(Σ_recon)_i`

### 3. `src/plugins/head/ekf_nll_loss.py`

```python
class EKFGGDNLLLoss(nn.Module):
    # log_beta: nn.Parameter (single learnable scalar, init=0.5 → beta≈1.65)
    # forward(y_true, mu_pred, sigma_pred_sq) → scalar NLL
    # alpha = sqrt(sigma_pred_sq + eps)  -- from EKF, not from a neural head
```

### 4. `src/models/hook_module.py` (modified)

`ModelInjectModule.__init__` now accepts `ekf_enabled: bool = False` and `sigma_z_mode: str = "mc"`.

When `ekf_enabled=True`:
- Builds `GroundTruthSigmaZ` from `net.net.x1_encoder` / `net.net.x2_encoder`
- Registers `diag_sigma_z` as a buffer (computed once at init)
- Instantiates `EKFGGDNLLLoss` as `self.ekf_loss`
- In `training_step`: extracts z = cat([z1,z2]) from frozen encoders, calls `full_ekf_propagation`, computes `ekf_nll`, adds to loss
- Logs: `train/ekf_nll`, `train/sigma_pred_mean`, `train/beta`
- `configure_optimizers`: includes `self.ekf_loss.parameters()` (trains β)

### 5. Config additions

`configs/plugins/toy.yaml` — appended:
```yaml
sigma_z:
  mode: "mc"
  n_mc: 5000
  x_range_train: [-1.0, 1.0]
  x_range_ood: [-0.8, 1.2]

ekf:
  enabled: true
  jacobian_mode: "autograd"
  diagonal: true
  sigma_net_recon: 0.0
  sigma_net_pred: 0.0

nll:
  learn_calibration: false
  beta_init: 0.5
  eps: 1.0e-8
```

`configs/train_hook.yaml` — added under `model:`:
```yaml
  sigma_z_mode: ${plugins.sigma_z.mode}
  ekf_enabled: ${plugins.ekf.enabled}
```

---

## How to Run

```bash
# Setup
conda env create -f environment.yaml && conda activate myenv
# or: pip install -r requirements.txt

# Phase 0: Train backbone (generates data/checkpoints/toy.pth)
python src/train.py trainer.max_epochs=100 trainer=gpu

# Phase 1: EKF experiment (main task)
python src/train_hook.py trainer.max_epochs=50 trainer=gpu model.ekf_enabled=true

# Evaluate
python src/eval.py ckpt_path=logs/train_hook/[DATE]/checkpoints/best.ckpt

# Baseline (original BayesCap, no EKF)
python src/train_hook.py trainer.max_epochs=50 trainer=gpu model.ekf_enabled=false
```

---

## What to Monitor

| Metric | Expected behavior |
|---|---|
| `train/ekf_nll` | Decreasing |
| `train/beta` | Converges to [1.5, 2.5] |
| `train/sigma_pred_mean` | Positive, stable, non-zero |
| `val/loss_nll_best` | Checkpoint criterion |

**Key verification test** — OOD uncertainty growth:
```python
for shift in [0.0, 0.1, 0.2, 0.5, 1.0]:
    x_range = (-1.0 + shift, 1.0 + shift)
    # evaluate sigma_pred_sq on test samples from this shifted range
    # Expected: sigma_pred_sq grows with shift
```

**Correlation diagnostic**:
```python
corr = corrcoef(sigma_pred_sq, (y_true - mu_pred)**2)[0,1]
# Target: r > 0.3 (EKF uncertainty predicts actual errors)
```

---

## Research Context: The Bigger Picture

### Three settings (publication strategy)

| Setting | Σ_z source | A data needed? | Novelty |
|---|---|---|---|
| **SD** (source-dependent) | Mahalanobis from A's class-conditional Gaussian | Yes (A features) | Strong math, clean |
| **SF** (source-free) | Flows / SWAG from M_A's weights | No — but needs A training pipeline | Actually most restrictive operationally |
| **B-only** (target) | BN shift score s(z)·I or K-means on B | No A data at all | Most practical, understudied |

**Publication target**: Lead with B-only (NeurIPS/ICML). SD as oracle upper bound. SF as ablation.

### B-only method (Phase 2, not yet implemented in training)

BN shift score (implemented in `BNShiftSigmaZ` but not wired):
```
s(z) = (1/L) Σ_l (1/d_l) Σ_c (z^l_c − μ^l_A)² / σ²^l_A
Σ_z = s(z) · I   (isotropic, per-sample)
```

Uses `running_mean` and `running_var` from backbone's BatchNorm layers — zero additional data needed.

**Phase 2 task**: swap `GroundTruthSigmaZ` → `BNShiftSigmaZ`, wire `compute_shift_score` using intermediate activations hooked from the backbone, feed `sigma_z = BNShiftSigmaZ.get_sigma_z(shift_score)` as per-sample (B,32) instead of shared (32,) to `ekf_propagation`.

**Key metric for publication**: rank correlation between `sigma_pred_sq` from Phase 1 (groundtruth) vs Phase 2 (BN estimate). High rank correlation → BN shift score preserves uncertainty ordering → contribution is valid.

---

## Known Issues / Open Questions

1. `make_reconstructor_fn` uses `unsqueeze(0)/squeeze(0)` for FeedForward layers — verify this handles batch dims correctly under `vmap`
2. `compute_predictor_jacobian` uses `create_graph=True` which adds overhead — consider `create_graph=False` if not doing second-order optimization
3. `full_ekf_propagation` calls `vmap(reconstructor_fn)(z)` inside `torch.no_grad()` then separately computes Jacobian — confirm no double-forward issue
4. `mask_rate=0.0` in current config — missing modality training not yet enabled; for realistic experiments, set `mask_rate=0.3`
5. Backbone checkpoint path: `data/checkpoints/toy.pth` — must exist before Phase 2 training

---

## Conventions

- Write concise, technical notes — no padding
- All written output (abstracts, paper text) should follow ICLR/ICML oral style: direct subject-verb, no AI-tell phrases ("leverage", "delve into", "comprehensive")
- When editing paper text, run `/humanizer` afterward


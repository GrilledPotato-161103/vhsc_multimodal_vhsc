# VHSC Multimodal — Claude Context Export

> Comprehensive project context for the VHSC (Visual-Haptic-Sensory-Cognitive) multimodal uncertainty estimation framework.
> Generated: 2026-06-18 | Branch: `main`

---

## Project Identity

**Name**: SURE — Stochastic Uncertainty under Representation Shift
**Framework**: PyTorch Lightning + Hydra (Lightning-Hydra-Template fork)
**Domain**: Multimodal learning, uncertainty estimation, distribution shift detection
**Core problem**: Post-hoc uncertainty calibration for frozen pretrained models under domain shift with potentially missing modalities.

---

## Architecture Overview

### High-Level Pipeline

```
Backbone (BiModalRegressor, frozen):
  x1 → MLP(1→16→16→16) → z1 ─┐
  x2 → MLP(1→16→16→16) → z2 ─┤
                               ├→ concat([z1,z2]) → MLP(32→32→16→16→1) → ŷ
Reconstructor (BilinearReconstructor, trainable):
  ln12: FeedForward(16→24→16)   — reconstruct ẑ2 from z1
  ln21: FeedForward(16→24→16)   — reconstruct ẑ1 from z2

EKF Chain (uncertainty propagation):
  Σ_z (input cov) → J_f (reconstructor Jacobian) → Σ_recon → J_g (predictor Jacobian) → σ²_pred

Uncertainty Head (EKFBiModalInferer / BayesCap1D / HessianBiModalInferer):
  Takes (z, Σ_z, ŷ) → (μ, α, β, σ²_pred)
  NLL: |y - μ|^β / α^β + log(α) + lgamma(1/β) − log(β)
```

### Module Dependency Map

```
src/
├── train.py              # Backbone training entry point
├── train_hook.py          # Hook/EKF training entry point (custom pipeline)
├── eval.py               # Evaluation entry point
├── models/
│   ├── components/
│   │   ├── ffn.py         # FeedForward network + activation/norm factories
│   │   ├── toy.py         # BiModalRegressor, MLP, Residual blocks
│   │   └── mmbt.py        # MMBT multimodal transformer
│   ├── hook_modules/
│   │   ├── ekf.py         # ModelEKFInjectModule (LightningModule, manual opt)
│   │   ├── ekf_manifold.py # ModelEKFManifoldModule (automatic opt variant)
│   │   └── toy.py         # ModelInjectModule (BayesCap-only, no EKF)
│   ├── modules/
│   │   ├── toy_module.py  # BiModalLightningModule (clean wrapper for BiModalRegressor)
│   │   ├── manifold_module.py # ManifoldLightningModule (multi-modal variant)
│   │   ├── mmbt.py        # MMBTLitModule (LightningModule wrapper for MMBT)
│   │   └── mnist_module.py # MNISTLitModule (example template)
│   ├── losses/
│   │   └── nce.py         # NCELoss, GaussianAlignLoss, WeightedCrossEntropy, etc.
│   └── SURE/              # Legacy multimodal models (import paths broken)
│       ├── gmc.py, hamlet.py, mmbt.py, vilt.py, mmml.py
│       ├── modules/       # GMC, HAMLET, ViLT, MMBT, ContextModel modules
│       ├── losses/        # NCE loss (legacy copy)
│       └── trainers/      # DCA evaluation, metrics
├── plugins/
│   ├── hook.py            # Breakpoint/BreakpointController system
│   ├── var.py             # Data structures: BreakpointContext, BreakpointOutput
│   ├── ekf_propagation.py  # EKF diagonal + full covariance propagation
│   ├── sigma_z.py          # SDSigmaZ, GroundTruthSigmaZ, BNShiftSigmaZ
│   ├── aggregate.py        # Router/EndpointWrapper (GStreamer-inspired)
│   ├── fx_visualizer.py    # torch.fx graph visualizer
│   ├── head/
│   │   ├── ekf.py          # EKFBiModalInferer (GGD head with EKF alpha)
│   │   ├── bayescap.py     # BayesCap1D + BayesCap1DLoss + EKFGGDNLLLoss
│   │   └── hessian.py      # HessianBiModalInferer + DiscreteHessianBiModalInferer
│   └── reconstructor/
│       ├── linear.py       # BilinearReconstructor + LinearReconstructor
│       └── identity.py     # IdentityHook (no-op passthrough)
├── data/
│   ├── components/
│   │   ├── dataset.py      # ToyDataset, ManifoldToyDataset
│   │   └── sampler.py      # SortedBatchSampler
│   ├── toy_datamodule.py   # ToyBiModalDataModule
│   └── manifold_datamodule.py # ManifoldDataModule
└── utils/
    ├── instantiators.py    # Hydra callback/logger instantiation
    ├── logging_utils.py    # Hyperparameter logging
    ├── pylogger.py         # RankedLogger (multi-GPU log adapter)
    ├── rich_utils.py       # Rich config tree + tag enforcement
    ├── utils.py            # extras(), task_wrapper(), get_metric_value()
    └── callbacks/
        ├── hook_callback.py  # AdversarialVizCallback (uncertainty heatmaps)
        └── utils.py          # RunTestEveryNEpochs
```

---

## Key Source Files — Detailed Reference

### Entry Points

| File | Purpose | Config | Key Difference |
|------|---------|--------|----------------|
| `src/train.py` | Train backbone from scratch | `configs/train.yaml` | Standard Lightning pipeline |
| `src/train_hook.py` | Train plugins on frozen backbone | `configs/train_ekf_hook.yaml` | Loads checkpoint, wraps in BreakpointController, passes controller to model |
| `src/eval.py` | Evaluate a trained checkpoint | `configs/eval.yaml` | Hardcoded checkpoint path (`data/checkpoints/toy.pth`) |

### Hook Module Comparison (3 variants)

| File | Optimization | Optimizers | EKF | Key Difference |
|------|-------------|-----------|-----|----------------|
| `hook_modules/ekf.py` | Manual | 2 (recon + EKF) | Yes (full) | Phased training, per-signal metrics, full EKF propagation |
| `hook_modules/ekf_manifold.py` | Automatic | 1 (unified) | Yes (diag) | Simpler metrics, diag EKF, SDSigmaZ provider |
| `hook_modules/toy.py` | Automatic | 1 | No | BayesCap-only uncertainty, simplest variant |

### Sigma-Z Providers (3 strategies)

| Class | File | Input Cov Source | Requires Source Data? |
|-------|------|-----------------|----------------------|
| `GroundTruthSigmaZ` | `sigma_z.py` | Monte Carlo / analytical Jacobian from known distribution | Yes (distribution params) |
| `SDSigmaZ` | `sigma_z.py` | Mahalanobis-scaled source covariance | Yes (source latents) |
| `BNShiftSigmaZ` | `sigma_z.py` | BatchNorm shift score (per-sample) | No |

### Uncertainty Heads (4 variants)

| Class | File | Alpha Source | Beta Source | Key Feature |
|-------|------|-------------|-------------|-------------|
| `BayesCap1D` | `bayescap.py` | Neural net | Neural net | Full BayesCap, mu/alpha/beta heads |
| `EKFGGDNLLLoss` | `bayescap.py` | EKF `sqrt(sigma_pred_sq)` | Single scalar param | No neural head for alpha |
| `EKFBiModalInferer` | `ekf.py` | EKF sigma_pred | Neural net | Combines EKF alpha with learned beta |
| `HessianBiModalInferer` | `hessian.py` | Hessian curvature | Neural net | Exact 2nd-order propagation |
| `DiscreteHessianBiModalInferer` | `hessian.py` | Finite-diff approx | Neural net | Efficient Hessian approximation |

---

## Configuration System

### Hydra Config Tree

```
configs/
├── train.yaml              # Backbone training (data=toy, model=toy)
├── train_ekf_hook.yaml      # EKF hook training (data=hook, model=hook_ekf_recon, plugins=ekf)
├── train_hook.yaml          # Basic hook training (data=hook_manifold, model=hook, plugins=toy)
├── eval.yaml               # Evaluation config
├── model/
│   ├── toy.yaml             # BiModalLightningModule
│   ├── hook.yaml            # ModelInjectModule (toy BayesCap)
│   ├── hook_ekf.yaml        # ModelEKFInjectModule
│   ├── hook_ekf_manifold.yaml # ModelEKFManifoldModule
│   ├── hook_ekf_recon.yaml  # ModelEKFInjectModule (mask_rate=0.3, epoch_phase=30)
│   ├── manifold.yaml        # ManifoldLightningModule
│   └── ekf_net/             # EKF network sub-configs
│       ├── default.yaml     # EKFBiModalInferer
│       ├── discrete.yaml    # DiscreteHessianBiModalInferer
│       └── hessian.yaml     # HessianBiModalInferer
├── plugins/
│   ├── toy.yaml             # Breakpoint config for toy (BilinearReconstructor + BayesCap1D)
│   └── ekf.yaml             # EKF plugin config (sigma_z + ekf + nll)
├── data/
│   ├── toy.yaml             # ToyBiModalDataModule
│   ├── manifold.yaml        # ManifoldDataModule
│   ├── hook.yaml            # Hook data config
│   └── hook_manifold.yaml   # Hook manifold data config
├── callbacks/
│   ├── hook.yaml            # Hook training callbacks
│   ├── plot.yaml            # Plotting callbacks
│   └── ...
├── trainer/                 # GPU/CPU/DPP trainer configs
├── logger/                  # WandB/Neptune/MLflow/CSV logger configs
├── debug/                   # Debug profiles (overfit, fdr, profiler, limit)
└── paths/default.yaml       # Root/data/log/output paths
```

### Config Inheritance Pattern

Hydra defaults compose configs bottom-up:
```
train_ekf_hook.yaml
  ├── data: hook
  ├── model: hook_ekf_recon
  │     └── ekf_net: default
  ├── plugins: ekf
  ├── callbacks: hook
  ├── trainer: default
  ├── logger: wandb
  └── paths: default
```

---

## Training Modes

### Phase 1: Train Backbone
```bash
python src/train.py trainer.max_epochs=100 trainer=gpu
# Trains BiModalRegressor from scratch on toy data
# Saves checkpoint to data/checkpoints/
```

### Phase 2a: Basic Hook Training (BayesCap baseline)
```bash
python src/train_hook.py trainer.max_epochs=50 trainer=gpu
# Loads frozen backbone, trains BilinearReconstructor + BayesCap1D
# Config: train_hook.yaml (no EKF)
```

### Phase 2b: EKF Hook Training (main experiment)
```bash
python src/train_hook.py --config-name train_ekf_hook trainer.max_epochs=50 trainer=gpu
# Loads frozen backbone, trains reconstructor + EKF uncertainty head
# Two-phase: phase 1 trains reconstruction only, phase 2 adds uncertainty
```

### Evaluation
```bash
python src/eval.py ckpt_path=logs/train/runs/[TIMESTAMP]/checkpoints/last.ckpt
```

---

## Breakpoint/Hook System

The `Breakpoint` / `BreakpointController` system in `src/plugins/hook.py` is a custom PyTorch hook framework:

1. **`Breakpoint`** — an `nn.Module` that wraps a callback (`run_before` / `run_after`), stores execution traces.
2. **`BreakpointController`** — manages attaching/detaching breakpoints to model layers via `register_forward_hook` / `register_forward_pre_hook`.
3. **`BreakpointContext`** — runtime context passed to breakpoint callbacks (inputs, outputs, layer info).
4. **`BreakpointOutput`** — callback return value (output, trace, mutation flag).

**Key classes defined in `src/plugins/var.py`**: `BreakpointConfig`, `BreakpointContext`, `BreakpointOutput`.

---

## Known Issues — Prioritized Fix List

### Critical Bugs

1. **Nested kwargs bug** (`ekf_manifold.py`, `toy.py`): `self.model_step(batch, kwargs={"bp_signal": (1,1)})` passes a nested `kwargs` dict instead of unpacking `bp_signal`. The check `"bp_signal" in kwargs.keys()` always fails → random masking always runs.

2. **Undefined variable in `mmbt.py` line 128**: `tqdm_dict` is only assigned inside the `if self.dataset == 'book'` branch; the `else` branch references it, causing `NameError`.

3. **Infinite busy-wait in `linear.py` line 163**: `while not self.is_ready(): pass` hangs the process indefinitely.

4. **Wrong dataset sizes in `manifold_datamodule.py`**: `val_dataset` and `test_dataset` both use `n_samples=n_train` instead of `n_val` and `n_test`.

5. **`gather()` references nonexistent attribute in `hook.py`**: `breakpoint["breakpoint"].output` should be `breakpoint["breakpoint"].trace`.

6. **`run_before` mutation logic bug in `hook.py`**: Result assignment checks `result.output` but the return always passes through original `(inputs, kwargs)` regardless of the mutation flag.

7. **`scripts/run.sh` uses wrong Hydra flag**: `config-map=` instead of `--config-name`.

### High-Priority Code Quality Issues

8. **Triple-duplicated `HuberLoss`**: Identical class in `ekf.py`, `ekf_manifold.py`, and `toy.py` (also in `linear.py`).

9. **Triple-duplicated `check_gradient`**: Identical function in all three `hook_modules/` files.

10. **Duplicated `get_normalization` / `get_activation`**: `ffn.py` and `toy.py` independently implement near-identical factory functions.

11. **~80% code duplication between `train.py`, `train_hook.py`, `eval.py`**: Shared Hydra boilerplate interleaved with variant-specific logic.

12. **~95% code duplication between `toy_module.py` and `manifold_module.py`**: Only differ in `forward` signature and batch unpacking.

13. **Massive duplication between `HessianBiModalInferer` and `DiscreteHessianBiModalInferer`** (~80% shared `__init__`).

### Medium-Priority Quality Issues

14. **Hardcoded checkpoint path in `eval.py`**: `data/checkpoints/toy.pth` — should be config-driven.
15. **Hardcoded `.cuda()` calls in `train_hook.py`** lines 69, 73 — not portable.
16. **Debug `print()` calls**: `train_hook.py:74`, `ffn.py:127`, `bayescap.py` init, `dataset.py:152`.
17. **`_CUSTOM_CONFIG` unused variable** in `train_hook.py`.
18. **Missing `__init__.py` in `SURE/` and subdirectories**.
19. **Import paths broken for 4 of 5 SURE model files** — `gmc.py`, `hamlet.py`, `vilt.py`, `mmml.py` import from `models.modules.*` instead of `models.SURE.modules.*`.
20. **Missing dependencies in `environment.yaml`**: `transformers`, `scipy`, `plotly`, `wandb`.
21. **Outdated README**: Missing SURE structure, duplicate commands, no setup instructions.

### Low-Priority Cosmetic Issues

22. **Mixed Vietnamese/English comments** (`aggregate.py`, `dataset.py`, `ekf_propagation.py`, `callbacks/utils.py`).
23. **`list_breakpoints()` defined twice** in `hook.py`.
24. **Inconsistent `num_groups` default**: 2 in `ffn.py`, 3 in `toy.py`.
25. **Spelling**: `txt_reprensentation` throughout `mmbt.py`.
26. **In-place tensor mutation**: `full_ekf_propagation` calls `z.unsqueeze_(1)`.
27. **`Node` class** in `aggregate.py` is empty (dead code).
28. **Bare `except: pass`** in `mmbt.py` line 113.
29. **`IPython.embed()` blocks** in `nce.py` (5 instances).
30. **`EndpointWrapper.forward`**: passes tuple as single arg instead of `*args`.

---

## How to Run (Complete)

```bash
# Setup
conda env create -f environment.yaml && conda activate myenv
pip install transformers scipy plotly wandb  # missing from environment.yaml

# Phase 0: Train backbone
python src/train.py trainer.max_epochs=100 trainer=gpu

# Phase 1: EKF experiment (main)
python src/train_hook.py --config-name train_ekf_hook trainer.max_epochs=50 trainer=gpu

# Phase 2: Baseline (BayesCap only)
python src/train_hook.py --config-name train_hook trainer.max_epochs=100 trainer=gpu

# Evaluate
python src/eval.py ckpt_path=logs/train/runs/[TIMESTAMP]/checkpoints/last.ckpt
```

---

## Key Metrics to Monitor

| Metric | Meaning | Expected |
|--------|---------|----------|
| `train/loss_nll` | GGD negative log-likelihood | Decreasing |
| `train/beta` | GGD shape parameter | Converges to [1.5, 2.5] |
| `train/sigma_pred_mean` | Mean predictive uncertainty | Positive, stable |
| `val/loss_nll_best` | Validation NLL (checkpoint criterion) | Decreasing |
| `val/loss_unc_pcc` | Pearson r between loss and uncertainty | > 0.3 target |

---

## Research Context

### Three Input Covariance Settings

| Setting | Σ_z Source | Source Data Needed? | Status |
|---------|-----------|-------------------|--------|
| **SD** (source-dependent) | Mahalanobis from source class-conditional Gaussian | Yes (A features) | Implemented (`SDSigmaZ`) |
| **SF** (source-free) | Flows / SWAG from pretrained weights | No (but needs training pipeline) | Not implemented |
| **B-only** (target-only) | BN shift score s(z)·I or K-means on target | No data at all | Implemented (`BNShiftSigmaZ`), not wired into training |

### OOD Verification Test
```python
for shift in [0.0, 0.1, 0.2, 0.5, 1.0]:
    x_range = (-1.0 + shift, 1.0 + shift)
    # evaluate sigma_pred_sq on test samples from shifted range
    # Expected: sigma_pred_sq grows with distribution shift magnitude
```

---

## Conventions

- **Code style**: Match existing patterns. Use `black`-compatible formatting. Snake_case for functions, PascalCase for classes.
- **Comments**: English only for new code. No `print()` debugging — use `log.info()` / `log.debug()`.
- **Config**: All paths and hyperparameters must be Hydra-configurable. No hardcoded paths.
- **Device management**: Use `torch.device` / `self.device`, never `.cuda()` directly.
- **Documentation**: Update this file when adding new modules or changing architecture.
- **Git**: Branch for experiments, PR for review. Commit messages in imperative mood.

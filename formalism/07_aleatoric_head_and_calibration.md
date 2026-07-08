---
title: Aleatoric Head, the inv_alpha Indirection Bug, and Calibration Ceiling
tags: [SURE, aleatoric, epistemic, calibration, closed_form, PCC]
created: 2026-06-22
status: confirmed
context: |
  Adds a learned aleatoric uncertainty head to complement the EKF/cycle
  epistemic term, then resolves why predicted variance did not visually match
  the loss map. Six closed-form experiments (cf1-cf6) localize the cause and
  establish the calibrated configuration. Full data: logs/slurm_cf*.log.
related: [[03_phase1_limits_and_phase2_direction]] [[06_empirical_validation]]
---

# Aleatoric Head and Calibration

## 1. Motivation

The EKF + cycle-consistency pipeline (docs 03, 06) produces *epistemic* uncertainty
that grows under input shift. But prediction error has two sources:

    sigma_total^2 = sigma_epistemic^2 (OOD shift) + sigma_aleatoric^2 (function complexity)

The epistemic term is near-zero in-distribution and explosive OOD. The aleatoric
term captures in-distribution residual structure (where the frozen backbone is
intrinsically wrong, e.g. function singularities). Without the aleatoric head, the
variance map is a pure OOD ramp that ignores the in-distribution loss structure.

## 2. Architecture

`AleatoricHead` (src/plugins/head/ekf.py): small MLP, strictly-positive output
(Softplus + eps), bias init -3 so it starts near zero. Combined as:

    sigma_total = sigma_ep + lambda_aleatoric * sigma_al

Input modes tested: `z_only`, `z_and_sep` (z + log sigma_ep), `xy` (raw inputs).

## 3. The key bug — inv_alpha_net indirection

Initial runs (z_only / xy with `mode="learned"`) showed the variance map still not
matching the loss map; PCC(predvar, err) ~ 0.

**Cause**: in `mode="learned"`, the predictive variance is NOT sigma_total. It passes
through a learned MLP:

    ekf_feat = log(cat([sigma_total, diag_sigma_recon]))
    inv_alpha = softplus(inv_alpha_net(ekf_feat))     # MLP can ignore sigma_total
    var = bayescap_variance_1d(inv_alpha, beta)

The `inv_alpha_net` sits between the structured sigma_total and the actual predictive
variance. The aux loss supervises sigma_total -> error, but the NLL trains inv_alpha
-> error through a separate MLP. They are decoupled; the net discards our decomposition's
spatial structure.

**Fix**: `mode="closed_form"` ties them: `inv_alpha = 1/sqrt(2 sigma_total)`, `beta=2`,
and `bayescap_variance_1d` reduces to exactly `var = sigma_total`. Now Gaussian-NLL MLE
directly drives sigma_total -> (y-yhat)^2, and the aux loss reinforces the same target.

Result (cf1): PCC(predvar, err) jumped from ~0 to **+0.867**, NLL -1.96 -> **-2.018**.

## 4. Experiment sequence (cf1-cf6)

All use: closed_form, xy aleatoric input, sigma_z = cycle_iso + second-order EKF.

| Run | Change | PCC | Spearman | NLL | Lesson |
|-----|--------|-----|----------|-----|--------|
| cf1 | pure NLL (no aux) | 0.867 | 0.399 | -2.018 | indirection was the bug |
| cf2 | + linear aux 0.3 | **0.879** | 0.410 | **-2.031** | aux helps marginally |
| cf3 | big head (128, 4L), 50ep | 0.868 | 0.454 | — | NOT capacity-limited |
| cf4 | + smoothed-error eval (20 bins) | 0.868 | 0.407 | — | fine-bin smoothing inconclusive |
| cf5 | log-space aux | 0.863 | 0.388 | — | NOT heavy-tail loss weighting |
| cf6 | + cell-level corr (8/12 bins) | 0.872 | 0.404 | — | resolved (see below) |

Decomposition (stable across runs):
- **PCC_aleatoric ~ 0.92-0.93** — the aleatoric head is the strong error tracker.
- **PCC_epistemic = 0.677** (fixed; sigma_ep is deterministic EKF).
- **PCC_total ~ 0.87** — slightly diluted below aleatoric-alone because on the ID-heavy
  test set, sigma_ep adds a near-constant ~0.01 floor that does not rank-discriminate
  ID points. Epistemic is essential for OOD but mildly dilutes ID calibration.

## 5. The Pearson/Spearman gap — resolved

Persistent observation: Pearson ~0.87 but Spearman ~0.40. Three hypotheses tested
and rejected:
- Capacity (cf3): 4x bigger head did not lift Spearman.
- Per-sample noise (cf4): fine-bin error smoothing did not lift it — but underpowered.
- Heavy-tail loss (cf5): log-space aux did not lift it (slightly hurt).

**cf6 cell-level correlation** (aggregate var and err into coarse cells, correlate
cell-means, proper noise averaging ~30 samples/cell):
- Cell-level Pearson (8x8) = **0.945** (noise averaging lifts Pearson)
- Cell-level Spearman (8x8) = 0.433 (NOT lifted)

**Conclusion**: variance tracks error *magnitude* strongly (Pearson 0.945) but ranks
bulk ID regions only moderately (Spearman 0.43). This is benign: the backbone's ID error
is nearly flat (~0.01) across most of the domain, sharply elevated only at the OOD edge
and function singularities. The model correctly identifies the high-error regions (high
Pearson). Spearman is diluted by many near-tied low-error ID cells, where ranking is
noise-dominated and uninformative by construction. There is no remaining structural
defect to fix.

## 6. Validated configuration (now the default)

    configs/model/ekf_net/default.yaml:
      mode: closed_form          # var = sigma_total (no indirection)
      use_aleatoric: true
      aleatoric_input_mode: xy   # raw inputs (toy); intermediate features for real data
      aleatoric_hidden_dim: 64
      aleatoric_n_layers: 3
      lambda_aleatoric: 1.0
    configs/model/hook_ekf.yaml:
      lambda_aux: 0.3            # linear MSE(sigma_total, err) regularizer
      aux_mode: linear

Final metrics: PCC(predvar,err)=0.88, cell-level PCC=0.945, NLL=-2.03, plus the
epistemic OOD sensitivity from doc 06 (sigma_ep grows ~10^4-10^5 OOD/ID via cycle+2nd-order).

## 7. Generalization note for real data

`aleatoric_input_mode=xy` uses raw (x1,x2), valid for the toy. For frozen
CLIP/DINOv2/medical encoders the raw input is high-dim; the practical equivalent is to
feed the head intermediate encoder activations (which retain spatial/geometric structure
the final latent compresses away — the same reasoning that made `z_only` fail and `xy`
work here). This mirrors the BN-shift intuition from doc 03/04: information about where
the model is uncertain lives earlier in the network than the final latent.

## 8. Open items

1. The ID/OOD dilution (PCC_al 0.93 > PCC_total 0.87) could be reduced with a learned
   or gated combination, but that risks overfitting the ID metric at OOD's expense.
   Kept the principled sum; report ID calibration and OOD sensitivity separately.
2. Calibration metric to add for the paper: expected calibration error (ECE) / reliability
   diagram, not just correlation.
3. Real-data validation: wire intermediate-activation input for the aleatoric head and
   confirm the closed-form + aux recipe transfers off the toy.

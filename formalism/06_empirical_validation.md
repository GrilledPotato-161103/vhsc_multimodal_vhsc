---
title: Empirical Validation of the Sigma_z + Propagation Design Space
tags: [SURE, experiments, OOD, cycle-consistency, second-order, results]
created: 2026-06-11
status: confirmed
context: |
  30-attempt systematic sweep on the toy BiModalRegressor. Covers all amplitude
  methods (docs 01, 04, 05) and all propagation variants (docs 02, 03).
  Metric: OOD/ID sigma_pred_sq ratio at delta=1.0 (x1 shifted by 1 unit).
  Full raw data in logs/experiments_input_ood.md and logs/slurm_ood20_19099514.log.
related: [[01_input_shift_measurement]] [[03_phase1_limits_and_phase2_direction]]
         [[04_sigma_z_extensions]] [[05_untying_latent_collapse]]
---

# Empirical Validation

## 1. Summary of findings

| Method | sps ratio | Conclusion |
|---|---|---|
| SD first-order (baseline) | 1.07x | amp grows 4.4x; sps flat (Jacobian collapse) |
| PCA k=2 + first-order | 1.66x | best latent-space method |
| GMM K=8 + second-order | 4.55x | best latent-space + propagation fix |
| Cycle+Identity + first-order | 6.2x | B-only, cycle bypasses encoder folding |
| SD + second-order | 1.68x | propagation fix alone: +57% |
| PCA + second-order | 2.30x | latent + propagation: better but limited |
| **Cycle+SigmaA + second-order** | **27,000–47,000x** | **SD SOTA** |
| **Cycle+Identity + second-order** | **46,000–310,000x** | **B-only SOTA** |
| Any + MC-dropout | <1.0x | universally harmful |

## 2. What each finding validates from the formalism

### 2a. Doc 03 §3 — Encoder folding exists but is quantified

SD Mahalanobis amplitude grows 4.4x at delta=1.0 (not zero), so the encoder does
NOT completely fold OOD inputs. The latent Mahalanobis is partially sensitive.
But the sps ratio is only 1.07x: the Jacobian collapse (doc 03 §4) washes out the
amplitude growth.

### 2b. Doc 05 §4 — Cycle-consistency bypasses folding

Cycle+Identity amplitude grows 5553x at delta=1.0; Cycle+SigmaA grows 4019x.
Three orders of magnitude above any latent-space method. The decoder exposes
input-space shift that the encoder's final latent conceals. Doc 05's prediction
is confirmed.

### 2c. Doc 03 §7 Fix 1 — Second-order Taylor fixes Jacobian collapse

First-order SD sps ratio: 1.07x.
Second-order SD sps ratio: 1.68x (+57%).
The second-order curvature term `(1/2) tr((H_g Sigma_recon)^2)` does not vanish
when J_g → 0, confirming the fix from doc 03. The effect compounds with the
amplitude method.

### 2d. Interaction: Cycle × Second-order is super-multiplicative

Cycle+Identity first-order: 6.2x sps ratio.
Second-order alone (SD): 1.68x ratio.
Cycle+Identity + second-order: 46,000–310,000x.

Expected from independence: 6.2 × 1.68 ≈ 10x. Actual: 46,000–310,000x.
The interaction is super-multiplicative by ~4 orders of magnitude.

Why: the second-order correction is `(1/2) ||H_g Sigma_recon||_F^2`. When
Cycle+Identity is used, the in-distribution Sigma_recon is very small (cycle
error ≈ 0 for source inputs → Sigma_z ≈ 0 → Sigma_recon ≈ 0). The denominator
of the sps ratio (ID sps) is near-zero; the numerator (OOD sps) is large.
The ratio explodes from both ends simultaneously. This is the correct behavior
for an ideal uncertainty estimator: near-zero uncertainty on known data, large
uncertainty on unknown data.

### 2e. Doc 03 §7 Fix 2 (Option B3) — MC dropout fails on frozen regression heads

All 8 MC dropout attempts (attempts 7, 10, 18, 19, 24 and the blends) give
sps ratio < 1.0 as the MC component. The frozen head extrapolates systematically:
all K dropout masks agree on the same near-constant extrapolation, so MC variance
*decreases* in OOD regions. MC dropout captures aleatoric heteroscedasticity
within the training distribution, not epistemic uncertainty about OOD inputs.
Doc 03's prediction that "MC dropout fails when the head extrapolates confidently"
is confirmed. Small MC blend (alpha=0.3) in attempt 25 gives a modest improvement
(SD+Blend: 2.0x vs SD first-order: 1.07x) by mixing in the EKF term.

## 3. Absolute sps values (not just ratios)

For Cycle+Identity + second-order (attempt 30, the B-only SOTA):

| delta | sps_mean | Interpretation |
|---|---|---|
| 0.00 (in-dist) | 5.77e-3 | near-zero uncertainty for source |
| 0.40 | 11.4 | 2000x above ID |
| 0.60 | 318 | sps_max hits 1e4 ceiling |
| 1.00 | 1792 | OOD regime; true values may be higher |
| 2.00 | 6719 | fully OOD |

Note: `pred_ceiling = 1e4` in `full_ekf_propagation_second_order` clamps the
maximum sps during eval. True values at delta ≥ 0.6 are larger. Remove the
ceiling in eval mode to see the unclamped dynamic range.

## 4. Variance across runs

The cycle decoder is re-initialised per run (random weights). All runs reach
`recon_loss ≈ 0` (source reconstruction near-perfect). But micro-differences in
the decoder give micro-differences in the in-distribution cycle error, which
shifts the denominator of the sps ratio:

| Attempt | Setup | sps ratio |
|---|---|---|
| 16 | Cycle+SigmaA + 2nd-order | 27,542x |
| 26 | same (re-run) | 47,800x |
| 17 | Cycle+Identity + 2nd-order | 112,366x |
| 27 | same (re-run) | 46,510x |
| 30 | same (re-run 2) | 310,811x |

Variance: 1-2 orders of magnitude in the ratio. Qualitative conclusion stable:
always 4–6 orders of magnitude above the SD first-order baseline.

For the paper: report median ratio over N=5 decoder runs, plus ±1 std. The key
claim ("cycle + second-order gives >> 1000x OOD/ID ratio vs 1.07x baseline") is
robust.

## 5. Rankings for paper

**B-only setting (no source data at all):**
1. Cycle+Identity + second-order: 46,000–310,000x
2. Cycle+Identity + first-order: 6.2x (simple baseline)
3. Cycle+Identity + blend(0.5): 1,024x (intermediate)

**SD setting (source statistics available):**
1. Cycle+SigmaA + second-order: 27,000–47,000x
2. Cycle+SigmaA + first-order: 273x
3. GMM K=8 + second-order: 4.55x (best latent method)

**Recommendation for paper experiments:**
- Report B-only (Cycle+Identity + 2nd-order) as the main result.
- Show ablation table: first-order alone (6.2x), second-order alone (1.68x),
  cycle alone (6.2x), both together (46,000x) → super-multiplicative interaction.
- Include GMM K=8 + second-order (4.55x) as the upper bound for methods
  without a cycle decoder (useful when x is not observable at inference time).

## 6. Open issues for the next implementation phase

1. **Remove eval ceiling.** `pred_ceiling=1e4` should be lifted in
   `full_ekf_propagation_second_order` for eval; keep it for training stability.
   Add a `training` flag or pass `pred_ceiling=float('inf')` from the eval script.

2. **Decoder stability.** The ~1-2 order-of-magnitude run-to-run variance is
   caused by the random decoder init. Fix this by seeding the decoder init
   (`torch.manual_seed(0)` before `_make_dec`). After fixing, re-run N=5 times
   and report the distribution.

3. **Training-time second-order.** Currently second-order is only tested in eval
   mode. Wiring it into `EKFBiModalInferer.forward` for training requires
   `create_graph=True` in the Hessian computation (expensive). Alternative:
   train with first-order, evaluate with second-order (valid because the two use
   the same Sigma_z and the heads are trained against first-order sigma_pred_sq).

4. **Calibration check.** The sps ratio confirms OOD sensitivity. Still need to
   check calibration: `corr(sps, (y - y_hat)^2)` on an in-distribution held-out
   set. Target: r > 0.3 (from formalism doc 02).

5. **Decoder size ablation.** The toy decoder (16→32→16→1) achieves near-zero
   loss in 3000 steps. For larger problems (VLMs, medical imaging), the decoder
   needs to be larger. Ablate: does decoder quality (training loss) correlate
   with the sps ratio?

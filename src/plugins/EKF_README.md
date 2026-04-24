---
title: Input Uncertainty Modeling — Class-Conditional Covariance Approach
tags: [SURE, uncertainty, OOD, covariance, kalman, distribution-shift]
created: 2026-04-08
status: draft
---

# Input Uncertainty Modeling for SURE

> Core design document for the distribution-shift → input covariance module.  
> This is the bottleneck that feeds the downstream EKF-style propagation chain (see `related_kalman_reconstruction.md`).

---

## 1. Problem Statement

Let z ∈ ℝᵈ be a feature vector extracted from a target-domain input by a frozen encoder (e.g., CLIP, DINOv2). Let the source training domain supply class-conditional statistics

    {(μ_c, Σ_c) | c = 1…C}

estimated from source feature representations — precisely the construction described in **Lee et al. 2018** (see `related_OOD_uncertainty.md`, §1).

We require a function

    z  →  Σ_z  ∈  ℝ^{d×d}  (symmetric positive semi-definite)

such that Σ_z:

1. **Captures the direction and magnitude of z's deviation** from the in-distribution manifold — i.e., reflects *where* in feature space z has drifted and by *how much*.
2. **Can be consumed by the downstream KF-style propagation chain** as the prior covariance Σ_avail in the EKF predict step Σ_recon = J_{g_θ} Σ_avail J_{g_θ}ᵀ + Σ_net (see `related_kalman_reconstruction.md`, §2–3 and the mapping table).
3. **Is computationally tractable at inference time** — closed-form or a small fixed number of operations over pre-stored source statistics.

The existing TMLR uncertainty propagation framework assumes deterministic feature inputs (Σ_avail = 0). This module fills that gap by providing a principled, non-trivial Σ_z.

---

## 2. Baseline: Scalar Mahalanobis Mapping

The simplest approach derives a scalar variance from the Mahalanobis distance of z to the predicted source class.

### Procedure

1. Predict the source class:
   ```
   c* = argmax_c  p(c | z)
   ```
   (softmax over class-conditional log-likelihoods under the fitted Gaussians)

2. Compute the Mahalanobis distance:
   ```
   d²(z)  =  (z − μ_{c*})ᵀ  Σ_{c*}⁻¹  (z − μ_{c*})
   ```

3. Map to a scalar variance via a monotone, non-negative function:
   ```
   σ²_input  =  softplus(α · d²(z) + β)
   ```
   where α > 0 and β are scalar calibration parameters fit on a held-out validation set. Softplus ensures σ²_input > 0 for all inputs.

4. Construct the input covariance as an isotropic matrix:
   ```
   Σ_z  =  σ²_input · I
   ```

### Foundation

This construction is a direct application of **Lee et al. 2018**: the class-conditional Gaussian fit on source features gives both μ_{c*} and Σ_{c*}; the Mahalanobis distance is their OOD score. **SSD (Sehwag et al. 2021, §7 of OOD notes)** validates that this recipe transfers to SSL encoders (CLIP/DINOv2) without modification.

### Pros

- Closed-form and parameter-free at inference time (only α, β require offline calibration).
- Directly reuses Lee et al. 2018 infrastructure — no new training required.
- Monotone in OOD-ness: in-distribution samples get small σ²_input; OOD samples get large σ²_input.
- The isotropic Σ_z is trivially PSD and invertible (after variance floor).

### Cons — Why This Is Insufficient

The scalar approach **collapses directional information**. Two samples equidistant from μ_{c*} in different directions receive identical σ²_input and thus identical Σ_z = σ²I. However, their residuals z − μ_{c*} point in different directions in ℝᵈ; after propagation through the EKF step

    Σ_recon  =  J_{g_θ} Σ_z J_{g_θ}ᵀ  +  Σ_net

the term J_{g_θ} (σ²I) J_{g_θ}ᵀ = σ² J_{g_θ} J_{g_θ}ᵀ is identically scaled regardless of direction. The downstream predictor therefore receives identical posterior uncertainty for two geometrically distinct deviations — potentially very different in how much they affect the missing modality reconstruction. The full covariance formulation below preserves this directional structure.

---

## 3. Preferred Approach: Class-Conditional Full Covariance (Low-Rank)

This is the main proposal. It addresses the scalar approach's loss of geometric information by using the full class-conditional covariance structure from Lee et al. 2018, combined with Mahalanobis scaling for OOD sensitivity.

### 3a. Full Covariance Source Statistics

Estimate class-conditional Gaussians on source domain features:

    {N(μ_c,  Σ_c) | c = 1…C}

where Σ_c ∈ ℝ^{d×d} is the full sample covariance of source-domain embeddings from class c.

**Low-rank + isotropic decomposition** for memory and compute tractability:

    Σ_c  ≈  U_c  D_c  U_cᵀ  +  σ²_floor · I

where:
- U_c ∈ ℝ^{d×k}  with orthonormal columns (k leading eigenvectors of Σ_c)
- D_c ∈ ℝ^{k×k}  diagonal (corresponding eigenvalues)
- σ²_floor > 0  is a small isotropic floor (see §5)
- k << d  (typically k = 20–50 for d = 512–2048)

**Storage**: O(dk + k) per class instead of O(d²). For d = 1024, k = 32, C = 10 classes: 320k floats vs. 10M floats for dense Σ_c. The rank-k decomposition captures the dominant variance directions; the σ²_floor term ensures the covariance is full-rank and PSD.

This decomposition follows directly from the class-conditional Gaussian fitting in **Lee et al. 2018** and is validated for SSL encoders by **SSD (Sehwag et al. 2021)**.

### 3b. Per-Sample Input Covariance Options

Given z and the predicted class c*, we propose three options for constructing Σ_z, in increasing order of sophistication:

---

**Option A — Direct Class Covariance**

    Σ_z  =  Σ_{c*}

Use the source class covariance directly as the uncertainty estimate for z.

*Interpretation*: Σ_{c*} encodes the typical spread of in-distribution samples around μ_{c*}. By assigning Σ_z = Σ_{c*}, we declare that z carries the same directional uncertainty structure as an average source-domain sample of class c*. This is appropriate when z is actually in-distribution and we want a baseline covariance that preserves geometric structure.

*Limitation*: Insensitive to how far z is from μ_{c*}. An extreme OOD sample and a near-in-distribution sample both receive the same Σ_z — the uncertainty is not amplified as shift increases.

---

**Option B — Mahalanobis-Scaled Covariance (Recommended)**

    Σ_z  =  d²(z, c*)  ·  Σ_{c*}

Scale the class covariance by the Mahalanobis distance of z from the source class mean.

*Interpretation*:
- **In-distribution** (d²(z, c*) ≈ 1 for a chi-squared distributed in-domain sample): Σ_z ≈ Σ_{c*} — the input carries typical source-domain uncertainty.
- **OOD** (d²(z, c*) >> 1): Σ_z is amplified proportionally — the further z deviates from the source manifold, the larger and more stretched its uncertainty ellipsoid.

This combines:
1. The **geometric structure** of Σ_{c*} (principal directions of in-domain variance, preserving the directional information the scalar approach loses)
2. The **OOD sensitivity** of d²(z) (Lee et al. 2018's Mahalanobis score)

The EKF downstream propagation step then becomes:

    Σ_recon  =  d²(z, c*) · J_{g_θ} Σ_{c*} J_{g_θ}ᵀ  +  Σ_net

The scalar d²(z, c*) amplifies the contribution of input uncertainty to the reconstructed modality covariance, while J_{g_θ} Σ_{c*} J_{g_θ}ᵀ preserves the directional structure. This is the recommended option.

---

**Option C — Residual Outer Product**

    Σ_z  =  (z − μ_{c*})(z − μ_{c*})ᵀ

Rank-1 update capturing the specific direction and magnitude of z's deviation from the class centroid.

*Interpretation*: Σ_z is rank-1 by construction, placing all uncertainty mass in the direction of the actual residual. Cheap to compute (O(d) storage, O(d) compute). Does not require storing or inverting Σ_{c*}.

*Limitation*: Ignores the class-level covariance structure entirely — two samples with residuals of equal norm but pointing in different directions receive different Σ_z, but the shape is determined only by the residual direction rather than the source distribution's geometry. Also rank-1, which means the downstream KF propagation will see near-degenerate input covariance in directions orthogonal to the residual.

---

**Recommendation: Option B**

Option B is preferred because it:
1. **Preserves the full directional geometry** of the source class distribution (the principal axes of Σ_{c*} reflect which directions are normally variable vs. constrained in-domain)
2. **Scales appropriately with OOD-ness** — in-distribution samples carry source-level uncertainty; OOD samples carry amplified uncertainty in the same geometric directions
3. **Requires only stored source statistics** (μ_{c*}, Σ_{c*}) plus one Mahalanobis distance computation — O(dk) using the low-rank decomposition

Option A is a useful ablation (demonstrates effect of geometric structure without OOD sensitivity). Option C is a useful diagnostic (shows effect of per-sample residual direction without class-level structure).

---

## 4. Test-Time Adaptation of Source Statistics

**Risk**: Under large distribution shift, source statistics (μ_c, Σ_c) become stale. μ_c and the principal directions U_c were estimated from source training data; if the target domain has shifted significantly, these statistics no longer represent "what in-distribution looks like" for features extracted from target inputs. A stale Σ_c will produce miscalibrated d²(z) values and misaligned Σ_z directions.

**Evidence from literature**: **DUA (Mirza et al. 2022, §9 of OOD notes)** directly addresses this for batch normalization statistics: stale BN statistics degrade representation quality, and shift-proportional interpolation between source and test statistics recovers performance. The same logic applies to class-conditional covariances. **Tent (Wang et al. 2021, §8 of OOD notes)** shows that entropy minimization on BN parameters can partially re-align features under shift.

### Proposed Adaptation Strategy

**Step 1 — TENT-style BN adaptation** (if the encoder has BN layers):
Before estimating class-conditional covariances on the target batch, update BN affine parameters by minimizing entropy of class predictions on the test batch (Wang et al. 2021). This re-normalizes internal feature distributions to partially align with the target domain, improving the quality of the subsequent covariance estimation. One forward-backward pass per test batch; BN parameters only.

**Step 2 — Exponential moving average (EMA) of μ_c**:
During deployment, update the class-conditional mean estimates using an EMA on target-domain samples assigned to each class with high confidence:

    μ_c  ←  (1 − γ) μ_c  +  γ · z   (for samples with p(c* = c | z) > τ)

where γ ∈ (0, 1) is the EMA decay and τ is a confidence threshold. No labels needed; soft-assignment weighting by p(c | z) is an alternative to hard thresholding.

**Step 3 — Low-rank streaming update of Σ_c**:
Maintain a streaming PCA estimate of the target-domain covariance per class, initialized from the source Σ_c and updated using incoming target samples. The rank-k structure makes this tractable: update U_c, D_c using online SVD (e.g., incremental PCA with rank-k approximation). This is more expensive than EMA on μ_c but captures covariance drift.

**Trade-off**: All adaptation steps are unsupervised (no target labels). Tent is most robust (entropy minimization is well-studied); EMA on μ_c adds minimal overhead; streaming PCA on Σ_c is optional and triggered only when shift is large (as detected by d²(z) sustained above a threshold). The DUA framework provides the shift-proportional trigger logic for when adaptation is necessary.

---

## 5. Variance Floor and Numerical Stability

Any Σ_z constructed above must be PSD and invertible for use in the downstream EKF chain, which requires computing J_{g_θ} Σ_z J_{g_θ}ᵀ and potentially Σ_z⁻¹ (for the KF update step if observations are incorporated).

**Guarantee**: Add an isotropic floor:

    Σ_z  ←  Σ_z  +  ε · I      where  ε  =  σ²_min  >  0

This ensures:
- **PSD**: Σ_z + εI is strictly positive definite for any Σ_z ≽ 0
- **Invertibility**: all eigenvalues are at least ε, preventing singular matrix issues in Σ_z⁻¹
- **Numerical conditioning**: condition number is bounded by (λ_max + ε) / ε

**Precedent**: **Seitzer et al. 2022 (§6 of Kalman notes)** on beta-NLL training discusses minimum variance regularization as the analogous mechanism: setting a variance floor prevents the network from producing σ² → 0, which would assign infinite precision to a single sample and destabilize NLL training. The KF analog is identical: a floor ε prevents the covariance matrix from becoming numerically singular during propagation.

**Practical value**: σ²_min should be set to match the magnitude of numerical noise in the feature representation — typically 1e-4 to 1e-6 for normalized embeddings. It can also be interpreted as irreducible observation noise (there is always some measurement uncertainty regardless of distribution shift).

For the low-rank decomposition Σ_c ≈ U_c D_c U_cᵀ + σ²_floor I, the floor σ²_floor already serves this role: the full covariance has minimum eigenvalue σ²_floor > 0 by construction.

---

## 6. Open Questions

1. **Multi-modal source distributions per class**: The class-conditional Gaussian assumes each source class is unimodal. Medical imaging datasets often exhibit multimodality within a class (e.g., different scanner protocols, patient subgroups). A Gaussian mixture model (GMM) per class would capture this: d²(z) becomes the minimum Mahalanobis distance over mixture components, and Σ_{c*} is replaced by the covariance of the nearest component. Trade-off: GMM fit requires EM and additional hyperparameters (number of components); may be over-parameterized for small source datasets.

2. **Learned stochastic embedding as alternative**: A PFE-style network (**Shi & Jain 2019, §3 of OOD notes**) or HIB-style distributional loss (**Oh & Murphy 2021, §4 of OOD notes**) could replace the closed-form Mahalanobis approach entirely. The encoder would directly output (μ_z, Σ_z) and be trained end-to-end with an MLS or HIB objective. Advantage: Σ_z is learned from data rather than post-hoc estimated, potentially capturing non-Gaussian structure. Disadvantage: requires fine-tuning the encoder (not applicable to frozen CLIP/DINOv2) and training data must cover the anticipated shift range. This is the preferred long-term direction if encoder fine-tuning is permitted.

3. **Rank k selection**: The low-rank decomposition with k principal components is a key hyperparameter. Empirically, k = 20–50 is typical for d = 512–2048 feature spaces (the leading PCA components capture most variance in learned representations). A principled selection criterion: choose k such that ∑_{i=1}^k λ_i / ∑_{i=1}^d λ_i ≥ 0.95 (95% of source variance explained). The remaining variance is absorbed into σ²_floor. For ablation, k = 1 recovers a structured rank-1 approximation (the top principal direction only), and k = d recovers the full dense covariance.

4. **Offline storage vs. deployment access to source data**: The class-conditional statistics (μ_c, Σ_c or U_c, D_c) are estimated from source training data but can be stored as a compact offline artifact — no source data needed at deployment. For d = 1024, k = 32, C = 10: 320k + 32k + 10k ≈ 360k floats per class-conditional model ≈ 14MB total at float32. This fits comfortably in memory and does not require source data access at inference time. The TTA adaptation steps (§4) operate on target-domain samples only.

5. **Interface precision with the downstream TMLR framework**: The EKF chain (`related_kalman_reconstruction.md`, mapping table) expects Σ_avail as a d×d matrix. The downstream framework presumably handles the Jacobian computation and matrix multiplications. This module must expose: (a) μ_z (the feature mean — typically z itself for a deterministic encoder), (b) Σ_z as specified above. Whether the framework needs full, diagonal, or low-rank Σ_z determines which of Option A/B/C is most compatible without additional conversion steps. The low-rank format U_c D_c U_cᵀ + εI can be stored compactly and materialized to dense on demand.

---

## Summary Table

| Approach | Covariance Structure | OOD-Sensitivity | Training Needed | Recommended Use |
|---|---|---|---|---|
| Scalar Mahalanobis (§2) | σ²I (isotropic) | Yes (via d²) | No | Ablation baseline |
| Option A: Direct class cov (§3b) | Full Σ_{c*} (low-rank) | No | No | Ablation (geometry without scaling) |
| **Option B: Mahalanobis-scaled (§3b)** | **d²(z) · Σ_{c*} (low-rank)** | **Yes** | **No** | **Primary recommendation** |
| Option C: Residual outer product (§3b) | (z−μ)(z−μ)ᵀ (rank-1) | Yes (implicit) | No | Diagnostic |
| PFE/HIB learned (§6, Q2) | Learned diagonal/full | Yes (via training) | Fine-tune encoder | Long-term if fine-tuning permitted |

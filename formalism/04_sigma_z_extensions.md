---
title: Three Extensions to the Sigma_z Shift Descriptor
tags: [SURE, theory, sigma_z, BN-shift, GMM, manifold, OOD]
created: 2026-06-11
status: draft
context: |
  The SD-setting Sigma_z (doc 01) uses a single Gaussian N(mu_A, Sigma_A) fit on
  source latents and Mahalanobis distance as the shift amplitude. This works for
  in-distribution calibration (doc 02-03) but fails to detect shift when the encoder
  folds OOD inputs back into the latent cloud (doc 03 §3). This note elaborates
  three alternative / complementary shift descriptors, each addressing the failure
  from a different angle.
related: [[01_input_shift_measurement]] [[03_phase1_limits_and_phase2_direction]]
---

# Three Extensions to the Sigma_z Shift Descriptor

All three are drop-in replacements for the amplitude function $s(z)$ in the product decomposition $\Sigma_z(z) = s(z) \cdot \Phi(z)$ from doc 01 §4. The shape term $\Phi$ follows the same options as before (isotropic $I$, source covariance $\Sigma_A$, cluster covariance $\Sigma_k$). The three extensions differ in *how* they define "anomalous" and *where* they measure it.

---

## Extension 1 — BatchNorm Shift Score

### Motivation

Latent Mahalanobis fails because the encoder erases input-space shift (doc 03 §3). BN-shift measures shift *inside* the encoder at every BN layer, before the folding is complete.

### Mechanism

Every BN layer $l$ stores two tensors in the frozen model's weights:
- $\mu^l \in \mathbb{R}^{d_l}$: EMA of channel means during training on $A$.
- $\sigma^{l,2} \in \mathbb{R}^{d_l}$: EMA of channel variances during training on $A$.

These are **source statistics** that cost nothing to obtain — they live inside the frozen $M_A$ already.

For a target sample $x_B$, we run a forward pass and hook the **pre-normalization activation** $h^l \in \mathbb{R}^{B \times d_l}$ at each BN layer $l$. The per-layer shift score is:

$$s_l(x_B) = \frac{1}{d_l} \sum_{c=1}^{d_l} \frac{(h^l_c - \mu^l_c)^2}{\sigma^{l,2}_c + \epsilon}$$

This is the per-channel standardized squared deviation from A's expected activations, averaged over channels. Summed over layers:

$$\boxed{s_{\mathrm{BN}}(x_B) = \frac{1}{L} \sum_{l=1}^{L} s_l(x_B)}$$

The output is a scalar per sample, computed from a single forward pass through the frozen encoder (hooks are zero-overhead at inference if PyTorch's `register_forward_hook` is used).

### Why it escapes encoder folding

Folding is a property of the **output** of $f_\theta$. The intermediate activations at early layers $h^1, h^2$ are much closer to the input — they carry signal about $x_B$ that has not yet been normalized away. Concretely:

```
x_B (OOD) → h¹ (still OOD, s₁ is high)
           → BN₁ normalizes h¹  ← uses μ¹, σ¹ from A
           → h² (partially folded, s₂ moderate)
           → BN₂ normalizes h²
           → ...
           → z_B (fully folded, d_M² ≈ 1)
```

The BN-shift score averages $s_l$ over all layers, so early-layer OOD signal is not discarded. Even if $s_L \approx 0$ (fully folded at the output), $s_1, s_2$ are non-zero and pull the average up.

### Properties

- **(P1) In-distribution**: for $x_B \sim A$, each $h^l \sim \mathcal{N}(\mu^l, \mathrm{diag}(\sigma^{l,2}))$ by construction (BN runs in eval mode, statistics were estimated on $A$). So $\mathbb{E}_A[s_l] = 1$ per channel per layer, and $\mathbb{E}_A[s_{\mathrm{BN}}] = 1$.

- **(P2) Monotone in shift**: as $x_B$ drifts from $A$, early-layer activations deviate more. $s_{\mathrm{BN}}$ grows.

- **(P3) EKF-consumable**: convert to covariance as $\Sigma_z(x_B) = s_{\mathrm{BN}}(x_B) \cdot \Phi$, exactly as in doc 01 §4.

- **B-only**: requires zero source data. Only the frozen model's BN running stats.

### Relation to the literature

DUA (Mirza et al. 2022) uses BN statistics to *adapt* the model under shift — shifting the BN running stats toward target-domain estimates. We use the *deviation* from those stats as a detection signal; the two are complementary. Tent (Wang et al. 2021) shows BN stats are sensitive enough to the distribution to be used for entropy minimization — implying they do carry shift information.

### One subtlety: per-layer weighting

Layer 1 is closest to input space; layer $L$ is closest to the folded latent. Equal weighting (arithmetic mean) is simple, but a depth-weighted version that discounts later layers may be more informative:

$$s_{\mathrm{BN}}^{\mathrm{weighted}}(x_B) = \frac{1}{Z} \sum_{l=1}^{L} w_l \cdot s_l, \quad w_l = e^{-\lambda (l-1)/L}$$

where $\lambda > 0$ upweights early layers. Ablate $\lambda \in \{0, 0.5, 1, 2\}$; $\lambda = 0$ recovers the uniform average.

---

## Extension 2 — Multi-Cluster (Gaussian Mixture) Mahalanobis

### Motivation

The single-Gaussian assumption in SDSigmaZ forces $\Sigma_A$ to span the *entire* variation of $A$'s latent distribution. If $A$ has internal structure — scanner types, patient subgroups, semantic clusters — $\Sigma_A$ becomes large and diffuse. A sample near one cluster center but far from the others is mislabeled as "typically uncertain". The Mahalanobis metric is averaged over unrelated directions.

The fix: fit $K$ clusters $\{(\mu_k, \Sigma_k, \pi_k)\}$ to $A$'s source latents, then measure distance to the *nearest* cluster rather than the global center.

### Formulation

**Fitting phase** (offline, once, on source data):

1. Cluster source latents $\{z^A_i\}$ into $K$ groups. Options: $K$-means on $z$ (fast), EM-GMM (principled), spectral clustering (nonlinear).
2. For each cluster $k$: estimate $\mu_k$ (mean), $\Sigma_k$ (sample covariance), $\pi_k$ (relative size).
3. Apply the same shrinkage and eigen-clamping from doc 01 / Change 2 to each $\Sigma_k$ independently.

**Inference phase** (per sample):

Hard assignment (simpler):
$$k^*(z_B) = \argmin_k \, (z_B - \mu_k)^\top \Sigma_k^{-1} (z_B - \mu_k)$$
$$s_{\mathrm{GMM}}(z_B) = d_M^2(z_B; \mu_{k^*}, \Sigma_{k^*}) / d_z$$
$$\Sigma_z(z_B) = s_{\mathrm{GMM}}(z_B) \cdot \Sigma_{k^*}$$

Soft assignment (more stable):
$$p(k \mid z_B) \propto \pi_k \cdot \exp\!\big(-\tfrac{1}{2} d_M^2(z_B; \mu_k, \Sigma_k)\big)$$
$$s_{\mathrm{GMM}}^{\mathrm{soft}}(z_B) = \sum_k p(k \mid z_B) \cdot d_M^2(z_B; \mu_k, \Sigma_k) / d_z$$
$$\Sigma_z(z_B) = \sum_k p(k \mid z_B) \cdot s_{\mathrm{GMM}}^k(z_B) \cdot \Sigma_k$$

The soft version is differentiable and produces a mixture of local covariances weighted by cluster posterior.

### Properties

- **(P1) In-distribution**: for $z_B \sim p_A^z$, the sample belongs to one cluster with $d_M^2 \approx d_z$ by construction (each cluster is fit to look like a $d_z$-dimensional Gaussian from within). So $s_{\mathrm{GMM}} \approx 1$.

- **(P2) Monotone**: farther from *all* cluster centers → larger amplitude.

- **(P3) EKF-consumable**: covariance is positive semi-definite by construction.

- **SD-setting**: requires source latents to fit the clusters. Can be made B-only if we fit clusters to a target-domain batch (approximate source structure from available target data).

### Choosing K

- **K=1**: recovers the single-Gaussian SDSigmaZ. The baseline.
- **K = number of classes** (if labeled): class-conditional Gaussians — exactly Lee et al. 2018. The label becomes the cluster assignment; the discriminant score becomes the OOD score.
- **K selected by BIC/AIC**: for unlabeled source data, fit GMMs for $K \in \{1, 2, 4, 8\}$ and pick the $K$ with the best BIC. This is the unsupervised version.

For the toy dataset: $K = 1$ is likely correct — the source distribution is a product of two 1D uniforms with no internal clustering. For medical imaging datasets with multiple scanner protocols, $K = 2$–$5$ may be needed.

### Why this is strictly more expressive than single-Gaussian

The single Gaussian's $\Sigma_A$ is the *total* covariance:
$$\Sigma_A = \sum_k \pi_k \big(\Sigma_k + (\mu_k - \mu_A)(\mu_k - \mu_A)^\top\big)$$

The between-cluster term $(\mu_k - \mu_A)(\mu_k - \mu_A)^\top$ inflates $\Sigma_A$ in the directions separating cluster centers. For those directions, $\Sigma_A^{-1}$ assigns *smaller* weight — making the single-Gaussian Mahalanobis less sensitive to between-cluster OOD directions. The multi-cluster version removes that between-cluster inflation within each local $\Sigma_k^{-1}$.

---

## Extension 3 — Manifold Projection Distance

### Motivation

Both SDSigmaZ and the GMM extension assume the source distribution is (locally) Gaussian in latent space. But a frozen encoder maps a low-dimensional input ($x \in \mathbb{R}^{d_x}$, here $d_x = 1$) through a high-dimensional bottleneck ($z \in \mathbb{R}^{d_z}$, here $d_z = 16$ per modality). The image of $f_\theta$ is therefore a **$d_x$-dimensional manifold** embedded in $\mathbb{R}^{d_z}$ — a curve, not an ellipsoid.

For the toy: $z_1 = f_\theta(x_1)$ is a **curve** in $\mathbb{R}^{16}$ parameterized by $x_1 \in [-1, 1]$. The full latent $z = [z_1, z_2]$ is a **2D surface** in $\mathbb{R}^{32}$ (product of two independent curves). A 32D Gaussian is wildly over-parameterized — it cannot know that 30 of the 32 dimensions are constrained.

The Mahalanobis metric implicitly "knows" this via the near-zero eigenvalues of $\Sigma_A$ in the off-manifold directions, but those directions are then dominated by the regularization floor (`cov_floor`). The manifold approach makes the low-dimensionality explicit.

### Formulation

The **manifold distance** from $z_B$ to $\mathcal{M}_A = f_\theta(A)$ is:

$$d_{\mathcal{M}}(z_B) = \min_{z \in \mathcal{M}_A} \|z_B - z\|$$

i.e. the Euclidean distance from $z_B$ to its closest point on the source manifold. This replaces $d_M^2$ as the amplitude:

$$\Sigma_z(z_B) = d_{\mathcal{M}}^2(z_B) \cdot \Phi(z_B)$$

### Three implementations, cheap to expensive

**Implementation A — Linear subspace (PCA) projection distance**

Approximate $\mathcal{M}_A$ as a $k$-dimensional affine subspace spanned by the top-$k$ eigenvectors $U_k \in \mathbb{R}^{d_z \times k}$ of $\Sigma_A$:

$$\mathcal{M}_A \approx \mu_A + \mathrm{span}(U_k)$$

Projection distance:
$$d_{\mathrm{PCA}}^2(z_B) = \|(z_B - \mu_A) - U_k U_k^\top (z_B - \mu_A)\|^2 = \sum_{i > k} \lambda_i^{-1} \hat{z}_i^2$$

where $\hat{z}_i = u_i^\top (z_B - \mu_A)$ is the $i$-th PC coefficient. This is the squared norm of the residual *off* the top-$k$ subspace. **In-manifold** samples have small residual (their variance lives in the top-$k$ subspace); **OOD** samples have large residual (they deviate in off-manifold directions).

Note the connection to Mahalanobis: if $k = d_z$ (keep all PCs), $d_{\mathrm{PCA}}^2 = 0$ identically. If $k = 0$, $d_{\mathrm{PCA}}^2 = \|z_B - \mu_A\|^2$ (Euclidean). Intermediate $k$ interpolates: trust the top-$k$ directions as "natural variation", measure anomaly only in the remaining $d_z - k$ directions.

Choosing $k$: select the smallest $k$ such that $\sum_{i=1}^k \lambda_i / \sum_{i=1}^{d_z} \lambda_i \geq 0.95$ (95% of source variance explained). For our toy with 1D input, $k = 1$ per modality is the theoretically correct choice.

**Implementation B — Autoencoder reconstruction error**

Train a shallow autoencoder:
$$\mathrm{enc}: \mathbb{R}^{d_z} \to \mathbb{R}^k, \quad \mathrm{dec}: \mathbb{R}^k \to \mathbb{R}^{d_z}$$
on source latents $\{z^A_i\}$, with bottleneck dimension $k \ll d_z$. The reconstruction error:
$$d_{\mathrm{AE}}^2(z_B) = \|z_B - \mathrm{dec}(\mathrm{enc}(z_B))\|^2$$
approximates the squared distance to the manifold $\mathcal{M}_A$ that the autoencoder learned. A nonlinear autoencoder can capture *curved* manifolds, which the PCA subspace cannot.

Cost: one extra small network to train on source latents. At inference: two extra forward passes (enc + dec). For $d_z = 32$ and $k = 2$, the autoencoder is tiny (< 1000 parameters).

**Implementation C — Normalizing flow negative log-density**

Fit a normalizing flow $p_\theta(z)$ on source latents. The shift score:
$$s_{\mathrm{flow}}(z_B) = -\log p_\theta(z_B)$$
is exactly $d_{\mathrm{dens}}$ from doc 01 §3. For in-distribution samples, $-\log p_\theta(z_A) \approx H[p_A^z]$ (entropy). For OOD, $-\log p_\theta(z_B) > H[p_A^z]$. This is the density-based formulation that reduces to Mahalanobis under a Gaussian flow and to manifold distance for a flow that learns a low-dimensional manifold structure.

Most expressive but most expensive to fit and most prone to overfit on small source datasets.

### Choosing the implementation

| | PCA (A) | Autoencoder (B) | Flow (C) |
|---|---|---|---|
| Handles curved manifolds | no | yes | yes |
| Source data needed | yes | yes | yes |
| Training cost | O(N d²) SVD | small MLP | normalizing flow |
| Inference cost | O(k d) | 2 forwards | 1 forward |
| Can become B-only | no | no (needs A training) | no |
| Theoretical grounding | strong (linear PCA) | moderate (AE) | strongest (exact density) |

**Recommendation**: for the toy (1D input → 1D manifold), PCA with $k=1$ per modality is theoretically exact and computationally free. For medical imaging settings (where the manifold is higher-dimensional and potentially curved), the autoencoder is the practical sweet spot.

### The PCA case is a natural complement to Mahalanobis

The single-Gaussian Mahalanobis measures:
$$d_M^2 = \sum_{i=1}^{d_z} \hat{z}_i^2 / \lambda_i \quad \text{(weighted by all eigenvectors)}$$

The PCA projection distance measures:
$$d_{\mathrm{PCA}}^2 = \sum_{i=k+1}^{d_z} \hat{z}_i^2 \quad \text{(unweighted, only off-manifold directions)}$$

The *within-manifold* part of $d_M^2$ is:
$$d_M^{2,\mathrm{on}} = \sum_{i=1}^k \hat{z}_i^2 / \lambda_i \quad \text{(variation along the manifold)}$$

So the full picture decomposes:
$$\underbrace{d_M^2}_{\text{Mahalanobis}} = \underbrace{d_M^{2,\mathrm{on}}}_{\text{within-manifold variation}} + \underbrace{d_M^{2,\mathrm{off}}}_{\propto d_{\mathrm{PCA}}^2 \text{ up to cov floor}}$$

A sample from $B$ that is OOD *perpendicularly* to the manifold (the geometry-relevant case) has large $d_{\mathrm{PCA}}^2$ and small $d_M^{2,\mathrm{on}}$. A sample that is simply at the edge of A's support (but still on the manifold) has large $d_M^{2,\mathrm{on}}$ and small $d_{\mathrm{PCA}}^2$. The two distances are thus complementary: one detects "moved along the data manifold but past its extent", the other detects "moved off the data manifold entirely".

---

## Summary Table

| Extension | Measurement point | Distributional model | B-only? | Best for |
|---|---|---|---|---|
| **SD Mahalanobis (current)** | Final latent $z$ | Single Gaussian | No | Baseline; in-dist calibration |
| **BN-Shift (Extension 1)** | BN activations at every layer | None (score-based) | Yes | Detecting shift the encoder folds |
| **Multi-Cluster GMM (Extension 2)** | Final latent $z$ | Gaussian Mixture | Partially | Multimodal source distributions |
| **PCA projection (Extension 3A)** | Final latent $z$ | Low-rank subspace | No | Low-dim manifold structure |
| **AE reconstruction (Extension 3B)** | Final latent $z$ | Nonlinear manifold | No | Curved source manifolds |

All four are compatible with the product decomposition $\Sigma_z = s \cdot \Phi$. All four can be combined:
- BN-Shift + GMM: apply GMM to the BN-shift features rather than the latent.
- PCA + BN-Shift: use PCA off-manifold distance in latent space, then weight by BN-shift score from early layers. Captures both geometry and early-layer signal.

---

## Discussion: which to pursue for the paper

### For the B-only publication claim

Only **Extension 1 (BN-Shift)** is truly B-only. Extensions 2 and 3 require fitting source statistics (cluster means/covariances, or training an AE) on source data $A$. The publication narrative (doc 03 §9) positions B-only as the lead method, so BN-Shift is the one to implement first.

### For understanding the geometry of the problem

**Extension 3A (PCA)** is theoretically cleanest for the toy. We know the source manifold is 2D (1D per modality, 2 modalities). Fitting $k = 1$ PCA component per modality and measuring off-manifold deviation should give a perfect shift detector for the toy's geometry. Running it would confirm whether the Mahalanobis failure is due to (a) wrong distributional model (Gaussian vs manifold) or (b) wrong measurement point (latent vs intermediate layers). These are empirically separable.

### For medical imaging / VLMs

**Extension 2 (GMM)** is most realistic. Medical datasets routinely have multiple scanner protocols, age/sex subgroups, etc. A class-conditional version (Lee et al. 2018) is the natural instantiation if labels are available. The unsupervised GMM is the label-free fallback.

---

## Open questions

1. **For the toy, does PCA projection distance actually detect the shift that Mahalanobis misses?** Empirical test: compute $d_{\mathrm{PCA}}^2$ with $k = 2$ on the same samples where $d_M^2 \approx 1$ for OOD inputs. If $d_{\mathrm{PCA}}^2$ grows with shift, the failure was distributional-model mismatch, not measurement-point mismatch (i.e. the encoder does expose the shift at the output, but in off-manifold directions that Mahalanobis underweights).

2. **Is the BN-shift score layer-wise signal monotone decreasing with depth?** If yes, the early layers carry all the information and we can use a truncated version (first $L'$ layers only). If not, some late layers may still contribute OOD signal via a different mechanism.

3. **Can we combine BN-Shift and PCA projection into a single unified score?** Both measure deviation from the source distribution at different points in the network. A learnable linear combination $s = \alpha \cdot s_{\mathrm{BN}} + (1-\alpha) \cdot d_{\mathrm{PCA}}^2$ trained on held-out OOD data might dominate either alone.

4. **For the GMM, does soft assignment with the correct posterior actually recover the information lost by hard assignment?** Near cluster boundaries, hard assignment introduces a discontinuity. For smooth downstream EKF propagation, soft assignment is preferable — but it also mixes cluster covariances, which may blur the directional information that individual $\Sigma_k$ carry.

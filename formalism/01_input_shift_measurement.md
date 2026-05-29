---
title: Formalizing the Per-Sample Input-Shift Descriptor
tags: [SURE, theory, sigma_z, distribution-shift, EKF]
created: 2026-05-29
status: draft
context: |
  Addresses the first conceptual error in the current EKF pipeline:
  `GroundTruthSigmaZ` conflates marginal feature variance Var_A[z] with
  per-sample input-shift uncertainty. This note formalizes what the
  per-sample shift descriptor should be, and how it feeds the EKF chain.
---

# Formalizing the Per-Sample Input-Shift Descriptor

## 1. What is "shift" — two concepts, only one is ours

Notation:

- Source distribution: $A$ over inputs $x \in \mathbb{R}^{d_x}$, density $p_A(x)$.
- Encoder: $f_\theta : \mathbb{R}^{d_x} \to \mathbb{R}^{d_z}$, deterministic, frozen.
- Push-forward latent distribution under source: $p_A^z(z) = \int \delta(z - f_\theta(x))\, p_A(x)\, dx$.
- Query sample: $x_B \in B$ (target), with latent $z_B = f_\theta(x_B)$.

Two perfectly defensible notions of "shift" get conflated in the current code:

**Aleatoric (observation-level)**. $x_B$ is a *noisy realization* of a clean input. Uncertainty about $x_B$ propagates to $z_B$ via the classical EKF step at the encoder:
$$\Sigma_z(x_B) = J_f(x_B) \Sigma_x J_f(x_B)^\top$$
Requires a known input noise model $\Sigma_x$. Per-sample, principled. **Does not depend on $A$** — not what we want.

**Epistemic / distribution mismatch**. $x_B$ may be drawn from $B \neq A$. We want a descriptor of "how far has $z_B$ drifted from the bulk of $p_A^z$?". This is what SURE actually targets.

**Decision**: SURE works with the second concept. Shift is measured *with respect to $A$*.

## 2. Define what a shift descriptor formally is

A **shift descriptor**:
$$S : \mathbb{R}^{d_z} \longrightarrow \mathcal{S}$$
for some target space $\mathcal{S}$, satisfying:

(P1) **Anchored on $A$.** $S(z) \approx 0_{\mathcal{S}}$ when $z \sim p_A^z$ — null shift for in-distribution.

(P2) **Monotone in atypicality.** $\|S(z)\|$ grows as $z$ moves into low-$p_A^z$-density regions.

(P3) **EKF-consumable.** $S(z)$ converts to a PSD matrix $\Sigma_z(z) \in \mathbb{S}^{d_z}_{+}$, since the EKF step $\Sigma_{\mathrm{recon}} = J_f \Sigma_z J_f^\top$ requires a covariance.

The choice of $\mathcal{S}$ determines what survives:

| $\mathcal{S}$ | Information preserved | Lost |
|---|---|---|
| $\mathbb{R}_+$ (scalar) | magnitude of deviation | direction |
| $\mathbb{R}^{d_z}$ (vector) | per-coordinate deviation | covariance structure |
| $\mathbb{S}^{d_z}_+$ (PSD matrix) | full directional uncertainty | nothing |

The current `GroundTruthSigmaZ` lives in $\mathcal{S} = \mathbb{R}^{d_z}$ (diagonal) but violates (P2): it computes the marginal $\mathrm{Var}_A[z]$ once at init and never updates. $S$ is constant in $z$ — has no shift information.

## 3. Three canonical scalar shift measures

**(a) Density-based**:
$$s_{\mathrm{dens}}(z) = -\log p_A^z(z)$$
Requires modeling $p_A^z$. Cleanest from a probability standpoint; reduces to other forms under simplifying assumptions.

**(b) Mahalanobis to fitted Gaussian(s)** (Lee et al. 2018). Assume $p_A^z \approx \sum_c \pi_c \mathcal{N}(\mu_c, \Sigma_c)$. Then:
$$s_{\mathrm{Maha}}(z) = \min_c (z - \mu_c)^\top \Sigma_c^{-1} (z - \mu_c)$$
Identical to $-2 \log p_A^z(z) + \mathrm{const}$ when the mixture is hard-assigned. Closed-form, well-studied.

**(c) Discriminative (BN-shift, classifier confidence, free-energy)**:
$$s_{\mathrm{BN}}(z) = \frac{1}{L}\sum_l \frac{1}{d_l}\sum_i \frac{(z_i^l - \mu^l_{\mathrm{BN},i})^2}{\sigma^{l\,2}_{\mathrm{BN},i}}$$
over BN layers (what `BNShiftSigmaZ` currently computes).

All are *scalars*. Going from scalar to PSD covariance is the next step.

## 4. Scalar → covariance: amplitude × shape

The cleanest bridge satisfying (P3) is a **product decomposition**:
$$\boxed{\Sigma_z(z_B) = s(z_B) \cdot \Phi(z_B)}$$
where:

- $s(z_B) \in \mathbb{R}_+$ is the **amplitude** — a scalar shift descriptor.
- $\Phi(z_B) \in \mathbb{S}^{d_z}_+$ is the **shape** — a PSD matrix encoding directions of uncertainty.

The factorization is principled because it separates two orthogonal questions:
- *How much* uncertainty? → $s$
- *In what directions*? → $\Phi$

Concrete choices for $\Phi$, cheap to expressive:

| $\Phi$ | Rank | Rationale |
|---|---|---|
| $I_{d_z}$ | full | isotropic — loses all geometry |
| $\Sigma_A$ (or $\Sigma_{c^*}$) | full or low-rank | source covariance — aligns with directions $A$ varies on |
| $(z_B - \mu_A)(z_B - \mu_A)^\top$ | 1 | residual outer product — aligns with the specific deviation |
| $\Sigma_A + (z_B - \mu_A)(z_B - \mu_A)^\top$ | full | law of total covariance — combines both |

**Principled default**: $\Phi = \Sigma_A$ (or $\Sigma_{c^*}$ for class-conditional). Justification:

> The encoder $f_\theta$ was optimized to make $A$'s data well-described by some Gaussian-like structure in latent space. The columns of $\Sigma_A$ tell us which directions in $\mathbb{R}^{d_z}$ encode high-variance ("nuisance") features of $A$ vs. low-variance ("tight") features. When $z_B$ is anomalous, we have epistemic uncertainty about it — and the natural prior on that uncertainty is "shaped like $A$", spreading the uncertainty along directions $f_\theta$ already knows are flexible.

This recovers **Option B** in `src/plugins/EKF_README.md` and adds the formal justification: not the only option, but the one with the cleanest geometric interpretation.

## 5. Concrete formal proposal

**Definition (per-sample input-shift covariance)**. For target sample $x_B$ with latent $z_B = f_\theta(x_B)$:
$$\Sigma_z(x_B) \;=\; s(z_B) \cdot \Phi_A(z_B)$$
where:

- $\Phi_A(z_B)$ — **directional shape**, PSD matrix from $A$'s latent structure.
  - Unimodal (toy): $\Phi_A(z_B) = \Sigma_A$ (constant in $z$).
  - Class-conditional: $\Phi_A(z_B) = \Sigma_{c^*(z_B)}$.

- $s(z_B)$ — **amplitude**, non-negative anomaly score with $\mathbb{E}_{z \sim p_A^z}[s(z)] = O(1)$ and $s(z) \to \infty$ as $z$ moves into low-density regions. Canonical choice: $s(z_B) = d^2_M(z_B; A) / d_z = (z_B - \mu_A)^\top \Sigma_A^{-1} (z_B - \mu_A) / d_z$.

**Verification of properties**:

- (P1) In-distribution: $\mathbb{E}_A[d^2_M / d_z] = 1$ (Mahalanobis $\chi^2_{d_z}/d_z$), so $\Sigma_z \approx \Sigma_A$. Descriptor reduces to "$z$'s latent uncertainty is $A$'s typical spread" — clean baseline.
- (P2) OOD: $d^2_M / d_z \gg 1$, so $\Sigma_z$ amplifies proportionally.
- (P3) PSD by construction (non-negative scalar times PSD matrix).

## 6. Three publication settings instantiate the same formalism

**SD (source-dependent)** — oracle:
- Compute $\mu_A, \Sigma_A$ (or $\mu_c, \Sigma_c$) by passing source data $\{x_i^A\}$ through $f_\theta$ and taking sample mean/covariance.
- Low-rank decomposition $\Sigma_A \approx U D U^\top + \sigma_{\mathrm{floor}}^2 I$.

**SF (source-free)** — recover from $M_A$ weights:
- BN running statistics give per-layer $\mu_l, \sigma_l^2$ summary of $A$'s intermediate distribution.
- Last-layer classifier weights $W$ approximate class means: $\mu_c \approx W_c$ up to normalization.
- SWAG-style covariance from training trajectory if checkpoints saved.

**B-only** — no source access:
- Approximate $\mu_A, \Sigma_A$ from a held-out target batch, treating target latents as a proxy for source. Assumes most of $B$ overlaps $A$ — reasonable for moderate shift.
- Or use BN-shift score directly as $s(z)$, with either $\Phi = I$ (current code, weak) or $\Phi = \Sigma_B$ from batch statistics (better).

## 7. Code-level implications

To honor this formalism, three changes are required:

1. **`diag_sigma_z` becomes a per-sample tensor**, not a `register_buffer`. New signature: a `SigmaZProvider` interface `Σ_z : z → (B, d_z, d_z)` (full) or `(B, d_z)` (diagonal). Swappable across SD/SF/B-only implementations.

2. **EKF chain accepts full $\Sigma_z$**, not just diagonal. Propagation $J_f \Sigma_z J_f^\top$ computed exactly. If diagonal is retained for efficiency, document that directional information is being dropped.

3. **$\Sigma_z$ recomputed per batch** at training and eval — depends on $z$.

## 8. Design choices to resolve before implementation

**Choice A — full vs diagonal $\Sigma_z$**:

- **Full** $(B, d_z, d_z)$: honest representation. $O(d_z^2)$ per sample. For $d_z = 32$: 1024 floats/sample — trivial for the toy.
- **Diagonal** $(B, d_z)$: cheap. But $\mathrm{diag}(s \cdot \Sigma_A) \neq s \cdot \mathrm{diag}(\Sigma_A)$ unless $\Sigma_A$ is already diagonal — so diagonal means approximating $\Phi_A$ by its diagonal, losing cross-coordinate covariance.

**Choice B — amplitude function $s$**:

- **Raw Mahalanobis**: $s(z) = d^2_M(z) / d_z$. Mean $\approx 1$ on $A$. Simple, monotone.
- **Softplus-anchored**: $s(z) = \mathrm{softplus}(\alpha \cdot d^2_M(z) + \beta)$ with $\alpha, \beta$ fit on held-out set. Allows nonlinear shaping.
- **Density-based**: $s(z) = -\log p_A^z(z) - \mathrm{baseline}$. Equivalent to Mahalanobis under Gaussian fit; more flexible under richer $p_A^z$.

**Recommended starting point**: full $\Sigma_z = (d^2_M / d_z) \cdot \Sigma_A$ on the toy (Choice A = full, Choice B = raw Mahalanobis). Simplest concrete instance of the formalism; ablate to diagonal / softplus / density-based after baseline is calibrated.

## 9. Open questions for later

1. **Class-conditional vs unimodal for the toy.** The toy has no class structure — just a continuous regression. A single Gaussian on $z \sim A$ is the right approximation. The class-conditional version applies when we move to classification benchmarks. We should make `SigmaZProvider` agnostic to that choice via a "number-of-mixture-components" hyperparameter.

2. **Encoder calibration.** $\Sigma_A$ is only meaningful if the encoder produces a roughly-Gaussian latent. For MLPs with BatchNorm/SiLU this is approximately true on the toy. For pretrained vision encoders (CLIP/DINOv2) it's well-validated by SSD (Sehwag et al. 2021). For domain-specific (medical) encoders this should be verified empirically.

3. **Test-time adaptation of $\Sigma_A$.** Under sustained shift, the source statistics drift. DUA (Mirza et al. 2022) and Tent (Wang et al. 2021) offer adaptation recipes — see `src/plugins/EKF_README.md` §4.

4. **Interaction with the BayesCap NLL head**. The downstream heads $\mathrm{inv}_\alpha$ and $\beta$ currently take EKF outputs $(\sigma^2_{\mathrm{pred}}, \mathrm{diag}(\Sigma_{\mathrm{recon}}))$ as input. After the change, those tensors will be per-sample and carry real shift signal. Whether the heads should also condition on $s(z_B)$ explicitly (in addition to via $\sigma^2_{\mathrm{pred}}$) is an architectural choice.

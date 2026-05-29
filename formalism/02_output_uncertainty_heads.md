---
title: Closed-Form Derivation of α and β for the Output Uncertainty Head
tags: [SURE, theory, bayescap, GGD, EKF, alpha, beta]
created: 2026-05-29
status: draft
context: |
  Addresses the second design question in the EKF pipeline: what should be
  the inputs to `inv_alpha_net` and `beta_net` of `EKFBiModalInferer`?
  Current architecture feeds `concat(σ²_pred, diag(Σ_recon))` to α and
  `SVD(J_f)/max` to β — both choices are heuristic. This note derives a
  closed-form characterization under the linear assumption to identify
  what the principled inputs are.
related: [[01_input_shift_measurement]]
---

# Closed-Form Derivation of α and β for the Output Uncertainty Head

## 1. Setup

The frozen pipeline is $\hat{y} = g(f(z))$, where:

- $z \in \mathbb{R}^{d_z}$ is the latent (per-sample), with associated uncertainty $\Sigma_z(z) \in \mathbb{S}^{d_z}_+$ from the SD-setting provider.
- $f : \mathbb{R}^{d_z} \to \mathbb{R}^{d_z}$ is the reconstructor.
- $g : \mathbb{R}^{d_z} \to \mathbb{R}$ is the predictor head (frozen, scalar output).

We model the predictive distribution as a Generalized Gaussian Distribution (GGD):
$$p(y \mid z) = \frac{\beta(z)}{2\,\alpha(z)\,\Gamma(1/\beta(z))} \exp\!\Bigg(-\!\left|\frac{y-\mu(z)}{\alpha(z)}\right|^{\beta(z)}\Bigg)$$
with $\mu(z) = \hat{y}(z)$ frozen (no learned mean head), and $\alpha(z), \beta(z) > 0$ learned scale and shape.

**Question.** What signals should $\alpha$ and $\beta$ be functions of? The current architecture answers this heuristically; we derive a principled answer by analyzing the linear limit.

## 2. The linear-Gaussian limit

Assume:
- $f(z) = F z + c_f$, with $F \in \mathbb{R}^{d_z \times d_z}$.
- $g(\tilde z) = w^\top \tilde z + c_g$, with $w \in \mathbb{R}^{d_z}$.
- The input uncertainty is Gaussian: $z \sim \mathcal{N}(z_0, \Sigma_z)$.

Then $\hat y(z) = w^\top F z + w^\top c_f + c_g$ is linear in $z$, so the induced output distribution is **exactly Gaussian**:
$$\hat y(z) \sim \mathcal{N}\!\big(\mu_y,\; \sigma^2_{\text{pred}}\big)$$
with
$$\mu_y = w^\top F z_0 + w^\top c_f + c_g, \qquad \sigma^2_{\text{pred}} = w^\top F \,\Sigma_z\, F^\top w = (Fw)^\top \Sigma_z (Fw).$$

## 3. Closed-form α and β under linearity

**β.** A Gaussian has GGD shape $\beta = 2$. So under the linear limit:
$$\boxed{\beta = 2 \text{ (constant)}}$$
No dependence on $z$, $\Sigma_z$, $J_f$, $J_g$, or anything else. Geometry of the Jacobian is *irrelevant* under exact linearity.

**α.** The GGD with shape $\beta=2$ has variance $\sigma^2 = \alpha^2 / 2$ (since the second moment of $\text{GGD}(0,\alpha,\beta)$ is $\alpha^2 \Gamma(3/\beta)/\Gamma(1/\beta)$, evaluated at $\beta=2$ gives $\alpha^2 \cdot \Gamma(3/2)/\Gamma(1/2) = \alpha^2/2$). So:
$$\boxed{\alpha = \sqrt{2\,\sigma^2_{\text{pred}}}}$$
α is fully determined by $\sigma^2_{\text{pred}}$. No other input adds information.

## 4. General GGD moment coupling

For general $\beta$, the second moment of GGD$(0, \alpha, \beta)$ is:
$$\mathbb{E}[y^2] = \alpha^2 \cdot \frac{\Gamma(3/\beta)}{\Gamma(1/\beta)}.$$
Equating to the EKF-propagated variance $\sigma^2_{\text{pred}}$:
$$\boxed{\alpha = \sigma_{\text{pred}} \cdot \sqrt{\frac{\Gamma(1/\beta)}{\Gamma(3/\beta)}}}$$
This is a **deterministic coupling**: once $\sigma_{\text{pred}}$ and $\beta$ are known, $\alpha$ is fixed. It is not an additional learnable degree of freedom.

In particular:
- $\beta = 2$ (Gaussian): $\alpha = \sigma_{\text{pred}} \sqrt{2}$.
- $\beta = 1$ (Laplace): $\alpha = \sigma_{\text{pred}} / \sqrt{2}$.
- $\beta \to \infty$ (uniform): $\alpha \to \sigma_{\text{pred}} \sqrt{3}$.

## 5. Implications for the architecture

### What α should be a function of

In the strict linear case, **only $\sigma^2_{\text{pred}}$**. Equivalently, given the moment-coupling formula, $\alpha$ is a deterministic transform of $\sigma_{\text{pred}}$ once $\beta$ is set. No need for `diag(Σ_recon)`, `Σ_z` structure, etc. — those are all already absorbed into the scalar $\sigma^2_{\text{pred}}$ via the EKF quadratic form.

The current `inv_alpha_net(concat(σ²_pred, diag(Σ_recon)))` is **over-parameterized** by 32 dimensions in this limit. Those extra inputs can only become useful when $f, g$ are nonlinear and the EKF first-order propagation produces a $\sigma^2_{\text{pred}}$ that's *biased* (i.e., it captures the right *order* but the wrong *magnitude*).

### What β should be a function of

Under linearity, **$\beta$ is a constant** (equal to 2). All learning of $\beta$ is therefore implicitly modeling *deviation from linearity* — i.e., higher-order Taylor corrections to the linear-Gaussian story.

The current `beta_net(S_f / max(S_f))` (normalized singular values of $J_f$) is **not a measure of nonlinearity**:

- Under exact linearity, $J_f = F$ is a constant matrix — its SVD is just a constant feature. β has no per-sample variation; the head is wasted.
- Under signal=(1,1) where $J_f = I_{d_z}$, $S_f = (1, \ldots, 1)$ identically, regardless of nonlinearity of the underlying $f$. So the current input cannot capture nonlinearity in the (1,1) regime at all.

The geometry of $J_f$ is a property of the *reconstructor*; β should depend on properties of the *composition* $g \circ f$ near $z$.

### What signals would correctly drive β

Second-order Taylor expansion of $g \circ f$ around $z_0$:
$$g(f(z_0 + \epsilon)) \approx g(f(z_0)) + \nabla(g\!\circ\! f)^\top \epsilon + \tfrac{1}{2}\epsilon^\top \nabla^2(g\!\circ\! f)\, \epsilon$$

The Gaussian-output limit holds only if the quadratic term is negligible compared to the linear term. So **β should depend on a curvature-vs-gradient ratio**. Concrete candidates:

1. **Hessian-weighted curvature score**:
$$\kappa(z) = \frac{\| \nabla^2(g\!\circ\! f) \cdot \Sigma_z \|_F}{\|\nabla(g\!\circ\! f)\|^2 \cdot \mathrm{tr}(\Sigma_z)}$$
Dimensionless; vanishes when $g \circ f$ is locally linear.

2. **Finite-difference proxy** (cheaper, no Hessian):
$$\kappa_{\text{fd}}(z) = \frac{|g(f(z + \delta)) + g(f(z - \delta)) - 2 g(f(z))|}{|g(f(z + \delta)) - g(f(z - \delta))|}$$
for $\delta$ drawn from the principal direction of $\Sigma_z$. Numerator vanishes for linear $g \circ f$.

3. **Mahalanobis-shift surrogate**: under input shift, the operating point of $g \circ f$ leaves the regime where the first-order linearization is calibrated. So $d_M^2(z)$ from the SD-setting provider can act as a coarse stand-in. Heavier-tailed residuals are expected when $z$ is anomalous → $\beta < 2$.

Of these, (3) is cheapest and already computed by `SDSigmaZ`. (1) is the most principled but requires Hessian-vector products. (2) is a compromise.

## 6. Refined architecture proposal

**Option C (current)**: $\alpha$ and $\beta$ are both freely learned from features. Closed-form structure is "lost" — the network must rediscover that $\alpha \propto \sigma_{\text{pred}}$ from data. High variance, weak inductive bias.

**Option B (recommended): closed-form baseline + learned correction**.

$$\alpha(z) = \sigma_{\text{pred}}(z) \cdot \sqrt{\frac{\Gamma(1/\beta)}{\Gamma(3/\beta)}} \cdot \exp\!\big(\Delta_\alpha(z)\big)$$

$$\beta(z) = 2 + \Delta_\beta\!\big(\text{nonlinearity features}\big)$$

where:

- $\sigma_{\text{pred}}(z)$ comes from the EKF chain — *not* learned, computed.
- $\Delta_\alpha(z)$ is a small learned scalar correction (default 0, expressed in log-space for positivity). Input: $\sigma^2_{\text{pred}}$ and optionally an anomaly score $d_M^2$ for cross-sample heteroscedasticity.
- $\Delta_\beta$ is a learned correction to the Gaussian baseline. Input: nonlinearity features (one of the three candidates above). Constrained so $\beta(z) > 0$ (e.g., $\beta = \mathrm{softplus}(2 + \Delta_\beta)$).

This keeps the closed-form physics as the prior and lets the network repair calibration errors without re-deriving the entire relationship from data.

**Option A (purest)**: drop $\Delta_\alpha$ entirely; use the closed-form $\alpha = \sigma_{\text{pred}} \sqrt{\Gamma(1/\beta)/\Gamma(3/\beta)}$ exactly. Only learn $\beta$. Best for ablations / theoretical baseline.

## 7. Concrete input recommendations

| Head | Current input | Proposed input | Justification |
|---|---|---|---|
| `inv_alpha_net` | `concat(σ²_pred, diag(Σ_recon))` ∈ ℝ³³ | `log(σ²_pred)` ∈ ℝ¹ (Option A) or `[log(σ²_pred), d²_M]` ∈ ℝ² (Option B) | Under linearity, $\alpha$ depends only on $\sigma_{\text{pred}}$. Log scale stabilizes optimization across many orders of magnitude. $d_M^2$ adds a per-sample heteroscedastic correction. |
| `beta_net` | `S_f / max(S_f)` ∈ ℝ^{d_z} (constant when J_f=I) | `[κ(z)]` ∈ ℝ¹ or `[d²_M, ‖J_g‖, …]` ∈ ℝ² (curvature surrogates) | Under linearity, $\beta$ is constant. Learning $\beta$ models nonlinearity; SVD of $J_f$ does not. Anomaly score / gradient magnitude are cheap surrogates. |

## 8. Architecture-level changes

These are notes for implementation, separate from this derivation:

1. **Output activation** of both heads: replace `ReLU` with `softplus` for numerical stability (no dead neurons at init, smooth gradient near zero). For $\beta$, initialize the bias such that $\mathrm{softplus}(b) \approx 2$ at start (i.e., $b \approx \log(e^2 - 1) \approx 1.85$). For $\alpha$, init around 1.

2. **Closed-form coupling layer**: instead of having `inv_alpha_net` learn the full mapping, factor it as
   $$\alpha = \alpha_{\text{closed}}(\sigma_{\text{pred}}, \beta) \cdot \exp(\Delta_\alpha)$$
   with $\Delta_\alpha$ from the small MLP. The closed-form factor is a single line of code.

3. **Coupling order**: $\beta$ must be computed *before* $\alpha$, since $\alpha$'s closed-form depends on $\beta$. Sequential forward pass through the inferer.

4. **Loss compatibility**: BayesCap loss expects `inv_alpha`, which is just $1/\alpha$. Compute $\alpha$ via the closed form, then take reciprocal. Or have the inferer expose both.

## 9. Why this is more than cosmetic

In the linear-Gaussian case the EKF chain *already* gives a fully-calibrated $\sigma_{\text{pred}}$, and the only thing left to do is convert that variance into GGD parameters via a fixed formula. The current architecture, with its overparameterized $\alpha$ head and its irrelevant $\beta$ input, can in principle "rediscover" this — but it has no inductive bias toward doing so, and it can equally well *break* the closed-form relationship by absorbing model bias into $\alpha$.

Empirically, this matters most when:
- Input shift is large → $\sigma_{\text{pred}}$ becomes a strong signal that should map almost mechanically to $\alpha$.
- Residual distribution is non-Gaussian → $\beta$ should adapt, but the current input can't tell whether it's encountering non-Gaussian residuals.

Under signal=(1,1) (current operating regime), the issue is starker: $J_f = I$ means $S_f$ is constant, so the entire $\beta$ pathway is silent. With this refactor, $\beta$ has a meaningful per-sample signal (via $d_M^2$ or curvature surrogates) even when the reconstructor is bypassed.

## 10. Open questions

1. **Hessian access**: do we want true $\nabla^2(g\!\circ\!f)$ via double-backward, or is the finite-difference proxy sufficient? Affects compute budget.

2. **Is $\beta$ identifiable per-sample with this little supervision?** GGD with both $\alpha$ and $\beta$ free is known to be hard to fit on small batches. May need to start with $\beta = 2$ fixed (Option A) and only relax it once the rest of the pipeline is calibrated.

3. **Interaction with the BayesCap `nll_mode = "repo"` setting**, which uses a simplified NLL form $(|\mu-y| \cdot \mathrm{inv}_\alpha \cdot \beta)$ instead of the exponentiated $(|\mu-y|/\alpha)^\beta$. The closed-form coupling derived here is for the *paper* form. We should either switch to `nll_mode = "paper"` to make the derivation match, or re-derive the coupling under the simplified form.

4. **Calibration on the toy**: under the new SD-setting Σ_z and the closed-form α, the linear-Gaussian limit predicts a specific relationship between $\sigma_{\text{pred}}$ and squared residual. A direct check on the toy dataset: $\mathbb{E}[(y - \hat y)^2 \mid \sigma_{\text{pred}}]$ should be $\approx \sigma^2_{\text{pred}}$ on in-distribution data. If it's not, the linear-Gaussian assumption is materially violated and we need the curvature-driven $\beta$.

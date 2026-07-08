---
title: Phase 1 Limits — Encoder Folding and Jacobian Collapse; Phase 2 Direction
tags: [SURE, theory, sigma_z, EKF, OOD, BN-shift, second-order]
created: 2026-05-29
status: draft
context: |
  Empirical follow-up to docs 01 and 02. The Phase 1 pipeline — SD-setting
  Sigma_z, closed-form alpha-beta coupling, bounded GGD heads, paper-form
  NLL — now produces calibrated in-distribution variance (corr 0.52 with
  squared residual). It fails to grow under shift: corr(amp, sigma_pred) is
  ~ -0.3 and the OOD/in-dist variance ratio sits below 1. This note
  formalizes the two structural reasons why, and lays out the Phase 2 fixes.
related: [[01_input_shift_measurement]] [[02_output_uncertainty_heads]]
---

# Phase 1 Limits and Phase 2 Direction

## 1. What Phase 1 confirmed

The pipeline as currently implemented:

1. **Per-sample SD Sigma_z** — $\Sigma_z(z) = (d_M^2(z) / d_z) \cdot \Sigma_A$ with $\Sigma_A$ fit from source samples, shrunk and eigen-clamped (doc 01).
2. **Full-covariance EKF** — $\sigma^2_{\text{pred}} = J_g^\top (J_f \Sigma_z J_f^\top) J_g$.
3. **Bounded GGD heads** — `inv_alpha = softplus(...) + eps`, `beta = beta_min + (beta_max - beta_min) * sigmoid(...)`, log-space features (doc 02 §6).
4. **Paper-form GGD NLL** — $\mathcal{L} = (|r| \cdot \mathrm{inv}_\alpha)^\beta - \log \beta - \log \mathrm{inv}_\alpha + \log \Gamma(1/\beta)$.

Empirical confirmation of the formal claims:

| Claim | Doc | Status |
|---|---|---|
| In-distribution: $\Sigma_z \approx \Sigma_A$ on average | 01 §5 | confirmed; `corr(amp, sigma_pred) > 0` in-dist |
| Moment coupling pins $\alpha^2 \cdot \Gamma(3/\beta)/\Gamma(1/\beta) \to \mathbb{E}[r^2]$ | 02 §4 | confirmed; `corr(plotted_var, r^2) = 0.52` |
| Linear-Gaussian limit gives $\beta \approx 2$ | 02 §3 | $\beta$ stable in [0.5, 4] |
| GGD NLL provides MLE restoring force | 02 §6 | confirmed; switching `repo → paper` stopped variance inflation |

**Phase 1 works for in-distribution calibration.** That is a publishable result on its own.

## 2. The unfinished promise — OOD growth

The point of constructing $\Sigma_z$ as Mahalanobis-amplified was to make $\sigma^2_{\text{pred}}$ grow when $z_B$ is far from $A$. Empirically it does not. Measured on the toy with $x_1 \in [-0.6, 1.4]$ vs source $x_1 \in [-1, 1]$:

- `amp` (the Mahalanobis amplitude on latent $z$) barely exceeds 3 even on samples with $x_1 > 1$.
- `corr(amp, sigma_pred) ≈ -0.3` — the *more* anomalous a sample, the *smaller* its propagated variance.
- OOD/in-dist variance ratio < 1.

Two structural reasons, each provable from the formalism. Neither is an implementation bug.

## 3. Failure mode 1 — Encoder folding

### Statement

A frozen encoder $f_\theta : \mathbb{R}^{d_x} \to \mathbb{R}^{d_z}$ trained on $A$ has no incentive to preserve OOD displacement in its output. Under typical inductive biases of MLPs with BN and bounded activations, $f_\theta$ *contracts* OOD inputs toward the support of $f_\theta(A)$ in latent space.

Formally: let $\pi_A^z$ and $\pi_B^z$ be the push-forward distributions of $A$ and $B$ under $f_\theta$. Then
$$d_W(\pi_A^z, \pi_B^z) \;\leq\; \mathrm{Lip}(f_\theta) \cdot d_W(A, B)$$
which is a trivial upper bound. But for an *MLP with BN on every layer trained against $A$*, the effective Lipschitz constant *in the direction $B - A$* can be very small — far below the global Lipschitz constant. The encoder "folds" OOD inputs back to the support of $\pi_A^z$.

### Empirical signature

On our toy:
- Input shift $\|x_B - x_A\|$ ranges up to 0.4 (when $x_1 = 1.4$ vs A's edge at 1.0).
- Latent shift $\|z_B - \mu_A\|$ for those same samples lands inside A's typical $z$-cloud — Mahalanobis $d_M^2 \in [1, 3]$ rather than $\gg 1$.

The shift didn't survive into latent space.

### Implication

Any **latent-space** OOD detector — Mahalanobis on $z$, energy on $\log p_A^z$, k-NN distance in $z$ — has an irreducible blind spot for shift that the encoder erases. This is not a hyperparameter to tune; it is a property of $f_\theta$.

The fix has to act either (a) **before** the encoder, in input space, or (b) **inside** the encoder, at intermediate activations where the folding has not yet completed.

## 4. Failure mode 2 — First-order Jacobian collapse

### Statement

The EKF propagation $\sigma^2_{\text{pred}} = J_g(z)^\top \Sigma_z(z) J_g(z)$ is a first-order linearization of $g$ around $z$. For frozen MLP regression heads, the gradient $\|J_g(z)\|$ tends to *vanish* outside the training support — the head extrapolates with locally constant value, so its slope flattens.

Precisely: bound the propagated variance by the spectral norm
$$\sigma^2_{\text{pred}} \;\leq\; \|J_g(z)\|^2 \cdot \sigma_{\max}(\Sigma_z).$$
A frozen MLP head has $\|J_g(z)\| \to 0$ as $z$ moves into regions where the head is locally constant. Even if $\Sigma_z$ correctly grows with shift, the product can stay small or decrease.

### Empirical signature

In the toy run, samples flagged as anomalous (high $d_M^2$) often have *lower* $\sigma^2_{\text{pred}}$ than in-distribution samples. The correlation $\mathrm{corr}(d_M^2, \sigma^2_{\text{pred}})$ is negative.

This is consistent with: high $d_M^2$ → OOD region → head plateaus → $\|J_g\|$ small → $\sigma^2_{\text{pred}}$ shrinks. The propagator suppresses uncertainty exactly where epistemic uncertainty should rise.

### Implication

A first-order EKF on a frozen head is *anti-conservative* in OOD regions. The fix has to add a term that does not vanish when $J_g \to 0$. Two candidates:

- **Second-order Taylor**: $\sigma^2_{\text{pred}}$ picks up a curvature term $\tfrac{1}{2} \mathrm{tr}((H_g \Sigma_z)^2)$ which does not depend on $J_g$.
- **Ensemble / sampling-based propagation**: sample $z \sim \mathcal{N}(z_0, \Sigma_z)$, push through $g$, take empirical variance. Captures non-linearity exactly, at cost of $K$ extra forward passes.

## 5. Why these failures are predicted, not accidents

Both are direct consequences of two architectural choices that SURE makes by design:

| Choice | Cost |
|---|---|
| Use a *frozen* $f_\theta$ trained without uncertainty awareness | encoder folding (§3) |
| Approximate output variance by *first-order* Jacobian propagation | OOD Jacobian collapse (§4) |

Neither choice is wrong — they are exactly what the post-hoc, no-retraining setting forces. But they have predictable failure modes, and Phase 2 has to address each independently.

## 6. Phase 2 direction A — Input-space / intermediate-layer shift detection

**Proposal**: BatchNorm shift score (`BNShiftSigmaZ`, already implemented in `src/plugins/sigma_z.py` but not yet wired).

$$s(z) \;=\; \frac{1}{L} \sum_{l=1}^{L} \frac{1}{d_l} \sum_{i=1}^{d_l} \frac{(h^l_i - \mu^l_{\mathrm{BN},i})^2}{\sigma^{l,2}_{\mathrm{BN},i}}$$
computed over the encoder's BN layers, where $h^l$ is the layer-$l$ activation for the current sample and $\mu^l_{\mathrm{BN}}, \sigma^{l,2}_{\mathrm{BN}}$ are the BN running statistics estimated on $A$.

### Why this beats latent Mahalanobis

The BN-shift score samples the encoder at *every layer*, not just the output. If $f_\theta$'s folding happens in layers $l^\star+1, \ldots, L$, then layers $1, \ldots, l^\star$ still carry shift signal. The score is the average over all layers, so it picks up shift wherever in the depth it lives.

For a pretrained encoder with BN at the input or early in the stack — which is most encoders for medical imaging, CLIP, DINOv2 — the early layers are very close to input space. Their BN statistics directly reflect input-distribution properties. Shift becomes detectable.

### Setup as a Sigma_z provider

The amplitude × shape decomposition (doc 01 §4) still applies:
$$\Sigma_z(z) = s(z) \cdot \Phi(z).$$
Two natural choices for $\Phi$:
- $\Phi = I$ — pure B-only, isotropic. Lead method per [[project_sure_publication]].
- $\Phi = \Sigma_B$ — empirical target covariance, anisotropic. Hybrid with target-only statistics.

In neither case do we need $A$'s source data at deployment. The BN running statistics live inside the frozen model's weights — already there.

## 7. Phase 2 direction B — Second-order EKF or sampling propagation

**Proposal**: replace
$$\sigma^2_{\text{pred}}\;=\;J_g^\top \Sigma_z J_g$$
with one of:

### Option B1 — Second-order Taylor

$$\sigma^2_{\text{pred}} \;=\; J_g^\top \Sigma_z J_g \;+\; \tfrac{1}{2}\, \mathrm{tr}\!\big((H_g \Sigma_z)^2\big)$$

The second term is the leading curvature correction. It does not vanish when $J_g = 0$. Cost: a Hessian $H_g \in \mathbb{R}^{d_z \times d_z}$ per sample, computed by double-backward. For $d_z = 32$ this is direct; for larger latents use Hutchinson trace estimation.

### Option B2 — Unscented / sigma-point propagation

Sample deterministic sigma-points $\{z_k\}$ around $z_0$ matching $\mathcal{N}(z_0, \Sigma_z)$ to second order (the unscented transform). Propagate each through $g$, take empirical mean and variance. Cost: $2 d_z + 1 = 65$ forward passes per sample. Captures full non-linearity, not just second order.

### Option B3 — Monte Carlo dropout

Re-enable dropout in the frozen head at inference, sample $K$ forward passes per input, take empirical variance. Cost: $K \approx 20$ forwards. Requires the head to have dropout (it does — `dropout: 0.5` in `MLP`).

### Recommendation

B3 is cheapest to ship and tests the OOD-growth hypothesis fastest. B1 is the most principled. B2 is the middle ground. Start with B3 as a diagnostic. If it confirms that adding *any* non-vanishing OOD term recovers growth, move to B1 as the published method.

## 8. Combining 6 and 7

The two fixes are independent and compose. Two cases:

| BN-shift fixes detection? | 2nd-order fixes propagation? | Outcome |
|---|---|---|
| no | no | Phase 1 status quo — calibrated in-dist, no OOD growth |
| yes | no | Detection works; propagation still suppresses growth via $J_g \to 0$. Partial recovery only. |
| no | yes | Detection blind; propagation has non-vanishing term but no shift signal feeding it. Marginal recovery. |
| yes | yes | Detection picks up shift via early-layer BN stats; propagation propagates it even where $J_g = 0$. Full recovery. |

The B-only paper story needs *both*. They're not redundant.

## 9. Publication framing

Single paper, three claims:

1. **In-distribution calibration without retraining $M_A$.** SD-setting $\Sigma_z$ with closed-form $\alpha$-$\beta$ coupling and paper-form GGD NLL — `corr(var, r^2) = 0.52` on the toy. Foundation for everything else.

2. **Latent-space SD detection is blind to encoder-folded shift.** Document the empirical failure: $d_M^2$ stays $\approx 1$ on inputs that are demonstrably OOD. Introduce BN-shift score as an input-space / intermediate-layer alternative. **This is the lead contribution** ([[project_sure_publication]]) — B-only setting, most operationally restrictive, biggest novelty.

3. **First-order EKF collapses OOD; second-order or sampling propagation restores growth.** Document the Jacobian-collapse empirical signature. Show that B1/B2/B3 recover OOD/in-dist variance ratio > 1.

The two failure modes are positioned as *motivations*, not as negative results — each motivates one of the two Phase 2 contributions. The Phase 1 calibration result is what lets us claim the rest works.

## 10. Code task list for next week

Captured here rather than in code to keep diff small until we sit down together:

- [ ] **Wire `BNShiftSigmaZ` as a swappable Sigma_z provider.** Add a `type` field to `configs/plugins/ekf.yaml` (`"sd"` vs `"bn"`); `ModelEKFInjectModule.__init__` branches on it. Both providers should produce $(B, d_z, d_z)$ tensors; for `BNShiftSigmaZ` with $\Phi = I$, this is $s(z) \cdot I$ per sample.
- [ ] **Side-by-side comparison run.** Single training, two providers, log `corr(amp_SD, amp_BN)`, OOD/in-dist ratio for each. Decide whether the agreement is high (BN is a cheap surrogate) or low (they capture different things and we use both).
- [ ] **Add OOD-stress evaluation pass.** A held-out test set with $x_1 \in [1.0, 2.0]$ (strictly OOD), log `mean(sigma_pred_sq)` at $\delta = 0, 0.25, 0.5, 1.0$ shift. The growth curve is the headline result for claim 2.
- [ ] **Implement Option B3 (MC dropout) as a diagnostic.** One toggle in `EKFBiModalInferer.forward` — re-enable head dropout, $K = 20$ forwards, take empirical variance. Compare to first-order EKF variance on the same samples. If MC dropout recovers OOD growth, the Jacobian-collapse hypothesis is confirmed and we proceed to B1.
- [ ] **Implement Option B1 (second-order Taylor).** Add Hessian-vector products via double-backward; for $d_z = 32$ store full Hessian, for larger use Hutchinson. New helper in `ekf_propagation.py`.
- [ ] **Decide $\Phi$ for the B-only setting.** $\Phi = I$ (pure isotropic) vs $\Phi = \Sigma_B$ (target covariance, computed from a held-out target batch). Affects whether the paper claims "no statistics needed at all" or "only target-batch statistics needed".

## 11. Open theoretical questions

1. **Is encoder folding generic, or specific to MLPs with BN?** Worth testing whether transformers or convnets with LayerNorm exhibit the same property. If transformers do *not* fold (preserve more directional information), the latent-space SD approach may be salvageable for VLM/medical-VLM settings (per [[user-profile]]).

2. **What is the relationship between the second-order term and the variance of dropout ensembles?** Both are "non-vanishing under flat Jacobian". A theoretical equivalence (or its absence) would be a clean diagnostic for choosing between B1 and B3.

3. **Does the BN-shift score's depth-averaging hide the layer at which folding happens?** Worth ablating: BN-shift from layer 1 only, layer L only, weighted average. If the early layers carry all the signal, the cheap version of the score is "BN-shift on the first BN layer", and the rest is unnecessary cost.

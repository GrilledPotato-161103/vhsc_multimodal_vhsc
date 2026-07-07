---
title: Full Uncertainty Pipeline — Available Shift, Reconstruction, Fusion, Prediction
tags: [SURE, missing-modality, EKF, reconstruction, fusion, aleatoric, epistemic]
created: 2026-06-22
status: design
context: |
  Generalizes the cf2 (all-modalities-present) pipeline to the missing-modality
  case. Three stages: (1) input-shift on AVAILABLE modalities only, (2) reconstruct
  missing modalities + their uncertainty, (3) propagate through fusion + head to a
  predictive variance that decomposes into model (aleatoric) + propagated (epistemic).
  Maps each stage to current code: what exists, what is the gap.
related: [[01_input_shift_measurement]] [[07_aleatoric_head_and_calibration]]
---

# Full Pipeline for Missing Modality

## 0. Notation

- Modalities $m \in \{1, 2\}$ (general: any set). Availability signal $(p_1, p_2)$, $p_m=1$ present.
- Available set $A = \{m : p_m=1\}$, missing set $M = \{m : p_m=0\}$.
- $z_m = f_m(x_m) \in \mathbb{R}^{16}$ frozen per-modality encoders.
- $r_{A\to m'}$ reconstructor for missing $m'$ from available (e.g. $\mathrm{ln12}: z_1 \to \hat z_2$).
- $g$ frozen fusion+head; cf2 fusion = concat.

cf2 is the special case $A=\{1,2\}, M=\varnothing$, where stages 2–3 degenerate ($J_f=I$).

---

## Stage 1 — Input shift on AVAILABLE modalities only

For each available $m \in A$, a per-modality shift amplitude (cycle score):
$$s_m(x_m) = \frac{\lVert x_m - d_m(z_m)\rVert^2}{b_m}$$
and per-modality input covariance
$$\Sigma_z^{(m)} = s_m \cdot \Phi_m, \qquad \Phi_m \in \{I,\ \Sigma_A^{(m)}\}$$

The **available-block** covariance is block-diagonal over $A$:
$$\Sigma_{\text{avail}} = \mathrm{blockdiag}\big(\{\Sigma_z^{(m)}\}_{m\in A}\big)$$

**Key change vs cf2**: the cycle score must be evaluated *only on available modalities*. The
current `CycleSigmaZ.forward(z, x1, x2)` is signal-blind and sums both terms — correct for
cf2 ($A=\{1,2\}$), wrong for real missing data (the absent term is uncomputable: no $x_{m'}$,
and $z_{m'}$ would be a reconstruction).

**Code status**: cycle score exists; per-modality / signal-aware gating is the GAP.
Provider needs a `signal` argument; drop the absent modality's term.

---

## Stage 2 — Reconstruct missing modalities + their uncertainty

For missing $m' \in M$: $\hat z_{m'} = r_{A\to m'}(z_A)$. Its uncertainty has **two parts**:

**(i) Propagated** — the available input uncertainty pushed through the reconstruction map:
$$\Sigma^{\text{prop}}_{\hat z_{m'}} = J_r\,\Sigma_{\text{avail}}\,J_r^\top, \qquad J_r = \partial r_{A\to m'}/\partial z_A$$

**(ii) Intrinsic** — the reconstructor is imperfect even at zero input noise ($r(z_A)\neq z_{m'}^{\text{true}}$):
$$\Sigma^{\text{intr}}_{\hat z_{m'}} = \Sigma_{\text{net,recon}}^{(m')}$$
estimated by a learned reconstruction-error model (the `dev1/dev2` heads regress per-coord
recon error `dist`).

Total reconstructed-block covariance: $\Sigma_{\hat z_{m'}} = \Sigma^{\text{prop}} + \Sigma^{\text{intr}}$.

### Unifying form (this is the EKF predict step)

Stages 1–2 combine into a single expression. Let $J_f$ be the full reconstructor-map
Jacobian (identity on available coords, $J_r$ rows on reconstructed coords) and
$\Sigma_{\text{net}} = \mathrm{blockdiag}(0 \text{ on } A,\ \Sigma_{\text{net,recon}} \text{ on } M)$. Then:
$$\boxed{\ \Sigma_{\text{recon}} = J_f\,\Sigma_{\text{avail}}\,J_f^\top + \Sigma_{\text{net}}\ }$$
This is exactly the EKF predict equation. The first term carries propagated shift (and the
cross-covariance between available and reconstructed coords, since reconstruction is a
deterministic function of the available block); the second adds intrinsic reconstruction noise.

For signal $(0,1)$ (x2 missing), $J_f=\begin{bmatrix}I&0\\ J_{\mathrm{ln12}}&0\end{bmatrix}$, and
$$\Sigma_{\text{recon}} = \begin{bmatrix} \Sigma_z^{(1)} & \Sigma_z^{(1)}J_{\mathrm{ln12}}^\top \\ J_{\mathrm{ln12}}\Sigma_z^{(1)} & J_{\mathrm{ln12}}\Sigma_z^{(1)}J_{\mathrm{ln12}}^\top + \Sigma_{\text{net,recon}}^{(2)} \end{bmatrix}$$

**Code status**: $J_f \Sigma_z J_f^\top$ is implemented (`full_ekf_propagation_*`, $J_f$ per-signal,
zeros the unobserved column correctly). The additive $\Sigma_{\text{net}}$ is the GAP — currently 0.
The `dev` heads that should populate it exist but are inert under $(1,1)$ and not wired into the
EKF additive term.

---

## Stage 3 — Fusion + prediction head propagation

**Fusion** (concat in cf2): a linear map, Jacobian is a selection/identity, so it passes
$\Sigma_{\text{recon}}$ through unchanged into the head input. (A learned fusion would contribute
its own Jacobian block — future.)

**Prediction head** $g$ (first + second order, as cf2):
$$\sigma^2_{\text{prop}} = J_g^\top \Sigma_{\text{recon}} J_g \;+\; \tfrac{1}{2}\,\mathrm{tr}\!\big((H_g\Sigma_{\text{recon}})^2\big)$$
$J_g = \nabla g$, $H_g = \nabla^2 g$ at the (possibly reconstructed) head input.

**Code status**: implemented (`full_ekf_propagation_second_order`).

---

## Final — two quantities

$$\boxed{\ \sigma^2_{\text{total}} = \underbrace{\sigma^2_{\text{model}}}_{\text{aleatoric, learned}} + \underbrace{\sigma^2_{\text{prop}}}_{\text{epistemic, propagated}}\ }$$

- $\sigma^2_{\text{model}}$ — intrinsic prediction error the frozen head makes regardless of input
  uncertainty (function complexity). The `AleatoricHead`.
- $\sigma^2_{\text{prop}}$ — everything propagated: available shift + reconstruction (prop+intrinsic),
  through fusion and head. The EKF chain.

Then $\alpha = \sqrt{\sigma^2_{\text{total}}}$, $\beta=2$ feed the closed-form GGD; $\mathrm{var}=\sigma^2_{\text{total}}$.

**Code status**: implemented (cf2). $\sigma^2_{\text{model}}$ = aleatoric head, $\sigma^2_{\text{prop}}$ = EKF,
combined as $\sigma^2_{\text{ep}} + \lambda\sigma^2_{\text{al}}$, var$=$total via closed_form.

---

## Summary: done vs gap

| Stage | Quantity | Code status |
|---|---|---|
| 1 | per-available cycle shift $\Sigma_{\text{avail}}$ | score exists; **signal-aware gating = GAP** |
| 2(i) | propagated recon $J_r\Sigma J_r^\top$ | implemented (per-signal $J_f$) |
| 2(ii) | intrinsic recon $\Sigma_{\text{net,recon}}$ | **GAP** (=0; dev-heads exist, unwired) |
| 3 | fusion + head propagation (1st+2nd order) | implemented |
| final | $\sigma^2_{\text{model}} + \sigma^2_{\text{prop}}$ | implemented (cf2) |

## Concrete implementation deltas for missing modality

1. **Signal-aware cycle** (`CycleSigmaZ.forward(z, x1, x2, signal)`): zero the absent modality's
   reconstruction-error term; build $\Sigma_{\text{avail}}$ as a block over available modalities only.
2. **Wire $\Sigma_{\text{net,recon}}$**: feed the `dev`-head reconstruction-error estimate into the EKF
   as the additive $\Sigma_{\text{net}}$ block on reconstructed coordinates.
3. **Turn on `mask_rate>0`** and re-validate: missing-modality samples should show larger
   $\sigma^2_{\text{prop}}$ (from $J_{\mathrm{ln12}}\Sigma J^\top + \Sigma_{\text{net,recon}}$) than fully-available ones.
4. **Two-phase gating** already present (`epoch_phase`): skip deficit-input uncertainty loss early.

## Why this is the right shape

The three stages cleanly separate *what is observed* (stage 1: shift on what we have), *what is
inferred* (stage 2: reconstruction and its compounded uncertainty), and *how it reaches the output*
(stage 3: propagation). The final split — model vs propagated — is the standard aleatoric/epistemic
decomposition, but here the epistemic term is **structurally derived** (EKF over a known
reconstruction+fusion+head) rather than sampled. Missing modality is the regime where the
reconstruction Jacobian $J_f\neq I$ and the intrinsic term $\Sigma_{\text{net,recon}}$ both switch on —
the parts that are dormant in cf2.

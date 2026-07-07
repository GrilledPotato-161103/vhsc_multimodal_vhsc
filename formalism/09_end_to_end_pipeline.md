---
title: SURE End-to-End Pipeline — Master Reference
tags: [SURE, pipeline, cycle, reconstruction, EKF, aleatoric, calibration, missing-modality]
created: 2026-06-29
status: living
context: |
  Single consolidated reference for the full post-hoc uncertainty pipeline as
  implemented on the toy BiModalRegressor: input -> per-modality shift (cycle) ->
  reconstruction of missing modalities -> EKF propagation (1st/2nd order) through
  fusion + frozen head -> combine with a learned aleatoric head -> closed-form GGD
  NLL. States the best configurations achieved and the concrete next steps.
  Supersedes/summarizes docs 01-08 for day-to-day reference.
related: [[01_input_shift_measurement]] [[05_untying_latent_collapse]]
         [[06_empirical_validation]] [[07_aleatoric_head_and_calibration]]
         [[08_full_pipeline_missing_modality]]
---

# SURE End-to-End Pipeline

## 0. Problem and setup

A model $M_A$ (two frozen encoders $f_1, f_2$ + frozen fusion/head $g$) is trained on
source domain $A$, then deployed on target $B$ (distribution shift, possibly a missing
modality). Goal: **post-hoc** calibrated predictive uncertainty — turn the deterministic
$\hat y = g(f_1(x_1), f_2(x_2))$ into $p(y\mid x)$ **without retraining $M_A$**.

Toy: $x_1, x_2 \sim \mathcal U$; $y = \big((x_1^2 - x_2^2)/(|x_1|+|x_2|+0.1)\big)\cdot|1-\sqrt{x_1^2+x_2^2}|$.
$z_m = f_m(x_m)\in\mathbb R^{16}$, $z=[z_1,z_2]\in\mathbb R^{32}$, $\hat y = g(z)\in\mathbb R$.

The predictive variance decomposes into two sources:
$$\sigma^2_{\text{total}} = \underbrace{\sigma^2_{\text{model}}}_{\text{aleatoric (learned)}} + \underbrace{\sigma^2_{\text{prop}}}_{\text{epistemic (EKF-propagated)}}$$

---

## 1. Stage 1 — Per-modality input shift (cycle score)

### Why cycle-consistency (not latent Mahalanobis)

Naively, "how OOD is this input" could be a Mahalanobis distance of $z$ to the source
latent Gaussian $\mathcal N(\mu_A,\Sigma_A)$. **This fails**: the encoder, trained only to
predict $y$, *folds* OOD inputs back into the in-distribution latent cloud — it has no
incentive to keep $x$ separable once past the training range. Empirically the latent
Mahalanobis amplitude barely reaches 1–3 on genuinely-OOD inputs (doc 03 §3, doc 05).

The fix is to measure shift in **input space** by inverting the encoder. Train a small
decoder $d_m: Z\to X$ per modality on source pairs $(z_m, x_m)$ (once, at init, frozen).
The **cycle score** is the input-space reconstruction error:
$$s_m(x_m) = \frac{\lVert x_m - d_m(z_m)\rVert^2}{b_m}, \qquad b_m = \mathbb E_{\text{source}}\big[\lVert x_m - d_m(z_m)\rVert^2\big]$$

In-distribution: decoder reconstructs well → $s_m\approx 1$ (normalized). OOD: the encoder
folded $x$ to some $z\approx f_m(x')$ for a source $x'$, the decoder maps back to $x'$, and
$s_m \approx \lVert x - x'\rVert^2/b_m$ — the input-space gap the encoder concealed. This
**bypasses encoder folding** because the decoder exposes what the forward pass compressed.

Empirically cycle beat every latent-space detector by 3+ orders of magnitude on OOD
sensitivity (doc 06): cycle+2nd-order gave OOD/ID variance ratio ~$10^4$–$10^5$ vs 1.07× for
latent Mahalanobis + first-order.

### Signal-aware: available modalities only

The cycle score is computed **only for available modalities**. Availability follows the
reconstructor convention (`ln12`/`ln21`): modality 1 available iff `signal[1]==1`, modality
2 available iff `signal[0]==1`. A missing modality contributes **zero input variance** — its
latent is going to be replaced by a reconstruction, and the EKF Jacobian zeros its input
column anyway, so 0 is exact.

### Shape → covariance

Per-coordinate diagonal scale $\text{scale}=[s_1\cdot\mathbf 1_{16},\ s_2\cdot\mathbf 1_{16}]$
(missing modality's block = 0). Then
$$\Sigma_{\text{avail}} = D^{1/2}\,\Phi\,D^{1/2}, \quad D=\mathrm{diag}(\text{scale})$$
For $\Phi=I$ (cycle_iso, our SOTA) this is $\mathrm{diag}(\text{scale})$; for full $\Phi=\Sigma_A$
it scales row $i$/col $j$ by $\sqrt{s_i}\sqrt{s_j}$ (stays PSD). Code: `CycleSigmaZ` in
`src/plugins/sigma_z.py`.

---

## 2. Stage 2 — Reconstruct missing modalities + their uncertainty

If modality $m'$ is missing, reconstruct $\hat z_{m'} = r_{A\to m'}(z_A)$ (e.g.
$\hat z_2 = \mathrm{ln12}(z_1)$). Its uncertainty has two parts:

**(i) Propagated** — available input uncertainty pushed through the reconstruction Jacobian:
$$\Sigma^{\text{prop}}_{\hat z_{m'}} = J_r\,\Sigma_{\text{avail}}\,J_r^\top, \quad J_r=\partial r_{A\to m'}/\partial z_A$$

**(ii) Intrinsic** — the reconstructor is imperfect even at zero input noise
($r(z_A)\neq z_{m'}^{\text{true}}$): an additive $\Sigma_{\text{net,recon}}$, estimated by the
`dev` heads (they regress the per-coordinate reconstruction error).

### Unifying form — this is the EKF predict step

Let $J_f$ be the full reconstructor-map Jacobian (identity on available coords, $J_r$ on
reconstructed) and $\Sigma_{\text{net}}=\mathrm{blockdiag}(0\text{ on }A,\ \Sigma_{\text{net,recon}}\text{ on }M)$. Then stages 1–2 collapse to:
$$\boxed{\ \Sigma_{\text{recon}} = J_f\,\Sigma_{\text{avail}}\,J_f^\top + \Sigma_{\text{net}}\ }$$

For $(1,1)$: $J_f=I$, $\Sigma_{\text{net}}=0$ → $\Sigma_{\text{recon}}=\Sigma_{\text{avail}}$
(reconstruction dormant). For missing modality 2, signal $(0,1)$:
$J_f=\begin{bmatrix}I&0\\ J_{\mathrm{ln12}}&0\end{bmatrix}$ and
$$\Sigma_{\text{recon}} = \begin{bmatrix}\Sigma_z^{(1)} & \Sigma_z^{(1)}J_{\mathrm{ln12}}^\top\\ J_{\mathrm{ln12}}\Sigma_z^{(1)} & J_{\mathrm{ln12}}\Sigma_z^{(1)}J_{\mathrm{ln12}}^\top + \Sigma_{\text{net,recon}}^{(2)}\end{bmatrix}$$
The $z_2$-input column is zeroed (we never observed it); the reconstructed block gets
modality-1 shift pushed through $J_{\mathrm{ln12}}$ plus the intrinsic recon noise.

**Practical constraints discovered** (needed for the reconstructor to run inside the EKF
Jacobian): the reconstructor must use **LayerNorm not BatchNorm** (BN's batch reduction is
incompatible with `vmap`), and must be in **eval mode** during the Jacobian (dropout trips
`vmap`'s randomness guard, and $J_f$ must be deterministic).

---

## 3. Stage 3 — Propagation through fusion + head (1st vs 2nd order)

Fusion (concat) is linear → passes $\Sigma_{\text{recon}}$ through unchanged. The frozen
head $g$ maps the (possibly reconstructed) latent to the scalar output. Propagate the
covariance to output variance.

**First-order (EKF / delta method):**
$$\sigma^2_{\text{prop}} = J_g^\top\,\Sigma_{\text{recon}}\,J_g, \qquad J_g=\nabla_z g$$

**Why it is insufficient alone — Jacobian collapse.** A frozen MLP head extrapolates
*flat* outside its training support, so $\lVert J_g\rVert\to 0$ exactly in OOD regions.
Then $\sigma^2_{\text{prop}}$ *shrinks* precisely where epistemic uncertainty should rise.
Empirically first-order gave OOD/ID variance ratio ≈ 1.07× (flat) even though the input
shift amplitude grew 4.4× (doc 06).

**Second-order (adds curvature):**
$$\sigma^2_{\text{prop}} = J_g^\top\Sigma_{\text{recon}} J_g \;+\; \tfrac12\,\mathrm{tr}\!\big((H_g\Sigma_{\text{recon}})^2\big), \qquad H_g=\nabla^2_z g$$
The second term does **not** vanish when $J_g\to 0$ — it captures the head's curvature,
which stays nonzero for smooth (SiLU/GELU) heads even on flat extrapolation plateaus. This
is what restores OOD growth. Code: `full_ekf_propagation_second_order`, Hessian via
double-backward ($d=32$ → full Hessian is cheap).

**What did NOT work: MC-dropout.** Re-enabling head dropout and taking sample variance
gives OOD/ID < 1 (actively worse): the frozen head extrapolates *confidently* in one
direction, so all dropout masks agree and the MC variance *decreases* OOD. MC captures
aleatoric-within-training, not epistemic-about-OOD.

---

## 4. Stage 4 — Aleatoric head + combination

The EKF term is epistemic (near-zero ID, grows OOD/under reconstruction). It does not model
the frozen head's **intrinsic** error (function complexity where $M_A$ is simply wrong).
A small learned head supplies that:
$$\sigma^2_{\text{al}} = \mathrm{softplus}\big(h_\phi(\text{input})\big)+\epsilon, \qquad \sigma^2_{\text{total}} = \sigma^2_{\text{ep}} + \lambda_{\text{al}}\,\sigma^2_{\text{al}}$$
Detach $z$ and $\sigma^2_{\text{ep}}$ at the head boundary so it never back-props into the
EKF chain. Bias init $-3$ (starts near zero, avoids early domination).

**Input choice matters.** `z_only` failed — the encoder's final latent compresses away the
geometric structure of the error map. Raw `xy=(x1,x2)` works (the error map *is* a function
of raw input). For real encoders the equivalent is intermediate activations, not the final
latent (same folding argument as the cycle score).

---

## 5. Stage 5 — Closed-form GGD and losses

**Why closed-form.** In `learned` mode the predictive variance passed through an
`inv_alpha_net` MLP, which *decoupled* it from $\sigma^2_{\text{total}}$ — the MLP could (and
did) ignore our decomposition, giving PCC(predvar, err) ≈ 0. **Closed-form ties them
exactly**: with $\text{inv}\_\alpha = 1/\sqrt{2\sigma^2_{\text{total}}}$ and $\beta=2$, the
GGD variance reduces to $\mathrm{Var}=\sigma^2_{\text{total}}$ (no indirection). This single
change jumped PCC from ~0 to 0.87.

**Losses:**
- **Primary — Gaussian/GGD NLL** (`nll_mode=paper`, $\beta=2$): its MLE drives
  $\sigma^2_{\text{total}}\to(\hat y-y)^2$. (`paper` form is the true log-density; the old
  `repo` surrogate was only linear in inv_alpha and let variance inflate ~1000×.)
- **Auxiliary — linear regression** ($\lambda_{\text{aux}}=0.3$):
  $\mathrm{MSE}(\sigma^2_{\text{ep,detach}}+\lambda_{\text{al}}\sigma^2_{\text{al}},\ (\hat y-y)^2)$.
  Supervising the *total* (not $\sigma_{\text{al}}$ alone) makes the aleatoric head learn
  only the residual the epistemic term misses. Log-space aux did **not** help (tried, hurt
  slightly).

**Numerical guards (from the blow-up debugging):** clamp $\sigma^2\in[10^{-8},10^4]$;
$\beta$ bounded (or fixed at 2); shrink+eigen-clamp $\Sigma_A$ (cond $8.7\text{e}4\to1.4\text{e}2$).

---

## 6. Best configurations achieved

Validated default (`configs/model/ekf_net/default.yaml` + `hook_ekf.yaml`):
`mode=closed_form, use_aleatoric=true, aleatoric_input_mode=xy, hidden_dim=64,
n_layers=3, lambda_aleatoric=1.0, lambda_aux=0.3, sigma_z=cycle_iso, prop_mode=second_order`.

### (1,1) all-present — calibration (cf2/cf6, mask_rate=0)

| Metric | Value |
|---|---|
| PCC(predvar, err) | **0.88** (per-sample) |
| Cell-level PCC (E[var\|cell] vs E[err\|cell]) | **0.945** |
| NLL | **−2.03** |
| PCC(aleatoric, err) | 0.93 |

Interpretation: predicted variance tracks error *magnitude* very well (cell-PCC 0.945). The
per-sample Spearman (~0.45) is capped because most ID cells have near-tied low error where
ranking is uninformative — not a defect.

### OOD sensitivity (doc 06)

Cycle+Identity + second-order: OOD/ID variance ratio ~$10^4$–$10^5$ (vs 1.07× for the
SD+first-order baseline). Cycle bypasses folding; second-order beats Jacobian collapse; the
two compound super-multiplicatively.

### mask_rate=0.5 — overall calibration improved

PCC 0.956, cell-PCC 0.97, NLL better — mask augmentation *helped* ID calibration. **But the
missing-modality-specific behavior is not yet correct (see §7).**

---

## 7. Known gap and next TODO — making (1,0)/(0,1) work

**The failure (mask_rate=0.5 run, job 19771320):** missing-modality samples have 1.31× the
actual error but 0.96× the predicted variance; the epistemic term is *lower* (0.66×) for
missing samples. Exactly backwards. Two root causes:

1. **`Σ_net,recon = 0`** — the reconstruction's intrinsic error is unmodeled. Missing
   modality loses its own shift term and gains nothing for "the reconstruction is wrong,"
   so epistemic drops. This is the dominant bug.
2. **Aleatoric head is mask-blind** — it always sees the real `(x1,x2)`, so it is identical
   for missing vs present at a point (can't inflate, and leaks the absent modality). It also
   dominates (~79% of total variance), so it dilutes any epistemic signal.

**TODO (both needed; neither alone suffices):**

- [ ] **Wire `Σ_net,recon`** (stage 2(ii)): feed the `dev`-head reconstruction-error
  estimate into the EKF as the additive block on reconstructed coordinates. Directly raises
  epistemic for missing samples. This is the principled, dominant fix.
- [ ] **Mask-aware aleatoric head**: feed the *masked* input (zero the absent modality) + an
  availability flag, so it learns higher aleatoric given less information and stops leaking
  the absent modality.
- [ ] Re-run mask_rate=0.5 and confirm `var_miss/var_both` tracks the actual `err` ratio
  (~1.3×), with per-signal calibration (PCC within each of (1,1)/(0,1)/(1,0)).

**Further out:**
- [ ] Real-data transfer: intermediate-activation input for both cycle decoder and aleatoric
  head (raw `xy` only works for the toy).
- [ ] Learned fusion (currently concat, Jacobian = identity) would add its own propagation
  block.
- [ ] Reliability diagram / ECE alongside correlation for the paper.

---

## 8. One-line mental model

> Measure shift where the encoder can't hide it (**cycle**, input space, available
> modalities only) → reconstruct missing modalities and compound their uncertainty
> (**EKF predict**: $J_f\Sigma J_f^\top+\Sigma_{\text{net}}$) → propagate through the frozen
> head with curvature so it doesn't collapse OOD (**second-order**) → add what the head is
> intrinsically bad at (**aleatoric**) → tie variance to the loss with no indirection
> (**closed-form GGD NLL**).

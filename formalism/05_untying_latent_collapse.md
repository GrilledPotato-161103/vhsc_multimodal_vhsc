---
title: Untying the Latent Collapse — Root Cause and Architecture-Agnostic Fix
tags: [SURE, theory, sigma_z, latent-collapse, decoder, cycle-consistency]
created: 2026-06-11
status: draft
context: |
  Follow-up to doc 04. Extensions 1–3 each have a specific weakness:
  BN-shift requires BN at early layers (architectural assumption); GMM and PCA
  still operate on the final latent and are equally blind to encoder folding.
  This note re-states the root cause precisely and proposes a fix that
  does not depend on any specific architecture feature.
related: [[03_phase1_limits_and_phase2_direction]] [[04_sigma_z_extensions]]
---

# Untying the Latent Collapse

## 1. Why all three extensions in doc 04 share the same failure mode

Reviewing the criticisms:

**Extension 1 (BN-shift)**: measures deviation at intermediate BN layers, which avoids the
*fully-folded* latent. But it assumes BN exists at early layers. Modern architectures
(ViT, CLIP, DINOv2, most transformers) use LayerNorm or no normalization at the first
several layers. This is a strong and reviewable architectural assumption.

**Extension 2 (GMM)** and **Extension 3 (PCA)**: strictly more expressive distributional
models than the single Gaussian, but both measure on the **final latent** $z_B = f_\theta(x_B)$.
If the encoder folds $x_B$ back into $\mathcal{M}_A$ before the output, both methods see
$z_B$ as in-distribution regardless of $K$ or $k$. They are improvements to the *metric*,
not to the *measurement point*.

The three extensions address different failure modes:

| Extension | What it fixes | What it does NOT fix |
|---|---|---|
| BN-shift (1) | Measurement point (early layers) | Architectural generality |
| GMM (2) | Distributional model (multimodal A) | Latent-space collapse |
| PCA (3) | Distributional model (low-dim manifold) | Latent-space collapse |

The root problem is none of the above. It is:

> **The encoder $f_\theta$ was trained to produce a compact, well-behaved latent for $A$.
> It has no incentive to keep OOD inputs distinguishable. The shift disappears during the
> forward pass, before any downstream measurement is taken.**

Until this is fixed, extensions 2 and 3 cannot work as shift detectors for the type of
OOD we have — inputs that the encoder silently folds to the edge of the source cloud.

---

## 2. The root cause in one sentence

The encoder $f_\theta$ implements a *one-way contraction*: it maps both
$A$-samples and nearby-OOD samples to approximately the same compact region of $Z$.
This is not a bug — it is precisely what representation learning objectives encourage.
A latent space that is well-separated for $A$ and invariant to nuisance variation is
exactly what a good encoder does. That same invariance makes it shift-blind.

Formally: let $x_B = x_A^{\mathrm{edge}} + \delta$ where $x_A^{\mathrm{edge}}$ is a
boundary point of $A$'s support and $\delta$ is the OOD displacement. Then:

$$\|f_\theta(x_B) - f_\theta(x_A^{\mathrm{edge}})\| \;\leq\; \mathrm{Lip}(f_\theta) \cdot \|\delta\|$$

For an MLP with BN and bounded activations trained on $A$, the effective Lipschitz
constant in the direction of $\delta$ is typically *much smaller* than the global Lipschitz
constant — the encoder has learned to be locally flat near the training boundary. So
$f_\theta(x_B) \approx f_\theta(x_A^{\mathrm{edge}})$ and any latent-space metric gives
$s(z_B) \approx s(z_{A^{\mathrm{edge}}}) \approx 1$.

---

## 3. The general condition for an untied distance

A shift descriptor $s$ is *untied from the latent collapse* if it satisfies:

**(U)** $s(x_B)$ grows with $\|x_B - \mathrm{proj}_A(x_B)\|_X$ even when
$\|f_\theta(x_B) - \mathrm{proj}_{f_\theta(A)}(f_\theta(x_B))\|_Z \approx 0$.

In words: the score must grow as the input moves away from $A$, *independently* of
whether the latent also moves away. This requires either:

(a) **Measuring before folding happens** — early-layer intermediate representations
    (what BN-shift does, with the architectural restriction noted).

(b) **Inverting the folding** — mapping $z_B$ back to input space via a learned inverse
    and measuring the shift *there*.

(c) **Using a signal orthogonal to the encoder** — a measurement that $f_\theta$ does
    not touch (raw input features, a second model).

Option (b) is the most principled and architecturally neutral. It is the focus of the rest
of this note.

---

## 4. The cycle-consistency shift score

### Construction

Train an auxiliary **decoder** $g_\phi: Z \to X$ on source data $\{(z_i^A, x_i^A)\}$:

$$g_\phi = \argmin_\phi \sum_i \|g_\phi(f_\theta(x_i^A)) - x_i^A\|^2$$

$g_\phi$ is a small MLP trained once, offline, after the backbone is frozen. It does not
affect $f_\theta$ at all.

At inference on target sample $x_B$:

$$\hat{x}_B = g_\phi(f_\theta(x_B)), \qquad \boxed{s_{\mathrm{cyc}}(x_B) = \|x_B - \hat{x}_B\|}$$

### Why this unties the collapse

The encoder folds $x_B$ to $z_B \approx z_{A'}$ for some source point $x_{A'} \in A$. The
decoder then maps $z_B \approx z_{A'}$ back to $\hat{x}_B \approx x_{A'}$. The cycle error is:

$$s_{\mathrm{cyc}}(x_B) = \|x_B - \hat{x}_B\| \approx \|x_B - x_{A'}\|$$

This is the **input-space distance** from $x_B$ to the source point it was folded to. If
$x_B$ is in-distribution, $x_{A'} \approx x_B$ and the error is small. If $x_B$ is OOD,
$x_{A'}$ is the nearest point in $A$, and the error reflects the actual input-space shift.

The collapse is untied: even though $z_B \approx z_{A'}$ (latents indistinguishable),
$s_{\mathrm{cyc}}$ reads out the input-space gap, which the encoder compressed but the
decoder exposes.

```
x_B (OOD)                  x_{A'} (source match)
    |                            ↑
    ↓         f_θ (folds)        |
   z_B   ≈   z_{A'}  ──── g_φ (inverts) ──→  \hat{x}_B ≈ x_{A'}
    |                            |
    └──── s_cyc = ||x_B - \hat{x}_B|| ────────┘
                    (input-space gap, large when OOD)
```

### Properties

**(P1) In-distribution**: for $x_B \sim A$, the encoder-decoder round-trip is the training
objective of $g_\phi$, so $\hat{x}_B \approx x_B$ and $s_{\mathrm{cyc}} \approx 0$.

**(P2) Monotone in shift**: as $x_B$ moves away from $A$'s support,
$\|x_B - x_{A'}\| = \|x_B - \mathrm{proj}_A(x_B)\|$ grows. So $s_{\mathrm{cyc}}$ grows.

**(P3) EKF-consumable**: use $s_{\mathrm{cyc}}$ as the amplitude $s$ in
$\Sigma_z = s \cdot \Phi$, with $\Phi$ from any of the extensions in doc 04.

**(Architecture-agnostic)**: $g_\phi$ is a plain MLP. No assumptions about BN, LayerNorm,
or depth of the backbone. Works for any frozen $f_\theta$.

### Cost

- **Training**: $g_\phi$ is trained once on source pairs $(f_\theta(x^A), x^A)$. For
  the toy ($X = \mathbb{R}^1$, $Z = \mathbb{R}^{32}$), this is a 32→1 MLP with a few
  hundred parameters. For medical imaging with a CLIP encoder
  ($Z \in \mathbb{R}^{512}$, $X$ = image), $g_\phi$ is a lightweight image decoder
  — not a full diffusion model, just a small convolutional decoder.
- **Inference**: one extra forward pass through $g_\phi$ per sample. Negligible.
- **Source data requirement**: needs source pairs $(x^A, z^A = f_\theta(x^A))$.
  This is the **SD-setting** (doc 01 §6). The decoder can be made B-only with the
  approximation in §5 below.

---

## 5. B-only variant — decoder trained on target reconstruction

If source data is unavailable (B-only setting), we cannot directly train
$g_\phi$ on $(z^A, x^A)$ pairs. But we can train $g_\phi$ on target data
$\{(z^B_i, x^B_i)\}$ and use the reconstruction error as a *relative* score:

$$s_{\mathrm{cyc}}^{B}(x_B) = \|x_B - g_\phi^B(f_\theta(x_B))\|$$

where $g_\phi^B$ was trained on target pairs. In-distribution target samples (those
close to $A$) will reconstruct well. OOD target samples (those far from $A$) will
reconstruct poorly, *to the extent that $g_\phi^B$ learned from the in-distribution
majority*.

This works if:
- The target distribution $B$ has a mix of in-distribution and OOD samples.
- $g_\phi^B$ fits the in-distribution majority and leaves OOD samples with high error.

This is the classic self-supervised anomaly detection setup (DSVDD, Ruff et al. 2018).
For the toy, B has $x_1 \in [-0.6, 1.4]$ with OOD at $x_1 > 1.0$ (~20% of samples) — 
the in-distribution majority is large enough for this to work.

---

## 6. Unlocking extensions 2 and 3

Once $s_{\mathrm{cyc}}$ provides a reliable, untied shift amplitude, extensions 2 and 3
from doc 04 become viable as *shape* ($\Phi$) or *metric* refinements:

**With cycle score + GMM shape**:
$$\Sigma_z(x_B) = s_{\mathrm{cyc}}(x_B) \cdot \Sigma_{k^*(x_B)}$$
where $k^*$ is the cluster of the nearest source point $\hat{x}_B$ (now in input space,
which means clustering in input space becomes an option too). This combines an
input-space amplitude with a latent-space shape — the two carry complementary information.

**With cycle score + PCA shape**:
$$\Sigma_z(x_B) = s_{\mathrm{cyc}}(x_B) \cdot U_k U_k^\top + \epsilon I$$
The amplitude comes from input-space, the direction from latent-space PCA. This
separates the concerns cleanly: *how much shift* (cycle error) vs *in what direction*
(latent PCA).

The key point: once $s$ is reliable, the choice of $\Phi$ is a second-order decision
about directionality. All the options in doc 04 §2–3 become valid shape descriptors
on top of an untied amplitude.

---

## 7. Connection to existing literature

**Cycle-consistency for domain adaptation** (Zhu et al. 2017, CycleGAN): learns mappings
$X \to Y$ and $Y \to X$ such that $x \approx F(G(x))$. We use the simpler one-directional
version: $x_B \approx g_\phi(f_\theta(x_B))$ is trained only on source; the cycle error
on target measures domain gap.

**Reconstruction-based anomaly detection** (Schlegl et al. 2017, f-AnoGAN; Ruff et al.
2018, Deep SVDD): trains an autoencoder or reconstruction network on in-distribution data;
anomalies have high reconstruction error. Our $g_\phi$ is exactly this: the "decoder" half
of an anomaly detection pipeline, trained on source and evaluated on target.

**Input-reconstruction OOD score** (Xiao et al. 2020, Likelihood Regret): uses the
reconstruction error of a VAE trained on in-distribution data as an OOD score. Same spirit.

The novel aspect here is not the score itself, but its role in the SURE pipeline: the cycle
error $s_{\mathrm{cyc}}$ feeds the EKF chain as the amplitude of $\Sigma_z$, converting
an input-space reconstruction signal into a latent-space covariance that the EKF can
propagate to output uncertainty. This is the formal bridge the literature does not provide.

---

## 8. Summary of the extension hierarchy

Ordered by what they fix:

```
Level 0 (current):  SD Mahalanobis on final z
  Problem: latent collapse; d_M^2 stays ~1 for OOD inputs

Level 1 (doc 04):   GMM or PCA on final z
  Problem: still latent collapse; better metric, wrong measurement point

Level 1 (doc 04):   BN-shift on intermediate activations
  Problem: architecture-dependent (BN required)

Level 2 (this doc): Cycle-consistency score on input space via decoder
  Fix: input-space distance, architecture-agnostic, unties the collapse
  Cost: auxiliary decoder, source data (or target-majority for B-only)

Level 3 (combined): Cycle score (amplitude) + GMM/PCA (shape)
  Best of both: reliable shift magnitude + directional structure
```

The formalism is the same at every level: $\Sigma_z = s \cdot \Phi$.
The extensions differ only in which $s$ and $\Phi$ they use. Level 2 is the first
one that satisfies condition (U) from §3 — shift-detection is untied from the encoder.

---

## 9. Open questions

1. **How large does $g_\phi$ need to be?** For the toy (32→1), near-trivially small. For
   a medical VLM with CLIP-sized latents (512-d) and image targets, the decoder is
   significant. The minimum $g_\phi$ that achieves low training reconstruction error on
   source determines the cost of the method.

2. **Is the cycle score sufficient alone, or does the shape $\Phi$ matter?** If
   $s_{\mathrm{cyc}} \cdot I$ (isotropic, B-only) already gives good calibration after
   EKF propagation, the directional structure from GMM/PCA adds complexity without gain.
   Only experimental comparison resolves this.

3. **Does the cycle error behave well when the encoder folds multiple OOD regions to the
   same source point?** If two very different OOD inputs $x_B^{(1)}$ and $x_B^{(2)}$ both
   fold to $z_{A'} \approx f_\theta(x_{A'})$, they get the same $\hat{x}_B$ and hence the
   same $s_{\mathrm{cyc}}$, even though they may be OOD in different directions. This is a
   fundamental limitation of any method that routes through $f_\theta$ at all.

4. **For the B-only variant, what is the required fraction of in-distribution samples in $B$
   for $g_\phi^B$ to generalize correctly?** If $B$ is heavily OOD (e.g. a domain completely
   different from $A$), $g_\phi^B$ may fit the OOD regime and fail to flag it. The method
   assumes $B$ is a *mixture* with an in-distribution majority.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Shared helper: fit source latents given encoders + x_range
# ---------------------------------------------------------------------------

def _sample_source_latents(encoder1, encoder2, x_range, n_source_samples, device):
    """Draw source samples, encode, return (z_A, x1_A, x2_A)."""
    a, b = x_range
    gen = torch.Generator(device=device).manual_seed(0)
    x1 = torch.rand(n_source_samples, 1, generator=gen, device=device) * (b - a) + a
    x2 = torch.rand(n_source_samples, 1, generator=gen, device=device) * (b - a) + a
    was1, was2 = encoder1.training, encoder2.training
    encoder1.eval(); encoder2.eval()
    with torch.no_grad():
        z1 = encoder1(x1)
        z2 = encoder2(x2)
    if was1: encoder1.train()
    if was2: encoder2.train()
    return torch.cat([z1, z2], dim=-1), x1, x2, z1, z2


def _fit_sigma_A(z_A, cov_floor, shrinkage, device):
    """Fit shrunk, eigen-clamped Gaussian to source latents. Returns (mu, Sigma, Sigma_inv)."""
    d_z = z_A.shape[1]
    mu_A = z_A.mean(0)
    centered = z_A - mu_A
    sigma_A = (centered.T @ centered) / (z_A.shape[0] - 1)
    eye = torch.eye(d_z, device=device)
    mean_var = sigma_A.diagonal().mean()
    sigma_A = (1.0 - shrinkage) * sigma_A + shrinkage * mean_var * eye
    sigma_A = sigma_A + cov_floor * eye
    evals, evecs = torch.linalg.eigh(sigma_A)
    evals = evals.clamp_min(cov_floor)
    sigma_A_inv = (evecs / evals) @ evecs.T
    print(f"  Sigma_A: min_eval={evals.min():.3e}  max_eval={evals.max():.3e}  "
          f"cond={evals.max()/evals.min():.3e}")
    return mu_A, sigma_A, sigma_A_inv


class SDSigmaZ(nn.Module):
    """Single-Gaussian Mahalanobis, formalism doc 01."""

<<<<<<< HEAD
    Fits a single Gaussian N(mu_A, Sigma_A) on source latents at init time, then
    at inference returns per-sample full Sigma_z(z) = (d_M^2(z) / d_z) * Sigma_A,
    with d_M^2 the Mahalanobis distance from z to (mu_A, Sigma_A).

    Properties:
      - In-distribution: E[d_M^2 / d_z] = 1, so Sigma_z ~ Sigma_A on average.
      - OOD: d_M^2 grows, Sigma_z is amplified in directions Sigma_A spans.
      - PSD by construction: non-negative scalar times PSD matrix.

    See formalism/01_input_shift_measurement.md for the derivation.
    """

    def __init__(self,
                 encoder1: nn.Module,
                 encoder2: nn.Module,
                 dataset: Dataset | None = None,
                 x_range: tuple = (-1.0, 1.0),
                 n_source_samples: int = 5000,
                 device: str = "cuda",
                 cov_floor: float = 1e-4,
                 shrinkage: float = 0.1):
        super().__init__()
        self.x_range = x_range
        self.n_source_samples = n_source_samples
        self.cov_floor = cov_floor
        self.shrinkage = shrinkage

        a, b = x_range
        # Deterministic source-sample fit: same mu_A / sigma_A across runs.
        if not dataset:
            gen = torch.Generator(device=device).manual_seed(0)
            x1 = torch.rand(n_source_samples, 1, generator=gen, device=device) * (b - a) + a
            x2 = torch.rand(n_source_samples, 1, generator=gen, device=device) * (b - a) + a
            # Pretrained backend always in eval() mode
        else:
            xss = [dataset[idx][0] for idx in range(len(dataset))]
            x1 = torch.tensor([xs[0] for xs in xss], device=device).unsqueeze_(-1)
            x2 = torch.tensor([xs[1] for xs in xss], device=device).unsqueeze_(-1)

        with torch.no_grad():
            z1 = encoder1(x1)  # (N, d_z/2)
            z2 = encoder2(x2)  # (N, d_z/2)
        z_A = torch.cat([z1, z2], dim=-1)  # (N, d_z)

        mu_A = z_A.mean(dim=0)  # (d_z,)
        centered = z_A - mu_A
        sigma_A = (centered.T @ centered) / (n_source_samples - 1)  # (d_z, d_z)
        d_z = mu_A.shape[0]
        eye = torch.eye(d_z, device=device)
        # Ledoit-Wolf-style shrinkage toward an isotropic target bounds cond(Sigma_A):
        # without it lambda_min sits at cov_floor (~1e-4) while lambda_max ~ O(1),
        # so cond ~ 1e4-1e5 and the Mahalanobis distance is dominated by near-null
        # directions that do NOT reflect the actual input shift. The shrunk metric is
        # closer to isotropic, so d_M^2 tracks the real OOD displacement.
        mean_var = torch.diagonal(sigma_A).mean()
        sigma_A = (1.0 - shrinkage) * sigma_A + shrinkage * mean_var * eye
        sigma_A = sigma_A + cov_floor * eye
        # Symmetric-eigendecomposition inverse (more stable than torch.linalg.inv on
        # an ill-conditioned matrix).
        evals, evecs = torch.linalg.eigh(sigma_A)
        evals = evals.clamp_min(cov_floor)
        sigma_A_inv = (evecs / evals) @ evecs.T

        # Diagnostic: report Sigma_A eigenvalue spectrum once at init.
        eigvals = torch.linalg.eigvalsh(sigma_A)
        print(f"[SDSigmaZ] Sigma_A eigenvalues: "
              f"min={eigvals.min().item():.3e}  "
              f"max={eigvals.max().item():.3e}  "
              f"cond={(eigvals.max() / eigvals.min()).item():.3e}")
        print(f"[SDSigmaZ] z_A   per-coord var range: "
              f"min={z_A.var(dim=0).min().item():.3e}  "
              f"max={z_A.var(dim=0).max().item():.3e}")

=======
    def __init__(self, encoder1, encoder2, x_range=(-1., 1.),
                 n_source_samples=5000, device="cuda",
                 cov_floor=1e-4, shrinkage=0.1):
        super().__init__()
        print("[SDSigmaZ] fitting source Gaussian...")
        z_A, _, _, _, _ = _sample_source_latents(encoder1, encoder2, x_range, n_source_samples, device)
        mu_A, sigma_A, sigma_A_inv = _fit_sigma_A(z_A, cov_floor, shrinkage, device)
>>>>>>> f4ea00808caa3beff81a7be3cbb65ec303cd8b04
        self.register_buffer("mu_A", mu_A)
        self.register_buffer("sigma_A", sigma_A)
        self.register_buffer("sigma_A_inv", sigma_A_inv)
        self.d_z = mu_A.shape[0]

    def amplitude(self, z):
        delta = z - self.mu_A
        return ((delta @ self.sigma_A_inv) * delta).sum(-1).clamp_min(0.0) / self.d_z

    def forward(self, z, x1=None, x2=None, signal=(1, 1)):
        amp = self.amplitude(z)
        return amp.view(-1, 1, 1) * self.sigma_A.unsqueeze(0)


class CycleSigmaZ(nn.Module):
    """Cycle-consistency shift score: s = ||x - g(f(x))||.

    Trains a small decoder g: Z -> X on source pairs.  At inference the
    reconstruction error in input space measures shift even when the encoder
    folds OOD inputs back into the source latent cloud.
    See formalism/05_untying_latent_collapse.md.
    """

    def __init__(self, encoder1, encoder2, x_range=(-1., 1.),
                 n_source_samples=5000, device="cuda",
                 cov_floor=1e-4, shrinkage=0.1,
                 n_train_steps=3000, lr=1e-3,
                 phi_mode="sigma_a"):
        """
        phi_mode: shape of Sigma_z
            "sigma_a"  – amplitude * Sigma_A  (default)
            "identity" – amplitude * I
        """
        super().__init__()
        self.phi_mode = phi_mode
        print(f"[CycleSigmaZ] fitting source data + training decoders "
              f"({n_train_steps} steps)...")
        z_A, x1_A, x2_A, z1_A, z2_A = _sample_source_latents(
            encoder1, encoder2, x_range, n_source_samples, device)
        d_half = z1_A.shape[1]

        # Two lightweight decoders: one per modality (R^16 -> R^1 for the toy)
        def _make_dec(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 32), nn.SiLU(),
                nn.Linear(32, 16), nn.SiLU(),
                nn.Linear(16, 1)
            ).to(device)

        # Seed decoder init for reproducible cycle baselines across runs.
        torch.manual_seed(0)
        dec1 = _make_dec(d_half)
        dec2 = _make_dec(d_half)

        opt = torch.optim.Adam(list(dec1.parameters()) + list(dec2.parameters()), lr=lr)
        z1_A = z1_A.detach(); z2_A = z2_A.detach()
        x1_A = x1_A.detach(); x2_A = x2_A.detach()

        dec1.train(); dec2.train()
        for step in range(n_train_steps):
            idx = torch.randint(n_source_samples, (256,), device=device)
            loss = (F.mse_loss(dec1(z1_A[idx]), x1_A[idx]) +
                    F.mse_loss(dec2(z2_A[idx]), x2_A[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 1000 == 0:
                print(f"  step {step+1}/{n_train_steps}  recon_loss={loss.item():.5f}")

        dec1.eval(); dec2.eval()
        self.dec1 = dec1
        self.dec2 = dec2
        self.d_half = d_half
        self.d_z = z_A.shape[1]

        if phi_mode == "sigma_a":
            mu_A, sigma_A, _ = _fit_sigma_A(z_A, cov_floor, shrinkage, device)
            self.register_buffer("mu_A", mu_A)
            self.register_buffer("sigma_A", sigma_A)
        else:
            self.register_buffer("sigma_A", torch.eye(self.d_z, device=device))

        # PER-MODALITY source baselines (for signal-aware normalisation).
        with torch.no_grad():
            e1 = (dec1(z1_A) - x1_A).pow(2).mean(-1)
            e2 = (dec2(z2_A) - x2_A).pow(2).mean(-1)
            b1 = e1.mean().clamp_min(1e-6)
            b2 = e2.mean().clamp_min(1e-6)
            baseline = (e1 + e2).mean().clamp_min(1e-6)   # kept for back-compat
        self.register_buffer("cycle_baseline", baseline)
        self.register_buffer("b1", b1)
        self.register_buffer("b2", b2)
        print(f"[CycleSigmaZ] source baselines: b1={b1.item():.5f} b2={b2.item():.5f}")

    @torch.no_grad()
    def per_modality_shift(self, z, x1, x2, signal=(1, 1)):
        """Per-modality cycle amplitude, ZEROED on missing modalities.

        Availability (reconstructor convention, ln12/ln21):
          modality 1 available  iff signal[1] == 1   (mod_1 reconstructed when p2==0)
          modality 2 available  iff signal[0] == 1   (mod_2 reconstructed when p1==0)
        A missing modality contributes 0 input variance: its latent will be
        REPLACED by a reconstruction downstream, and the EKF Jacobian J_f zeros
        its input column anyway, so 0 is exact (see formalism/08).
        """
        B = z.shape[0]
        z1 = z[:, :self.d_half]; z2 = z[:, self.d_half:]
        avail_1 = (signal[1] == 1)
        avail_2 = (signal[0] == 1)
        if avail_1:
            s1 = (self.dec1(z1) - x1).pow(2).mean(-1) / self.b1
        else:
            s1 = torch.zeros(B, device=z.device)
        if avail_2:
            s2 = (self.dec2(z2) - x2).pow(2).mean(-1) / self.b2
        else:
            s2 = torch.zeros(B, device=z.device)
        return s1.clamp_min(0.0), s2.clamp_min(0.0)

    def forward(self, z, x1=None, x2=None, signal=(1, 1)):
        assert x1 is not None and x2 is not None, \
            "CycleSigmaZ requires x1, x2 (raw inputs) in forward()"
        s1, s2 = self.per_modality_shift(z, x1, x2, signal=signal)
        # Per-coordinate diagonal scale: s1 over modality-1 coords, s2 over modality-2.
        scale = torch.cat([
            s1.unsqueeze(-1).expand(-1, self.d_half),
            s2.unsqueeze(-1).expand(-1, self.d_half),
        ], dim=-1)                                   # (B, d_z)
        # Sigma_z = D^{1/2} Phi D^{1/2}: for Phi=I gives diag(scale); for full Sigma_A
        # scales row i / col j by sqrt(s_i) sqrt(s_j) (stays PSD). Missing coords -> 0.
        rt = scale.clamp_min(0.0).sqrt()             # (B, d_z)
        Sigma = rt.unsqueeze(-1) * self.sigma_A.unsqueeze(0) * rt.unsqueeze(-2)
        return Sigma                                 # (B, d_z, d_z)


class GMMSigmaZ(nn.Module):
    """Multi-cluster Mahalanobis: fits K Gaussians on source latents.

    s_GMM(z) = soft-weighted average Mahalanobis to K cluster centers.
    Phi = soft-weighted average of cluster covariances.
    See formalism/04_sigma_z_extensions.md §Extension 2.
    """

    def __init__(self, encoder1, encoder2, x_range=(-1., 1.),
                 n_source_samples=5000, device="cuda",
                 cov_floor=1e-4, shrinkage=0.1,
                 n_clusters=4, n_kmeans_iters=100):
        super().__init__()
        print(f"[GMMSigmaZ] fitting {n_clusters} clusters on source latents...")
        z_A, _, _, _, _ = _sample_source_latents(encoder1, encoder2, x_range, n_source_samples, device)
        d_z = z_A.shape[1]

        # K-means initialisation via random subset
        # randperm generator must be CPU even when tensors live on CUDA
        gen = torch.Generator(device="cpu").manual_seed(42)
        idx = torch.randperm(n_source_samples, generator=gen)[:n_clusters]
        centers = z_A[idx].clone()

        for _ in range(n_kmeans_iters):
            dists = torch.cdist(z_A, centers)          # (N, K)
            assign = dists.argmin(dim=1)               # (N,)
            new_centers = torch.stack([
                z_A[assign == k].mean(0) if (assign == k).any() else centers[k]
                for k in range(n_clusters)
            ])
            if (new_centers - centers).norm() < 1e-6:
                break
            centers = new_centers

        # Per-cluster covariance
        mu_list, sigma_list, sigma_inv_list, pi_list = [], [], [], []
        for k in range(n_clusters):
            mask = assign == k
            z_k = z_A[mask]
            n_k = z_k.shape[0]
            if n_k < 2:
                z_k = z_A  # fallback: use all
                n_k = z_A.shape[0]
            mu_k = z_k.mean(0)
            c = z_k - mu_k
            sigma_k = (c.T @ c) / (n_k - 1)
            mu_k2, sigma_k2, sigma_k2_inv = _fit_sigma_A(
                z_k, cov_floor, shrinkage, device)
            mu_list.append(mu_k2)
            sigma_list.append(sigma_k2)
            sigma_inv_list.append(sigma_k2_inv)
            pi_list.append(torch.tensor(n_k / n_source_samples, device=device))

        self.register_buffer("mu_k",       torch.stack(mu_list))          # (K, d)
        self.register_buffer("sigma_k",    torch.stack(sigma_list))       # (K, d, d)
        self.register_buffer("sigma_k_inv",torch.stack(sigma_inv_list))   # (K, d, d)
        self.register_buffer("pi_k",       torch.stack(pi_list))          # (K,)
        self.d_z = d_z
        self.K = n_clusters

    def forward(self, z, x1=None, x2=None, signal=(1, 1)):
        B = z.shape[0]
        # Mahalanobis distance to each cluster: (B, K)
        delta = z.unsqueeze(1) - self.mu_k.unsqueeze(0)         # (B, K, d)
        d2 = torch.einsum("bki,kij,bkj->bk", delta, self.sigma_k_inv, delta)
        d2 = d2.clamp_min(0.0)

        # Soft assignment: posterior p(k|z)
        log_p = -0.5 * d2 + self.pi_k.log().unsqueeze(0)
        weights = torch.softmax(log_p, dim=1)                   # (B, K)

        # Weighted average amplitude and covariance
        amp = (weights * d2 / self.d_z).sum(-1)                 # (B,)
        sigma_z = torch.einsum("bk,kij->bij", weights, self.sigma_k)  # (B, d, d)
        sigma_z = amp.view(-1, 1, 1) * sigma_z
        return sigma_z


class PCASigmaZ(nn.Module):
    """Off-manifold PCA projection distance.

    s_PCA(z) = ||(z - mu_A) - U_k U_k^T (z - mu_A)||^2 / (d_z - k)
    i.e. squared residual in the directions NOT spanned by top-k PCs.
    See formalism/04_sigma_z_extensions.md §Extension 3A.
    """

    def __init__(self, encoder1, encoder2, x_range=(-1., 1.),
                 n_source_samples=5000, device="cuda",
                 cov_floor=1e-4, shrinkage=0.1,
                 n_components=2):
        super().__init__()
        print(f"[PCASigmaZ] fitting PCA ({n_components} components) on source latents...")
        z_A, _, _, _, _ = _sample_source_latents(encoder1, encoder2, x_range, n_source_samples, device)
        mu_A, sigma_A, _ = _fit_sigma_A(z_A, cov_floor, shrinkage, device)
        d_z = mu_A.shape[0]

        # SVD of centred source latents to get principal directions
        evals, evecs = torch.linalg.eigh(sigma_A)   # ascending order
        U_k = evecs[:, -n_components:]               # (d_z, k) top-k eigenvectors

        # For reference: fraction of variance captured
        var_frac = evals[-n_components:].sum() / evals.sum()
        print(f"  top-{n_components} PCs capture {var_frac*100:.1f}% of source variance")

        self.register_buffer("mu_A",    mu_A)
        self.register_buffer("sigma_A", sigma_A)
        self.register_buffer("U_k",     U_k)
        self.d_z = d_z
        self.k = n_components
        self.off_dim = d_z - n_components

    def forward(self, z, x1=None, x2=None, signal=(1, 1)):
        delta = z - self.mu_A                                       # (B, d_z)
        on_manifold = delta @ self.U_k @ self.U_k.T                 # (B, d_z)
        off_manifold = delta - on_manifold                          # (B, d_z)
        amp = off_manifold.pow(2).sum(-1) / max(self.off_dim, 1)    # (B,)
        amp = amp.clamp_min(0.0)
        return amp.view(-1, 1, 1) * self.sigma_A.unsqueeze(0)


class GroundTruthSigmaZ:
    """Computes and caches ground-truth diagonal Σ_z for known Uniform(a,b) input distributions."""

    def __init__(self, encoder1: nn.Module, encoder2: nn.Module,
                 x_range: tuple, n_mc: int = 5000,
                 mode: str = "mc", device: str = "cpu"):
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.x_range = x_range
        self.n_mc = n_mc
        self.mode = mode
        self.device = device
        self._diag_sigma_z = None

    def compute_mc(self) -> torch.Tensor:
        """MC sampling through frozen encoders. Returns (32,) diagonal variance."""
        a, b = self.x_range
        x1 = torch.rand(self.n_mc, 1, device=self.device) * (b - a) + a
        x2 = torch.rand(self.n_mc, 1, device=self.device) * (b - a) + a
        with torch.no_grad():
            z1 = self.encoder1(x1)  # (N, 16)
            z2 = self.encoder2(x2)  # (N, 16)
        return torch.cat([z1.var(dim=0), z2.var(dim=0)])  # (32,)

    def compute_jacobian(self) -> torch.Tensor:
        """Jacobian-based analytical approximation. Returns (32,)."""
        a, b = self.x_range
        mu_x = torch.tensor([(a + b) / 2.0], device=self.device)
        var_x = (b - a) ** 2 / 12.0
        with torch.no_grad():
            J1 = jacrev(self.encoder1)(mu_x.unsqueeze(0)).squeeze()  # (16,)
            J2 = jacrev(self.encoder2)(mu_x.unsqueeze(0)).squeeze()  # (16,)
        return torch.cat([var_x * J1 ** 2, var_x * J2 ** 2])  # (32,)

    @property
    def diag_sigma_z(self) -> torch.Tensor:
        if self._diag_sigma_z is None:
            if self.mode == "mc":
                self._diag_sigma_z = self.compute_mc()
            elif self.mode == "jacobian":
                self._diag_sigma_z = self.compute_jacobian()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        return self._diag_sigma_z


class BNShiftSigmaZ:
    """Per-sample BN shift score using backbone BatchNorm running statistics."""

    def __init__(self, backbone: nn.Module):
        self.bn_stats = []
        for name, module in backbone.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                self.bn_stats.append({
                    "name": name,
                    "running_mean": module.running_mean.clone().detach(),
                    "running_var": module.running_var.clone().detach(),
                })

    def compute_shift_score(self, z_activations: dict) -> torch.Tensor:
        """Per-sample BN shift score. Returns: (B,)"""
        scores = []
        for stats in self.bn_stats:
            name = stats["name"]
            if name not in z_activations:
                continue
            z_l = z_activations[name]  # (B, d_l)
            mu_l = stats["running_mean"].to(z_l.device)
            var_l = stats["running_var"].to(z_l.device)
            s_l = ((z_l - mu_l) ** 2 / (var_l + 1e-8)).mean(dim=-1)  # (B,)
            scores.append(s_l)
        if not scores:
            return torch.zeros(next(iter(z_activations.values())).shape[0])
        return torch.stack(scores).mean(dim=0)  # (B,)

    def get_sigma_z(self, shift_score: torch.Tensor, d: int = 32) -> torch.Tensor:
        """Returns (B, d) diagonal Σ_z = s(z) · I"""
        return shift_score.unsqueeze(-1).expand(-1, d)  # (B, 32)

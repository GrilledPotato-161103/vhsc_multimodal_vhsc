import torch
import torch.nn as nn
from torch.func import jacrev
from torch.utils.data import Dataset

class SDSigmaZ(nn.Module):
    """Source-dependent input-shift covariance provider (Option B in the formalism).

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

        self.register_buffer("mu_A", mu_A)
        self.register_buffer("sigma_A", sigma_A)
        self.register_buffer("sigma_A_inv", sigma_A_inv)
        self.d_z = d_z

    def mahalanobis_sq(self, z: torch.Tensor) -> torch.Tensor:
        """d_M^2(z; mu_A, Sigma_A). Returns (B,) non-negative scalar per sample."""
        delta = z - self.mu_A  # (B, d_z)
        return ((delta @ self.sigma_A_inv) * delta).sum(dim=-1).clamp_min(0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Per-sample full Sigma_z. z: (B, d_z) -> (B, d_z, d_z)."""
        amp = self.mahalanobis_sq(z) / self.d_z  # (B,)
        return amp.view(-1, 1, 1) * self.sigma_A.unsqueeze(0)  # (B, d_z, d_z)


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

"""Projected manifold dataset with high-dimensional latent z and physics-based regression.

Generates z ~ U(-1,1)^{emb_dim}, projects through overlapping subspace matrices
to produce n_modality observations, and computes y via Lorenz-96 ODE integration.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math

import torch
from torch.utils.data import Dataset


def _lorenz96_dynamics(x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Compute Lorenz-96 tendency: dx_j/dt = (x_{j+1} - x_{j-2}) * x_{j-1} - x_j + F_j.

    Args:
        x: State tensor of shape (N, J) or (J,).
        F: Forcing tensor broadcastable with x.

    Returns:
        Tendency tensor of same shape as x.
    """
    x1 = torch.roll(x, shifts=-1, dims=-1)   # x_{j+1}
    x2 = torch.roll(x, shifts=2, dims=-1)    # x_{j-2}
    x3 = torch.roll(x, shifts=1, dims=-1)    # x_{j-1}
    return (x1 - x2) * x3 - x + F


def _rk4_step(
    x: torch.Tensor, F: torch.Tensor, dt: float = 0.01
) -> torch.Tensor:
    """Single batched RK4 integration step for Lorenz-96.

    Args:
        x: Current state (N, J).
        F: Forcing terms (N, J).
        dt: Time step.

    Returns:
        Next state (N, J).
    """
    k1 = _lorenz96_dynamics(x, F)
    k2 = _lorenz96_dynamics(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_dynamics(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_dynamics(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ProjectedManifoldDataset(Dataset):
    """High-dimensional latent manifold with overlapping modality projections.

    Each modality observes a d_proj-dimensional projection of the true latent
    z ∈ R^{emb_dim}.  Projection matrices share ``overlap_degree`` fraction of
    their row-space basis, simulating real-world multimodal data where most
    information is shared but some is modality-private.

    The regression target y is computed via Lorenz-96 ODE integration, where
    the first ``n_vars`` dimensions of z control the spatially-varying forcing
    terms.  This creates a complex, non-trivial mapping z → y.

    Returns (per item):
        xs_noisy: (n_modals, d_proj)   noisy modality observations
        y:        (1,)                  regression target
        xs:       (n_modals, d_proj)   clean modality observations
        z:        (emb_dim,)            ground-truth latent
    """

    def __init__(
        self,
        n_samples: int,
        x_expressions: None = None,  # ignored; kept for config compatibility
        y_expression: None = None,    # ignored; kept for config compatibility
        emb_dim: int = 512,
        n_modals: int = 3,
        d_proj: int = 128,
        overlap_degree: float = 0.85,
        z_range: Sequence[Tuple[float, float]] | Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.1,
        physics: str = "lorenz96",
        physics_n_vars: int = 40,
        physics_n_steps: int = 500,
        physics_dt: float = 0.01,
        physics_burnin: int = 250,
        projection_matrices: Optional[torch.Tensor] = None,
        generator: torch.Generator | None = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.emb_dim = emb_dim
        self.n_modals = n_modals
        self.d_proj = d_proj
        self.overlap_degree = overlap_degree
        self.noise_std = noise_std
        self.dtype = dtype

        g = torch.Generator().manual_seed(seed) if generator is None else generator

        # --- generate latent z -------------------------------------------------
        if isinstance(z_range, tuple) and len(z_range) == 2 and isinstance(z_range[0], (int, float)):
            # Single (min, max) broadcast to all dims
            lo, hi = z_range
            self.z = torch.empty((n_samples, emb_dim)).uniform_(lo, hi, generator=g)
        else:
            # Per-dimension ranges
            self.z = torch.stack([
                torch.empty(n_samples).uniform_(lz, rz, generator=g)
                for (lz, rz) in z_range
            ], dim=-1)

        # --- build projection matrices -----------------------------------------
        if projection_matrices is not None:
            self.P = projection_matrices.to(dtype)
        else:
            self.P = self._build_projection_matrices(g)

        # --- compute modality observations -------------------------------------
        # xs: (n_samples, n_modals, d_proj)
        self.xs = torch.einsum("mde,ne->nmd", self.P, self.z.to(dtype))

        # add noise (avoid torch.full with generator — not universally supported)
        noise_mask = (
            torch.empty(n_samples, n_modals, d_proj)
            .uniform_(0.0, 1.0, generator=g)
            .lt(0.5)
            .int()
        )
        noise = noise_std * torch.empty_like(self.xs).uniform_(-1.0, 1.0, generator=g)
        self.xs_noisy = self.xs + noise_mask * noise

        # --- compute regression target y ---------------------------------------
        self.y = self._compute_y(self.z, physics, physics_n_vars, physics_n_steps,
                                 physics_dt, physics_burnin, g)

        # ensure y is (n_samples, 1)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Projection matrix construction
    # ------------------------------------------------------------------

    def _build_projection_matrices(
        self, generator: torch.Generator
    ) -> torch.Tensor:
        """Build n_modals projection matrices with controlled row-space overlap.

        Returns:
            Tensor of shape (n_modals, d_proj, emb_dim).
        """
        d_shared = int(self.d_proj * self.overlap_degree)
        d_private = self.d_proj - d_shared

        # Shared basis: random matrix, orthogonalise rows via QR
        W_shared = torch.randn(d_shared, self.emb_dim, generator=generator)
        W_shared = torch.linalg.qr(W_shared.T)[0].T  # (d_shared, emb_dim)

        matrices: list[torch.Tensor] = []
        for _ in range(self.n_modals):
            W_private = torch.randn(d_private, self.emb_dim, generator=generator)
            W_private = torch.linalg.qr(W_private.T)[0].T  # (d_private, emb_dim)
            P_i = torch.cat([W_shared, W_private], dim=0)  # (d_proj, emb_dim)
            # Normalise so E[||Pz||^2] ≈ 1 when z ~ U(-1,1)^d
            # Var[z_j] = 1/3 → Var[Pz] ≈ d_proj/3 → scale by sqrt(3/d_proj)
            P_i = P_i * math.sqrt(3.0 / self.d_proj)
            matrices.append(P_i)

        return torch.stack(matrices, dim=0).to(self.dtype)

    # ------------------------------------------------------------------
    # Physics engine
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_y(
        z: torch.Tensor,
        physics: str,
        n_vars: int,
        n_steps: int,
        dt: float,
        burnin: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Compute regression target from latent z via a physics simulation.

        Currently only ``"lorenz96"`` is supported.
        """
        if physics == "lorenz96":
            return ProjectedManifoldDataset._lorenz96_y(
                z, n_vars, n_steps, dt, burnin, generator
            )
        raise ValueError(f"Unsupported physics engine: {physics}")

    @staticmethod
    def _lorenz96_y(
        z: torch.Tensor,
        n_vars: int,
        n_steps: int,
        dt: float,
        burnin: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Lorenz-96 integration → kernel-smoothed bounded scalar target y.

        Maps the first ``n_vars`` dims of z to spatially-varying forcing
        terms F_j ∈ [-6, 6] (periodic dominant), integrates the Lorenz-96
        system, then applies Gaussian kernel smoothing over z-space to
        guarantee local smoothness for t-SNE visualization.
        """
        N = z.shape[0]
        device = z.device

        # Forcing: F_j = 6 * z_j  →  F ∈ [-6, 6]
        # Periodic-dominant regime — macro-structure preserved post-smoothing.
        F = z[:, :n_vars] * 6.0  # (N, J)

        # Initial condition: small random perturbation
        x = 0.01 * torch.randn(N, n_vars, generator=generator, device=device)

        # Integrate
        trajectory: list[torch.Tensor] = []
        for step in range(n_steps):
            x = _rk4_step(x, F, dt)
            # Prevent runaway trajectories
            x = torch.clamp(x, min=-20.0, max=20.0)
            if step >= burnin:
                trajectory.append(x)

        # y = mean over (time, spatial dimensions) → kernel-smooth → bounded
        traj = torch.stack(trajectory, dim=0)       # (T_post, N, J)
        y_raw = traj.mean(dim=0).mean(dim=-1)        # (N,)
        y = ProjectedManifoldDataset._smooth_y(z, y_raw, n_vars)
        y = torch.clamp(y, min=-10.0, max=10.0)      # bounded regression target
        return y

    @staticmethod
    def _smooth_y(
        z: torch.Tensor, y: torch.Tensor, n_vars: int,
        k: int = 100,
    ) -> torch.Tensor:
        """Gaussian kernel smooth y over the forcing dimensions of z.

        Caps the distance computation at 10 dims to avoid the curse of
        dimensionality — beyond ~10 dims the kernel bandwidth becomes
        meaningless.  Uses an adaptive bandwidth set to the median distance
        to the k-th neighbour.
        """
        N = y.shape[0]
        if N <= k:
            return y

        d_eff = min(n_vars, 10)                  # cap at 10D for meaningful distances
        z_f = z[:, :d_eff]                       # (N, d_eff)
        dists = torch.cdist(z_f, z_f)            # (N, N)
        nn_dists, nn_idx = dists.topk(min(k, N), dim=-1, largest=False)

        # Adaptive bandwidth: median distance to k-th neighbour
        sigma = nn_dists[:, -1].median().clamp_min(0.5).item()

        nn_z = z_f[nn_idx]                       # (N, k, d_eff)
        sq_dists = ((nn_z - z_f.unsqueeze(1)) ** 2).sum(dim=-1)  # (N, k)
        weights = torch.exp(-sq_dists / (2.0 * sigma ** 2))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        nn_y = y[nn_idx]                         # (N, k)
        y_smooth = (nn_y * weights).sum(dim=-1)   # (N,)
        return y_smooth

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.xs_noisy[idx],   # (n_modals, d_proj)
            self.y[idx],          # (1,)
            self.xs[idx],         # (n_modals, d_proj)
            self.z[idx],          # (emb_dim,)
        )

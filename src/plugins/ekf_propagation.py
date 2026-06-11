"""
EKF diagonal uncertainty propagation utilities for SURE experiment.

Implements:
  Σ_z → Σ_recon via reconstructor Jacobian J_f
  Σ_recon → σ²_pred via predictor Jacobian J_g

All propagation uses diagonal (not full matrix) approximation for efficiency.
"""

from typing import Callable, Tuple
import torch
import torch.nn as nn


def compute_reconstructor_jacobian(
    reconstructor_fn: Callable,
    z: torch.Tensor,
) -> torch.Tensor:
    """Batched Jacobian J_f = d(reconstructor_fn)/dz using vmap(jacrev).

    Args:
        reconstructor_fn: pure function (32,) -> (d',) — no batch dim
        z: (B, 32) input features

    Returns:
        J_f: (B, d', 32) batched Jacobians
    """
    from torch.func import jacrev, vmap
    return vmap(jacrev(reconstructor_fn), randomness='same')(z)


def propagate_sigma_z_to_sigma_recon(
    J_f: torch.Tensor,
    diag_sigma_z: torch.Tensor,
) -> torch.Tensor:
    """Diagonal EKF step 1: diag(Σ_recon)_i = Σ_k J_f[i,k]^2 * σ²_z[k].

    Args:
        J_f: (B, d', 32)
        diag_sigma_z: (32,) shared across batch

    Returns:
        diag_sigma_recon: (B, d')
    """
    # What if we
    # print(diag_sigma_z[:5], J_f.mean(dim=(0, 1))[:5])
    return (J_f ** 2) @  diag_sigma_z  # (B, d', 32) x (32,) -> (B, d')


def compute_predictor_jacobian(
    predictor_fn: Callable,
    z_recon: torch.Tensor,
) -> torch.Tensor:
    """Gradient of scalar predictor output w.r.t. reconstructed features.

    Args:
        predictor_fn: function (B, d') -> (B,) or (B, 1)
        z_recon: (B, d')

    Returns:
        J_g: (B, d') gradient vector (scalar output -> Jacobian is a vector)
    """
    # Defensive: re-enable autograd here. Lightning's val/test loops may run
    # inside torch.no_grad(); the surrounding wrappers in validation_step
    # / test_step protect model_step, but if anything downstream is in a
    # no_grad context this restarts a fresh grad-tracked chain.
    with torch.enable_grad():
        z_recon_g = z_recon.detach().requires_grad_(True)
        print(z_recon_g.shape)
        y_pred = predictor_fn(z_recon_g)
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze(-1)  # (B,)
        grads = torch.autograd.grad(
            y_pred.sum(), z_recon_g, create_graph=True
        )[0]  # (B, d')
    return grads


def propagate_sigma_recon_to_sigma_pred(
    J_g: torch.Tensor,
    diag_sigma_recon: torch.Tensor,
) -> torch.Tensor:
    """Diagonal EKF step 2: σ²_pred = Σ_i J_g[i]^2 * diag(Σ_recon)_i.

    Args:
        J_g: (B, d') predictor Jacobian
        diag_sigma_recon: (B, d')

    Returns:
        sigma_pred_sq: (B,) per-sample predictive variance
    """
    return (J_g ** 2 * torch.clamp_min(diag_sigma_recon, 1e-4)).sum(dim=-1)  # (B,)


def make_reconstructor_fn(reconstructor: nn.Module, signal: tuple) -> Callable:
    """Extract pure functional form of BilinearReconstructor for vmap/jacrev.

    The returned fn operates on a single sample (no batch dim).
    signal: (p1, p2) where 0 = modality missing, 1 = present.
    """
    def fn(z: torch.Tensor) -> torch.Tensor:
        # z: (32,) single sample
        mod_1 = z[:16]
        mod_2 = z[16:]
        # print(signal)
        p1, p2 = signal
        if p1 == 0:
            rec_1 = reconstructor.ln21(mod_2.unsqueeze(0)).squeeze(0)
        else:
            rec_1 = mod_1
        if p2 == 0:
            rec_2 = reconstructor.ln12(mod_1.unsqueeze(0)).squeeze(0)
        else:
            rec_2 = mod_2
        return torch.cat([rec_1, rec_2])  # (32,)
    return fn


def make_predictor_fn(head: nn.Module) -> Callable:
    """Extract functional form of predictor head.

    Operates on batched input (B, d').
    """
    def fn(z_recon: torch.Tensor) -> torch.Tensor:
        return head(z_recon)
    return fn


def full_ekf_propagation(
    z: torch.Tensor,
    diag_sigma_z: torch.Tensor,
    reconstructor_fn: Callable,
    predictor_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end EKF: Σ_z -> Σ_recon -> σ²_pred.

    Args:
        z: (B, 32) concatenated encoder features
        diag_sigma_z: (32,) diagonal input variance
        reconstructor_fn: pure fn (32,) -> (d',)
        predictor_fn: fn (B, d') -> (B,)

    Returns:
        sigma_pred_sq: (B,) per-sample output variance
        diag_sigma_recon: (B, d') intermediate reconstructor uncertainty
        J_f: (B, d', 32) reconstructor Jacobians
    """
    from torch.func import vmap

    # Step 1: Σ_z -> Σ_recon
    J_f = compute_reconstructor_jacobian(reconstructor_fn, z.unsqueeze_(1)).squeeze_()
    diag_sigma_recon = propagate_sigma_z_to_sigma_recon(J_f, diag_sigma_z)
    # Compute reconstructed features
    with torch.no_grad():
        z_recon = vmap(reconstructor_fn, randomness='same')(z)  # (B, d')

    # Step 2: Σ_recon -> σ²_pred
    J_g = compute_predictor_jacobian(predictor_fn, z_recon)
    sigma_pred_sq = propagate_sigma_recon_to_sigma_pred(J_g, diag_sigma_recon)
    # print(sigma_pred_sq.shape, diag_sigma_recon.shape, J_f.shape, J_g.shape)
    return sigma_pred_sq, diag_sigma_recon, J_f, J_g


# --------------------------------------------------------------------------
# Full-covariance per-sample EKF — used by the SD-setting Sigma_z provider.
# --------------------------------------------------------------------------

def propagate_sigma_z_to_sigma_recon_full(
    J_f: torch.Tensor,
    sigma_z: torch.Tensor,
) -> torch.Tensor:
    """Full EKF step 1: Σ_recon = J_f Σ_z J_f^T.

    Args:
        J_f:     (B, d', d) reconstructor Jacobian
        sigma_z: (B, d, d) per-sample input covariance

    Returns:
        sigma_recon: (B, d', d')
    """
    return J_f @ sigma_z @ J_f.transpose(-1, -2)


def propagate_sigma_recon_to_sigma_pred_full(
    J_g: torch.Tensor,
    sigma_recon: torch.Tensor,
) -> torch.Tensor:
    """Full EKF step 2: σ²_pred = J_g^T Σ_recon J_g (scalar quadratic form).

    Args:
        J_g:         (B, d')
        sigma_recon: (B, d', d')

    Returns:
        sigma_pred_sq: (B,)
    """
    return torch.einsum("bi,bij,bj->b", J_g, sigma_recon, J_g)


def full_ekf_propagation_full(
    z: torch.Tensor,
    sigma_z: torch.Tensor,
    reconstructor_fn: Callable,
    predictor_fn: Callable,
    diag_floor: float = 1e-6,
    pred_floor: float = 1e-8,
    pred_ceiling: float = 1e4,
    diag_ceiling: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end EKF with full per-sample input covariance.

    Args:
        z:        (B, d) latent batch
        sigma_z:  (B, d, d) per-sample input covariance
        reconstructor_fn: pure fn (d,) -> (d',)
        predictor_fn:     fn (B, d') -> (B,)

    Returns:
        sigma_pred_sq:    (B,)
        diag_sigma_recon: (B, d')  -- diagonal extracted from full Σ_recon
        J_f:              (B, d', d)
        J_g:              (B, d')
    """
    from torch.func import vmap

    J_f = compute_reconstructor_jacobian(reconstructor_fn, z.unsqueeze(1)).squeeze()
    print(J_f.shape)
    sigma_recon = propagate_sigma_z_to_sigma_recon_full(J_f, sigma_z)

    with torch.no_grad():
        z_recon = reconstructor_fn(z)

    J_g = compute_predictor_jacobian(predictor_fn, z_recon)
    sigma_pred_sq = propagate_sigma_recon_to_sigma_pred_full(J_g, sigma_recon)
    # Floor AND ceiling: the floor avoids log(0); the ceiling is a structural guard so a
    # badly-conditioned Sigma_z or an extreme Mahalanobis amplitude can never feed an
    # astronomically large variance into the heads / NLL / variance map (the original bug).
    sigma_pred_sq = sigma_pred_sq.clamp(min=pred_floor, max=pred_ceiling)

    diag_sigma_recon = sigma_recon.diagonal(dim1=-2, dim2=-1).clamp(min=diag_floor, max=diag_ceiling)
    return sigma_pred_sq, diag_sigma_recon, J_f, J_g

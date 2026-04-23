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
    return vmap(jacrev(reconstructor_fn))(z)


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
    return (J_f ** 2) @ diag_sigma_z  # (B, d', 32) x (32,) -> (B, d')


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
    z_recon_g = z_recon.detach().requires_grad_(True)
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
    return (J_g ** 2 * diag_sigma_recon).sum(dim=-1)  # (B,)


def make_reconstructor_fn(reconstructor: nn.Module, signal: tuple) -> Callable:
    """Extract pure functional form of BilinearReconstructor for vmap/jacrev.

    The returned fn operates on a single sample (no batch dim).
    signal: (p1, p2) where 0 = modality missing, 1 = present.
    """
    def fn(z: torch.Tensor) -> torch.Tensor:
        # z: (32,) single sample
        mod_1 = z[:16]
        mod_2 = z[16:]
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
    J_f = compute_reconstructor_jacobian(reconstructor_fn, z)
    diag_sigma_recon = propagate_sigma_z_to_sigma_recon(J_f, diag_sigma_z)

    # Compute reconstructed features
    with torch.no_grad():
        z_recon = vmap(reconstructor_fn)(z)  # (B, d')

    # Step 2: Σ_recon -> σ²_pred
    J_g = compute_predictor_jacobian(predictor_fn, z_recon)
    sigma_pred_sq = propagate_sigma_recon_to_sigma_pred(J_g, diag_sigma_recon)

    return sigma_pred_sq, diag_sigma_recon, J_f

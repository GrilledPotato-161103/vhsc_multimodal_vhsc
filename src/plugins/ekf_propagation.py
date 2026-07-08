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
        # print(z_recon_g.shape)
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
    J_f = compute_reconstructor_jacobian(reconstructor_fn, z.unsqueeze(1)).squeeze(1)
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
    # print(J_f.shape)
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


# --------------------------------------------------------------------------
# Second-order Taylor correction (formalism doc 03 §7 Fix 1)
# --------------------------------------------------------------------------

def compute_hessian(predictor_fn: Callable, z: torch.Tensor) -> torch.Tensor:
    """Full Hessian H_g = d²g/dz² for scalar predictor g.

    Uses reverse-over-reverse autodiff: one backward pass per column of H.
    Cost: d backward passes per call — feasible for d=32, expensive for d>128.

    Args:
        predictor_fn: (B, d) -> (B,) or (B,1)
        z: (B, d) — must NOT be inside a no_grad context

    Returns:
        H: (B, d, d) symmetric Hessian
    """
    B, d = z.shape
    z_g = z.detach().requires_grad_(True)
    y = predictor_fn(z_g)
    if y.dim() > 1:
        y = y.squeeze(-1)
    J = torch.autograd.grad(y.sum(), z_g, create_graph=True)[0]  # (B, d)
    H = torch.zeros(B, d, d, device=z.device, dtype=z.dtype)
    for i in range(d):
        col = torch.autograd.grad(
            J[:, i].sum(), z_g,
            retain_graph=(i < d - 1),
            create_graph=False
        )[0]  # (B, d)
        H[:, i, :] = col.detach()
    return H


def second_order_correction(
    H: torch.Tensor,
    sigma_recon: torch.Tensor,
) -> torch.Tensor:
    """Second-order variance correction: (1/2) tr((H Σ_recon)^2).

    Derivation: for z ~ N(z0, Σ), Var[g(z)] ≈ J^T Σ J + (1/2) tr((H Σ)^2)
    The quadratic term captures curvature and does NOT vanish when J→0.

    Uses identity: tr(A^2) = tr(A A) = (A * A^T).sum(dim=(-2,-1))

    Args:
        H:            (B, d, d) Hessian
        sigma_recon:  (B, d, d) propagated covariance at the predictor input

    Returns:
        correction: (B,) non-negative scalars
    """
    M = torch.bmm(H, sigma_recon)           # (B, d, d)
    # tr(M^2) = tr(M @ M) = (M * M^T).sum
    correction = 0.5 * (M * M.transpose(-1, -2)).sum(dim=(-2, -1))  # (B,)
    return correction.clamp_min(0.0)


def full_ekf_propagation_second_order(
    z: torch.Tensor,
    sigma_z: torch.Tensor,
    reconstructor_fn: Callable,
    predictor_fn: Callable,
    diag_floor: float = 1e-6,
    pred_floor: float = 1e-8,
    pred_ceiling: float = 1e4,
    diag_ceiling: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EKF with second-order Taylor variance correction.

    σ²_pred = J_g^T Σ_recon J_g  +  (1/2) tr((H_g Σ_recon)^2)

    The second term does not vanish when J_g→0 (Jacobian collapse fix).
    See formalism/03_phase1_limits_and_phase2_direction.md §7 Fix 1.

    Args: same as full_ekf_propagation_full.
    Returns: same tuple (sigma_pred_sq, diag_sigma_recon, J_f, J_g).
    """
    from torch.func import vmap

    J_f = compute_reconstructor_jacobian(reconstructor_fn, z)
    sigma_recon = propagate_sigma_z_to_sigma_recon_full(J_f, sigma_z)

    with torch.no_grad():
        z_recon = vmap(reconstructor_fn)(z)

    # J_g and H_g both need autograd
    with torch.enable_grad():
        H = compute_hessian(predictor_fn, z_recon)

    J_g = compute_predictor_jacobian(predictor_fn, z_recon)
    sps_first  = propagate_sigma_recon_to_sigma_pred_full(J_g, sigma_recon)
    sps_second = second_order_correction(H, sigma_recon)
    sigma_pred_sq = (sps_first + sps_second).clamp(min=pred_floor, max=pred_ceiling)

    diag_sigma_recon = sigma_recon.diagonal(dim1=-2, dim2=-1).clamp(min=diag_floor, max=diag_ceiling)
    return sigma_pred_sq, diag_sigma_recon, J_f, J_g


# --------------------------------------------------------------------------
# MC-dropout ensemble propagation (formalism doc 03 §7 Fix 2 / Option B3)
# --------------------------------------------------------------------------

def mc_dropout_propagation(
    predictor: nn.Module,
    z_recon: torch.Tensor,
    K: int = 20,
) -> torch.Tensor:
    """Prediction variance via K MC-dropout forward passes.

    Temporarily sets the predictor to train() so dropout is active.
    Returns the empirical variance across K samples — does NOT vanish
    when the first-order Jacobian is near zero (Jacobian-collapse fix).

    Args:
        predictor: the frozen head (must have dropout layers; dropout=0.5 in config)
        z_recon:   (B, d') input to the predictor
        K:         number of stochastic forward passes

    Returns:
        sigma_pred_sq: (B,) per-sample empirical variance
    """
    was_training = predictor.training
    predictor.train()   # activates dropout
    preds = []
    with torch.no_grad():
        for _ in range(K):
            y = predictor(z_recon)
            if y.dim() > 1:
                y = y.squeeze(-1)
            preds.append(y)
    if not was_training:
        predictor.eval()
    preds = torch.stack(preds, dim=0)   # (K, B)
    return preds.var(dim=0).clamp_min(1e-8)   # (B,)


def full_ekf_propagation_mc_dropout(
    z: torch.Tensor,
    sigma_z: torch.Tensor,
    reconstructor_fn: Callable,
    predictor: nn.Module,
    K: int = 20,
    blend_alpha: float = 0.5,
    diag_floor: float = 1e-6,
    pred_floor: float = 1e-8,
    pred_ceiling: float = 1e4,
    diag_ceiling: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EKF with MC-dropout variance blended with first-order EKF variance.

    σ²_pred = (1 - α) * EKF_first_order  +  α * MC_dropout_var

    The MC-dropout term captures curvature/epistemic uncertainty in OOD
    regions where J_g→0. blend_alpha=1.0 = pure dropout; 0.0 = pure EKF.

    Args:
        z, sigma_z, reconstructor_fn: same as full_ekf_propagation_full
        predictor: nn.Module with dropout (needed for train() mode)
        K: number of dropout samples
        blend_alpha: mixture weight for dropout term (default 0.5)
    Returns: same tuple (sigma_pred_sq, diag_sigma_recon, J_f, J_g).
    """
    from torch.func import vmap

    J_f = compute_reconstructor_jacobian(reconstructor_fn, z)
    sigma_recon = propagate_sigma_z_to_sigma_recon_full(J_f, sigma_z)

    with torch.no_grad():
        z_recon = vmap(reconstructor_fn)(z)

    J_g = compute_predictor_jacobian(predictor.forward, z_recon)
    sps_ekf = propagate_sigma_recon_to_sigma_pred_full(J_g, sigma_recon)
    sps_mc  = mc_dropout_propagation(predictor, z_recon, K=K)

    sigma_pred_sq = ((1 - blend_alpha) * sps_ekf + blend_alpha * sps_mc)
    sigma_pred_sq = sigma_pred_sq.clamp(min=pred_floor, max=pred_ceiling)

    diag_sigma_recon = sigma_recon.diagonal(dim1=-2, dim2=-1).clamp(min=diag_floor, max=diag_ceiling)
    return sigma_pred_sq, diag_sigma_recon, J_f, J_g

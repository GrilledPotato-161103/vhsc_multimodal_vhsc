"""
EKF Generalized Gaussian Distribution NLL Loss.

alpha (scale) comes from EKF-propagated sigma_pred, not from a neural network.
beta (shape) is a single learnable scalar parameter.

GGD NLL: (|y - mu| / alpha)^beta + log(alpha) + lgamma(1/beta) - log(beta)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.models.components.toy import MLP, Residual, get_normalization
from src.plugins.var import BreakpointContext, BreakpointOutput
from src.plugins.ekf_propagation import *


class AleatoricHead(nn.Module):
    """Learned aleatoric uncertainty head.

    Maps concat(z, log_sigma_ep) -> sigma_al (strictly positive).
    Captures in-distribution function complexity that the EKF epistemic
    term misses. Combined with sigma_ep via:
        sigma_total = sigma_ep + lambda_aleatoric * sigma_al

    input_mode:
      "z_and_sep" (default): concat(z, log_sigma_ep) — OOD-aware aleatoric
      "z_only":               z only — pure function complexity, ignores shift

    See formalism/06_empirical_validation.md §6 and the implementation plan.
    """

    def __init__(self,
                 z_dim: int = 32,
                 hidden_dim: int = 32,
                 n_layers: int = 2,
                 activation: str = "silu",
                 norm: str = "layer",
                 eps: float = 1e-6,
                 input_mode: str = "z_and_sep",
                 xy_dim: int = 2):
        super().__init__()
        self.eps = eps
        self.input_mode = input_mode
        act = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation]
        if input_mode == "xy":
            in_dim = xy_dim          # raw input coords bypass z entirely
        elif input_mode == "z_only":
            in_dim = z_dim
        else:
            in_dim = z_dim + 1       # z + log_sigma_ep
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=True), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       get_normalization(norm, hidden_dim),
                       act()]
        out_layer = nn.Linear(hidden_dim, 1)
        # Init bias so Softplus(bias) ≈ 0 at start — prevents early domination.
        nn.init.constant_(out_layer.bias, -3.0)
        nn.init.xavier_uniform_(out_layer.weight)
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, sigma_ep: torch.Tensor,
                xy: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z:        (B, z_dim) latent features
            sigma_ep: (B, 1) epistemic variance from EKF
            xy:       (B, xy_dim) raw input coords — required when input_mode='xy'
        Returns:
            sigma_al: (B, 1) strictly positive aleatoric variance
        """
        if self.input_mode == "xy":
            assert xy is not None, "xy required when input_mode='xy'"
            feat = xy.detach()
        elif self.input_mode == "z_only":
            feat = z.detach()
        else:
            log_sep = torch.log(sigma_ep.detach().clamp_min(self.eps))
            feat = torch.cat([z.detach(), log_sep], dim=-1)
        return F.softplus(self.net(feat)) + self.eps                 # (B, 1)


class AleatoricHead(nn.Module):
    """Learned aleatoric uncertainty head.

    Maps concat(z, log_sigma_ep) -> sigma_al (strictly positive).
    Captures in-distribution function complexity that the EKF epistemic
    term misses. Combined with sigma_ep via:
        sigma_total = sigma_ep + lambda_aleatoric * sigma_al

    input_mode:
      "z_and_sep" (default): concat(z, log_sigma_ep) — OOD-aware aleatoric
      "z_only":               z only — pure function complexity, ignores shift

    See formalism/06_empirical_validation.md §6 and the implementation plan.
    """
    def __init__(self,
                 z_dim: int = 32,
                 hidden_dim: int = 32,
                 n_layers: int = 2,
                 activation: str = "silu",
                 norm: str = "layer",
                 eps: float = 1e-6,
                 input_mode: str = "z_and_sep",
                 xy_dim: int = 2):
        super().__init__()
        self.eps = eps
        self.input_mode = input_mode
        act = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation]
        if input_mode == "xy":
            in_dim = xy_dim          # raw input coords bypass z entirely
        elif input_mode == "z_only":
            in_dim = z_dim
        else:
            in_dim = z_dim + 1       # z + log_sigma_ep
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=True), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       get_normalization(norm, hidden_dim),
                       act()]
        out_layer = nn.Linear(hidden_dim, 1)
        # Init bias so Softplus(bias) ≈ 0 at start — prevents early domination.
        nn.init.constant_(out_layer.bias, -3.0)
        nn.init.xavier_uniform_(out_layer.weight)
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, sigma_ep: torch.Tensor,
                xy: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z:        (B, z_dim) latent features
            sigma_ep: (B, 1) epistemic variance from EKF
            xy:       (B, xy_dim) raw input coords — required when input_mode='xy'
        Returns:
            sigma_al: (B, 1) strictly positive aleatoric variance
        """
        if self.input_mode == "xy":
            assert xy is not None, "xy required when input_mode='xy'"
            feat = xy.detach()
        elif self.input_mode == "z_only":
            feat = z.detach()
        else:
            log_sep = torch.log(sigma_ep.detach().clamp_min(self.eps))
            feat = torch.cat([z.detach(), log_sep], dim=-1)
        return F.softplus(self.net(feat)) + self.eps                 # (B, 1)


class EKFBiModalInferer(nn.Module):
    def __init__(self,
                    reconstructor: nn.Module,
                    predictor: nn.Module,
                    latent_dim: 16,
                    n_modals:int = 2,
                    pred_dim: int = 1,
                    output_dim: int  =  1,
                    hidden_dims:  Sequence | List | int = 256,
                    bottleneck_dim: int | None = None,
                    per_dim_uncertainty: bool = True,
                    dropout: float = 0.0,
                    activation: str = "gelu",
                    norm: str = "batch",
                    eps: float = 1e-6,
                    residual = False,
                    mode: str = "learned",
                    prop_mode: str = "first_order",
                    beta_min: float = 0.5,
                    beta_max: float = 4.0,
<<<<<<< HEAD
                    aleatoric_net: Optional[AleatoricHead] = None,
                    lambda_aleatoric: float = 1.0,
                    prop_mode: str = "first_order"
=======
                    use_aleatoric: bool = False,
                    aleatoric_hidden_dim: int = 32,
                    aleatoric_n_layers: int = 2,
                    aleatoric_activation: str = "silu",
                    aleatoric_norm: str = "layer",
                    aleatoric_input_mode: str = "z_only",
                    aleatoric_xy_dim: int = 2,
                    lambda_aleatoric: float = 1.0,
>>>>>>> f4ea00808caa3beff81a7be3cbb65ec303cd8b04
                    ):
        # We gonna take
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pred_dim = pred_dim
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.reconstructor = reconstructor
        self.predictor = predictor
        latent_size = latent_dim * n_modals
        self.per_dim_uncertainty = per_dim_uncertainty
        self.output_dim = output_dim if per_dim_uncertainty else 1
        self.eps = eps
        if mode not in ("learned", "closed_form"):
            raise ValueError(f"mode must be 'learned' or 'closed_form', got {mode!r}")
        if prop_mode not in ("first_order", "second_order"):
            raise ValueError(f"prop_mode must be 'first_order' or 'second_order', got {prop_mode!r}")
        self.mode = mode
        self.prop_mode = prop_mode
        self.lambda_aleatoric = lambda_aleatoric
        bottleneck_dim = bottleneck_dim or hidden_dims[-1]

        if use_aleatoric:
            self.aleatoric_net = AleatoricHead(
                z_dim=latent_size,
                hidden_dim=aleatoric_hidden_dim,
                n_layers=aleatoric_n_layers,
                activation=aleatoric_activation,
                norm=aleatoric_norm,
                input_mode=aleatoric_input_mode,
                xy_dim=aleatoric_xy_dim,
                eps=eps,
            )
        else:
            self.aleatoric_net = None

        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        if prop_mode not in ("first_order", "second_order"):
            raise ValueError(f"prop_mode must be 'first_order' or 'second_order', got {prop_mode!r}")
        self.mode = mode
        self.prop_mode = prop_mode
        self.lambda_aleatoric = lambda_aleatoric 
        self.aleatoric_net = aleatoric_net

        # Handle J_f scalar function (B, Src, Dst) -> (B, )
        stem_dim = hidden_dims[0]
        hidden_dim = hidden_dims[-1]
        hidden_dims = hidden_dims[1:-1]

        # Alpha blocks
        self.inv_alpha_stem = nn.Sequential( 
                                    nn.Linear(latent_size + output_dim, stem_dim, bias=False),
                                    nn.Softplus(),
                                )
        inv_alpha_blocks = MLP(in_dim=stem_dim,
                                hidden_dims=hidden_dims,
                                out_dim=hidden_dim,
                                activation=activation,
                                norm = norm,
                                residual= residual,
                                dropout=dropout
                                )
        # No terminal ReLU: a hard 0 makes alpha = 1/inv_alpha blow up to inf.
        # softplus(+eps) is applied in forward() instead -> inv_alpha is strictly
        # positive and smooth.
        inv_alpha_head = nn.Sequential(
                                    nn.Linear(hidden_dim, self.output_dim),
                                )
        self.inv_alpha_net = nn.Sequential(inv_alpha_blocks, inv_alpha_head)

        # Beta blocks
        self.beta_stem = nn.Sequential(
                                    nn.Linear(latent_size, stem_dim, bias=False),
                                    act(),
                                    nn.LayerNorm(stem_dim)
                                )
        beta_blocks = MLP(in_dim=stem_dim,
                                hidden_dims=hidden_dims,
                                out_dim=hidden_dim,
                                activation=activation,
                                norm = norm,
                                residual= residual,
                                dropout=dropout
                                )
        # No terminal ReLU: beta -> 0 makes lgamma(1/beta) overflow. Output a raw
        # logit; forward() squashes it into [beta_min, beta_max] with a sigmoid.
        beta_head = nn.Sequential(
                                    nn.Linear(hidden_dim, self.output_dim),
                                )

        self.beta_net = nn.Sequential(beta_blocks, beta_head)

    def get_parameters(self):
        params = []
        # params.extend(list(self.output_enc.parameters()))
        # params.extend(list(self.mu_head.parameters()))
        params.extend(list(self.inv_alpha_stem.parameters()))
        params.extend(list(self.inv_alpha_net.parameters()))
        params.extend(list(self.beta_stem.parameters()))
        params.extend(list(self.beta_net.parameters()))
        return params
    # Export 
    def get_recon_fn(self, signal: tuple = (1, 1)):
        def infer(z):
            return self.reconstructor.forward_raw(z, signal=signal)
        return infer
    
<<<<<<< HEAD
    def forward(self, z: torch.Tensor, sigma_z: torch.Tensor, pred: torch.Tensor, signal: tuple = (1, 1)):
=======
    def forward(self, z: torch.Tensor, sigma_z: torch.Tensor, signal: tuple = (1, 1),
                xy: torch.Tensor | None = None):
>>>>>>> f4ea00808caa3beff81a7be3cbb65ec303cd8b04
        """
        Args:
            z:       (B, d_z) latent batch
            sigma_z: (B, d_z, d_z) per-sample input covariance
            signal:  (p1, p2) modality-presence tuple
            xy:      (B, n_modals) raw input coordinates — required when
                     aleatoric_input_mode='xy'
        """
        assert self.predictor is not None, "Prediction head is None"
        recon_fn = self.get_recon_fn(signal=signal)
        pred_fn = self.predictor.forward
        # The EKF Jacobian J_f = vmap(jacrev(recon_fn)) runs through the
        # reconstructor (ln12/ln21). Force it to eval so dropout is OFF: vmap
        # rejects random ops, and J_f must be deterministic anyway. The recon-loss
        # forward already ran this batch, so toggling here is safe. Restore after.
        recon_was_training = self.reconstructor.training
        self.reconstructor.eval()
        if self.prop_mode == "second_order":
            sigma_pred_sq, diag_sigma_recon, J_f, J_g = full_ekf_propagation_second_order(
                z, sigma_z=sigma_z, reconstructor_fn=recon_fn, predictor_fn=pred_fn)
        else:
            sigma_pred_sq, diag_sigma_recon, J_f, J_g = full_ekf_propagation_full(
                z, sigma_z=sigma_z, reconstructor_fn=recon_fn, predictor_fn=pred_fn)
        if recon_was_training:
            self.reconstructor.train()   # restore mode for next batch's recon-loss forward
        if len(sigma_pred_sq.shape) < 2:
            sigma_pred_sq = sigma_pred_sq.unsqueeze_(-1)

        # Epistemic variance from EKF propagation.
        sigma_ep = sigma_pred_sq   # (B, 1)

        # Aleatoric variance: learned MLP on z + log(sigma_ep).
        # Zero if use_aleatoric=False (aleatoric_net is None).
        if self.aleatoric_net is not None:
            sigma_al = self.aleatoric_net(z, sigma_ep, xy=xy)    # (B, 1)
            sigma_total = sigma_ep + self.lambda_aleatoric * sigma_al
        else:
            sigma_al = torch.zeros_like(sigma_ep)
            sigma_total = sigma_ep

        if self.mode == "closed_form":
            inv_alpha = 1.0 / torch.sqrt(2.0 * sigma_total + self.eps)
            beta = torch.full_like(sigma_total, 2.0)
            return inv_alpha, beta, sigma_ep, sigma_al

        # mode == "learned" — bounded heads.
        S_f = torch.linalg.svdvals(J_f.permute(0, 2, 1))
        # output_latent = self.output_enc(pred)
        # mu = self.mu_head(output_latent)
        beta_stem = self.beta_stem(S_f / torch.amax(S_f, dim=-1, keepdim=True))
        beta_raw = self.beta_net(beta_stem)
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta_raw)
<<<<<<< HEAD
        # Feed the EKF variance in log-space so the head sees a well-scaled signal
        # regardless of whether sigma_pred_sq is 1e-4 or 1e4.
        ekf_feat = torch.log(torch.cat([sigma_pred_sq, diag_sigma_recon], dim=-1)).clamp_min(self.eps)
        # print(ekf_feat.shape)
        inv_alpha_stem = self.inv_alpha_stem(ekf_feat)
        inv_alpha = F.softplus(self.inv_alpha_net(inv_alpha_stem)) + self.eps
        return pred, inv_alpha, beta, sigma_pred_sq
        
=======
        # inv_alpha_net sees sigma_total (not just sigma_ep) so it adapts to the
        # combined epistemic + aleatoric variance.
        ekf_feat = torch.log(torch.cat([sigma_total, diag_sigma_recon], dim=-1).clamp_min(self.eps))
        inv_alpha = F.softplus(self.inv_alpha_net(ekf_feat)) + self.eps
        return inv_alpha, beta, sigma_ep, sigma_al
        
class EKFGGDNLLLoss(nn.Module):
    """Generalized Gaussian NLL where alpha = sqrt(sigma_pred_sq) from EKF chain.

    The existing BayesCap1D neural head is NOT used here; alpha is sourced
    directly from the EKF Jacobian propagation. beta is the only learned parameter.
    """

    def __init__(self, eps: float = 1e-8, learn_calibration: bool = False):
        """
        Args:
            eps: numerical floor for alpha to prevent log(0)
            learn_calibration: if True, learn affine (a, b) s.t. alpha = a*sqrt(sigma) + b
        """
        super().__init__()
        # log_beta initialized to 0.5 -> beta = exp(0.5) ~ 1.65 (between Laplace and Gaussian)
        self.log_beta = nn.Parameter(torch.tensor(0.5))
        self.eps = eps
        self.learn_calibration = learn_calibration
        if learn_calibration:
            self.log_a = nn.Parameter(torch.tensor(0.0))  # a = 1.0
            self.b = nn.Parameter(torch.tensor(0.0))       # b = 0.0

    def forward(
        self,
        y_true: torch.Tensor,
        mu_pred: torch.Tensor,
        sigma_pred_sq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_true: (B,) or (B, 1) ground-truth targets
            mu_pred: (B,) or (B, 1) point predictions from frozen model
            sigma_pred_sq: (B,) EKF-propagated predictive variance

        Returns:
            Scalar mean NLL loss
        """
        y_true = y_true.squeeze()
        mu_pred = mu_pred.squeeze()
        beta = torch.exp(self.log_beta)

        if self.learn_calibration:
            a = torch.exp(self.log_a)
            alpha = a * torch.sqrt(sigma_pred_sq + self.eps) + self.b
        else:
            alpha = torch.sqrt(sigma_pred_sq + self.eps)

        alpha = alpha.clamp(min=self.eps)
        residual = torch.abs(y_true - mu_pred)
        nll = (
            (residual / alpha) ** beta
            + torch.log(alpha)
            + torch.lgamma(1.0 / beta)
            - torch.log(beta)
        )
        return nll.mean()

    def extra_repr(self) -> str:
        return (f"beta={torch.exp(self.log_beta).item():.3f}, "
                f"learn_calibration={self.learn_calibration}")

    def get_variance(self, sigma_pred_sq):
        beta = torch.exp(self.log_beta)

        if self.learn_calibration:
            a = torch.exp(self.log_a)
            alpha = a * torch.sqrt(sigma_pred_sq + self.eps) + self.b
        else:
            alpha = torch.sqrt(sigma_pred_sq + self.eps)

        alpha = alpha.clamp(min=self.eps)

        return alpha.pow(2) * torch.exp(torch.lgamma(3 / beta) - torch.lgamma(1 / beta))
>>>>>>> f4ea00808caa3beff81a7be3cbb65ec303cd8b04

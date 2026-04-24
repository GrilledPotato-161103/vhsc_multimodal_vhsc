import torch
import torch.nn as nn
from torch.func import jacrev


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

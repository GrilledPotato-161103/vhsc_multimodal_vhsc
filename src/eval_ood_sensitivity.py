"""
Standalone OOD sensitivity sweep.

Measures the amplitude s(x, z) and propagated sigma_pred_sq for each
sigma_z provider type across a range of input shift magnitudes (delta).
Does NOT require retraining — loads the frozen backbone and uses the
trained EKF head from the latest checkpoint.

Usage:
    python src/eval_ood_sensitivity.py [--ckpt PATH] [--type TYPE] [--label LABEL]

Types: sd | cycle | cycle_iso | gmm | pca
"""
import argparse
import sys
import os

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

import torch
import numpy as np

from src.plugins.sigma_z import SDSigmaZ, CycleSigmaZ, GMMSigmaZ, PCASigmaZ
from src.plugins.ekf_propagation import (
    full_ekf_propagation_full,
    full_ekf_propagation_second_order,
    full_ekf_propagation_mc_dropout,
)
from src.plugins.hook import Breakpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
SOURCE_RANGE = (-1.0, 1.0)
N_SOURCE    = 5000
N_EVAL      = 2000
SHIFTS      = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
EXPRESSION  = "((x1**2 - x2**2) / (torch.abs(x1) + torch.abs(x2) + 0.1)) * torch.abs(1 - torch.sqrt(x1**2 + x2**2))"
# -------------------------------------------------------------------------


def eval_expression(x1, x2):
    import math
    safe_g = {"__builtins__": {}}
    safe_l = {"x1": x1, "x2": x2, "torch": torch, "math": math}
    return eval(EXPRESSION, safe_g, safe_l)


def make_provider(ptype, enc1, enc2, pca_k=2, gmm_k=4):
    kw = dict(encoder1=enc1, encoder2=enc2,
              x_range=SOURCE_RANGE, n_source_samples=N_SOURCE, device=DEVICE)
    if ptype == "sd":
        return SDSigmaZ(**kw)
    if ptype == "cycle":
        return CycleSigmaZ(**kw, phi_mode="sigma_a")
    if ptype == "cycle_iso":
        return CycleSigmaZ(**kw, phi_mode="identity")
    if ptype == "gmm":
        return GMMSigmaZ(**kw, n_clusters=gmm_k)
    if ptype == "pca":
        return PCASigmaZ(**kw, n_components=pca_k)
    raise ValueError(f"Unknown type: {ptype}")


def sweep(provider, backbone, reconstructor_fn_factory, head, delta,
          prop="first_order", K=20, alpha=0.5):
    """Evaluate on N_EVAL samples shifted by delta. Returns dict of stats."""
    torch.manual_seed(42)
    a = SOURCE_RANGE[0] + delta
    b = SOURCE_RANGE[1] + delta
    x1 = torch.rand(N_EVAL, 1, device=DEVICE) * (b - a) + a
    x2 = torch.rand(N_EVAL, 1, device=DEVICE)  # x2 not shifted

    backbone.eval()
    with torch.no_grad():
        z1 = backbone.x1_encoder(x1)
        z2 = backbone.x2_encoder(x2)
    z = torch.cat([z1, z2], dim=-1)

    # Amplitude
    if hasattr(provider, 'amplitude') and callable(getattr(provider, 'amplitude')):
        with torch.no_grad():
            try:
                amp = provider.amplitude(z, x1, x2)
            except TypeError:
                amp = provider.amplitude(z)
    else:
        with torch.no_grad():
            sigma_z_diag = provider(z, x1=x1, x2=x2).diagonal(dim1=-2, dim2=-1)
        amp = sigma_z_diag.mean(-1)

    # Propagation
    with torch.enable_grad():
        recon_fn = reconstructor_fn_factory(signal=(1, 1))
        sigma_z = provider(z, x1=x1, x2=x2)

        if prop == "first_order":
            sps, _, _, _ = full_ekf_propagation_full(z, sigma_z, recon_fn, head.forward)

        elif prop == "second_order":
            sps, _, _, _ = full_ekf_propagation_second_order(z, sigma_z, recon_fn, head.forward)

        elif prop == "mc_dropout":
            # Blend: (1-alpha)*EKF + alpha*MC
            sps, _, _, _ = full_ekf_propagation_mc_dropout(
                z, sigma_z, recon_fn, head, K=K, blend_alpha=alpha)

        elif prop == "mc_only":
            # Pure MC dropout (amplitude from sigma_z is ignored for propagation)
            from torch.func import vmap
            with torch.no_grad():
                z_recon = vmap(recon_fn)(z)
            from src.plugins.ekf_propagation import mc_dropout_propagation
            sps = mc_dropout_propagation(head, z_recon, K=K)

        else:
            raise ValueError(f"Unknown prop: {prop!r}")

    return {
        "delta":    delta,
        "amp_mean": amp.mean().item(),
        "amp_std":  amp.std().item(),
        "sps_mean": sps.mean().item(),
        "sps_std":  sps.std().item(),
        "sps_max":  sps.max().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default="data/checkpoints/checkpoint.pth")
    parser.add_argument("--type",  default="sd",
                        help="sigma_z provider: sd|cycle|cycle_iso|gmm|pca")
    parser.add_argument("--prop",  default="first_order",
                        help="propagation: first_order|second_order|mc_dropout|mc_only")
    parser.add_argument("--K",     type=int, default=20,
                        help="MC dropout samples (for mc_dropout/mc_only)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="blend weight for mc_dropout mode (0=EKF, 1=MC)")
    parser.add_argument("--pca_k",     type=int, default=2,
                        help="Number of PCA components for PCASigmaZ")
    parser.add_argument("--gmm_k",     type=int, default=4,
                        help="Number of GMM clusters for GMMSigmaZ")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    label = args.label or f"{args.type}+{args.prop}"
    print(f"\n=== OOD sensitivity sweep: type={args.type}  prop={args.prop}  label={label} ===")

    # Load frozen backbone
    backbone = torch.load(args.ckpt, weights_only=False, map_location=DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Build provider
    provider = make_provider(args.type, backbone.x1_encoder, backbone.x2_encoder,
                             pca_k=args.pca_k, gmm_k=args.gmm_k)
    provider.eval()

    # Reconstructor factory: signal=(1,1) means passthrough (J_f = I)
    # We load the latest EKF checkpoint to get the trained reconstructor
    ckpt_dir = "logs/train/runs"
    runs = sorted(os.listdir(ckpt_dir)) if os.path.exists(ckpt_dir) else []
    reconstructor_fn = None
    head = backbone.head

    if runs:
        latest = os.path.join(ckpt_dir, runs[-1], "checkpoints")
        ckpts = sorted(os.listdir(latest)) if os.path.exists(latest) else []
        if ckpts:
            import functools
            torch.serialization.add_safe_globals([functools.partial])
            from lightning import LightningModule
            lit_ckpt = os.path.join(latest, ckpts[-1])
            print(f"  loading EKF module from {lit_ckpt}")
            try:
                # load module to get the reconstructor callback
                from src.models.hook_ekf_module import ModelEKFInjectModule
                mod = LightningModule.load_from_checkpoint(
                    lit_ckpt, weights_only=False, map_location=DEVICE,
                    net=backbone, ekf_net=None, recon_bp="reconstructor.0",
                    strict=False)
                recon_module = mod.recon_bp.callback
                def reconstructor_fn_factory(signal=(1, 1)):
                    from src.plugins.ekf_propagation import make_reconstructor_fn
                    return make_reconstructor_fn(recon_module, signal)
            except Exception as e:
                print(f"  warning: could not load EKF module ({e}), using identity reconstructor")

    if reconstructor_fn is None:
        def reconstructor_fn_factory(signal=(1, 1)):
            def fn(z): return z  # passthrough = I
            return fn

    # Run sweep
    print(f"\n{'delta':>6}  {'amp_mean':>12}  {'sps_mean':>12}  {'sps_max':>12}")
    print("-" * 50)
    results = []
    for delta in SHIFTS:
        r = sweep(provider, backbone, reconstructor_fn_factory, head, delta,
                  prop=args.prop, K=args.K, alpha=args.alpha)
        results.append(r)
        print(f"  {delta:4.2f}   {r['amp_mean']:12.4e}   {r['sps_mean']:12.4e}   {r['sps_max']:12.4e}")

    # Print as markdown table row for the log
    print(f"\n### Result row (for log):")
    print(f"| {label} | " + " | ".join(f"{r['amp_mean']:.3e}" for r in results) + " |")
    print(f"| {label} (sps) | " + " | ".join(f"{r['sps_mean']:.3e}" for r in results) + " |")

    # Sensitivity ratio (delta=1.0 vs delta=0.0)
    r0 = next(r for r in results if r['delta'] == 0.0)
    r1 = next((r for r in results if r['delta'] == 1.0), results[-1])
    amp_ratio = r1['amp_mean'] / max(r0['amp_mean'], 1e-12)
    sps_ratio = r1['sps_mean'] / max(r0['sps_mean'], 1e-12)
    print(f"\n  OOD/ID amplitude ratio (delta=1.0 / delta=0.0): {amp_ratio:.3f}")
    print(f"  OOD/ID sps ratio (delta=1.0 / delta=0.0):       {sps_ratio:.3f}")
    print(f"\n  VERDICT: {'SENSITIVE ✓' if amp_ratio > 1.5 else 'NOT SENSITIVE ✗'} "
          f"(threshold >1.5x)")

    return results


if __name__ == "__main__":
    main()

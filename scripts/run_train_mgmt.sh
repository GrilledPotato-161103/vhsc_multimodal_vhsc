#!/usr/bin/env bash
# =============================================================================
# MGMT Classification — Training & Evaluation
# =============================================================================
#
# Two-phase training with frozen → unfrozen ViT backbone.
# Uses configs/train_mgmt.yaml (Hydra composition).
#
# Prerequisites:
#   1. Run scripts/generate_mgmt_splits.py first (produces labels.csv)
#   2. BraTS DICOM data converted to NIfTI under data/jepa/brats_radiogenomic/nifti/
#   3. UCSF-PDGM NIfTI data under data/jepa/ucsf_pdgm/
#
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../venv/python.exe}"

echo "============================================"
echo "MGMT Classification — Training"
echo "============================================"

# ---------------------------------------------------------------------------
# In-domain training (BraTS Radiogenomic)
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Training on BraTS Radiogenomic (in-domain) ..."
$PYTHON "$ROOT/src/train/train_mgmt_classification.py" \
    data=brats_radiogenomic \
    model=mgmt_classification \
    trainer.max_epochs=50 \
    seed=42

# ---------------------------------------------------------------------------
# Out-of-domain evaluation (UCSF-PDGM, disjoint test_ood)
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] OOD evaluation on UCSF-PDGM (disjoint cohort) ..."
$PYTHON "$ROOT/src/train/train_mgmt_classification.py" \
    data=ucsf_pdgm \
    model=mgmt_classification \
    train=false \
    test=true \
    ckpt_path="$(ls -t "$ROOT/logs/train_mgmt/runs/"*/checkpoints/last.ckpt 2>/dev/null | head -1)"

echo ""
echo "Done."

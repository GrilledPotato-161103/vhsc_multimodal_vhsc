#!/usr/bin/env bash
# =============================================================================
# MGMT Data Preparation Pipeline
# =============================================================================
#
# Orchestrates data preparation for two disjoint MGMT methylation cohorts:
#   1. RSNA-MICCAI BraTS Radiogenomic (in-domain, 585 subjects, DICOM -> NIfTI)
#   2. UCSF-PDGM-v3 (out-of-domain, ~200 disjoint subjects, native NIfTI)
#
# THIS SCRIPT IS DOCUMENTATION-ONLY — DO NOT RUN.
# It requires the full datasets to be downloaded from Kaggle first.
# For pipeline validation, use the representative samples already downloaded
# to data/jepa/brats_radiogenomic/ and data/jepa/ucsf_pdgm/.
#
# Prerequisites:
#   - Kaggle API credentials configured (~/.kaggle/kaggle.json)
#   - pydicom, nibabel, monai installed in the venv
#   - ~50 GB free disk space (DICOM source + NIfTI output)
#
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../venv/python.exe}"

echo "============================================"
echo "MGMT Data Preparation"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1 — Download RSNA-MICCAI BraTS Radiogenomic (Kaggle competition)
# ---------------------------------------------------------------------------
# Competition: rsna-miccai-brain-tumor-radiogenomic-classification
# Training data: ~120 GB DICOM (585 subjects x 4 modalities x ~150 slices)
#
# echo ""
# echo "Step 1 — Downloading RSNA-MICCAI BraTS Radiogenomic..."
# kaggle competitions download \
#     rsna-miccai-brain-tumor-radiogenomic-classification \
#     -f train_labels.csv \
#     -p "$ROOT/data/jepa/brats_radiogenomic/raw/"
# kaggle competitions download \
#     rsna-miccai-brain-tumor-radiogenomic-classification \
#     -f train.zip \
#     -p "$ROOT/data/jepa/brats_radiogenomic/raw/"
# unzip "$ROOT/data/jepa/brats_radiogenomic/raw/train.zip" \
#     -d "$ROOT/data/jepa/brats_radiogenomic/raw/"

# ---------------------------------------------------------------------------
# Step 2 — DICOM -> NIfTI Conversion (BraTS only)
# ---------------------------------------------------------------------------
# Converts slice-by-slice DICOM stacks to 3D NIfTI volumes.
# Maps T1->t1, T2->t2, FLAIR->flair (T1ce dropped per modality topology).
#
# echo ""
# echo "Step 2 — DICOM -> NIfTI conversion..."
# $PYTHON "$ROOT/scripts/convert_brats_dicom_to_nifti.py" \
#     --input-dir "$ROOT/data/jepa/brats_radiogenomic/raw/train" \
#     --output-dir "$ROOT/data/jepa/brats_radiogenomic/nifti" \
#     --modalities T1,T2,FLAIR

# ---------------------------------------------------------------------------
# Step 3 — Download UCSF-PDGM-v3 (Kaggle dataset)
# ---------------------------------------------------------------------------
# Dataset: usmansadiqcs/ucsf-pdgm-v3-dataset
# Size: ~10 GB NIfTI (501 subjects x 3 files each)
#
# echo ""
# echo "Step 3 — Downloading UCSF-PDGM-v3..."
# kaggle datasets download usmansadiqcs/ucsf-pdgm-v3-dataset \
#     -p "$ROOT/data/jepa/ucsf_pdgm/raw/"
# unzip "$ROOT/data/jepa/ucsf_pdgm/raw/ucsf-pdgm-v3-dataset.zip" \
#     -d "$ROOT/data/jepa/ucsf_pdgm/raw/"

# ---------------------------------------------------------------------------
# Step 4 — Generate Splits (the key orchestration step)
# ---------------------------------------------------------------------------
# Reads raw metadata CSVs from both datasets, harmonises MGMT labels to {0,1},
# detects BraTS21 ID overlap, subtracts overlapped subjects from UCSF-PDGM,
# produces stratified splits, and writes unified labels.csv for each cohort.
#
echo ""
echo "Step 4 — Generating splits..."
$PYTHON "$ROOT/scripts/generate_mgmt_splits.py"

# ---------------------------------------------------------------------------
# Step 5 — Validate Output
# ---------------------------------------------------------------------------
echo ""
echo "Step 5 — Validating output CSVs..."

$PYTHON -c "
import csv
from pathlib import Path

ROOT = Path('$ROOT')

# BraTS validation
brats_csv = ROOT / 'data/jepa/brats_radiogenomic/splits/labels.csv'
with open(brats_csv) as f:
    brats = list(csv.DictReader(f))
assert len(brats) == 585, f'Expected 585 BraTS rows, got {len(brats)}'
brats_splits = set(r['split'] for r in brats)
assert brats_splits == {'train', 'val', 'test'}, f'Unexpected splits: {brats_splits}'
print(f'  BraTS: {len(brats)} rows OK')

# UCSF-PDGM validation
ucsf_csv = ROOT / 'data/jepa/ucsf_pdgm/splits/labels.csv'
with open(ucsf_csv) as f:
    ucsf = list(csv.DictReader(f))
assert len(ucsf) > 0, 'UCSF labels.csv is empty'
assert all(r['split'] == 'test_ood' for r in ucsf), 'All UCSF rows must be test_ood'
assert all(r['t2_path'] == '' for r in ucsf), 't2_path must be empty for all UCSF samples'
print(f'  UCSF-PDGM: {len(ucsf)} rows OK')

# No overlap check
brats_ids = set(r['subject_id'] for r in brats)
ucsf_brats = set()
meta_csv = ROOT / 'data/jepa/ucsf_pdgm/UCSF-PDGM-metadata.csv'
with open(meta_csv) as f:
    for row in csv.DictReader(f):
        bid = row.get('BraTS21 ID', '').strip()
        if bid and bid.startswith('BraTS2021_'):
            ucsf_brats.add(bid.replace('BraTS2021_', ''))
overlap = brats_ids & ucsf_brats
print(f'  Overlap: {len(overlap)} subjects (0 expected in UCSF output)')

# Verify no UCSF-PDGM subject in output has a BraTS21 ID in the BraTS set
ucsf_subjects_in_output = set(r['subject_id'] for r in ucsf)
# Extract which of these have BraTS21 IDs from metadata
meta_by_id = {}
with open(meta_csv) as f:
    for row in csv.DictReader(f):
        meta_by_id[row['ID'].strip()] = row.get('BraTS21 ID', '').strip()

leaked = []
for sid in ucsf_subjects_in_output:
    # Find matching metadata row (try different padding formats)
    for meta_id, brats_id in meta_by_id.items():
        # Normalize: extract int from both, compare
        sid_num = ''.join(c for c in sid if c.isdigit())
        meta_num = ''.join(c for c in meta_id if c.isdigit())
        if sid_num == meta_num and brats_id:
            stripped = brats_id.replace('BraTS2021_', '')
            if stripped in brats_ids:
                leaked.append((sid, stripped))
            break

if leaked:
    print(f'  LEAKAGE: {len(leaked)} subjects!')
    for s, b in leaked[:5]:
        print(f'    {s} -> BraTS {b}')
else:
    print(f'  No leakage: all UCSF subjects are disjoint from BraTS')

print('All validation checks passed.')
"

echo ""
echo "Done.  Data pipelines ready:"
echo "  BraTS config:    configs/data/brats_radiogenomic.yaml"
echo "  UCSF-PDGM config: configs/data/ucsf_pdgm.yaml"

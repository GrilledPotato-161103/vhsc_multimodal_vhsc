# UCSF-PDGM-v3 & RSNA-MICCAI BraTS Radiogenomic — Joint Analysis & Curation Plan

## Overview

Two publicly available brain tumor MRI datasets sharing a common molecular target:
**MGMT promoter methylation status** (binary classification). This document
analyses both datasets and defines a curation plan to produce two disjoint,
task-aligned cohorts for Neuro-JEPA multimodal training with the fixed modality
topology `['t1', 't2', 'flair']`.

---

## 1. Dataset Summaries

### 1.1 UCSF-PDGM-v3

| | |
|---|---|
| **Source** | [Kaggle](https://www.kaggle.com/datasets/usmansadiqcs/ucsf-pdgm-v3-dataset) |
| **License** | Apache 2.0 |
| **Original paper** | Calabrese et al., _Radiology_ 2022 |
| **Subjects** | 501 |
| **Format** | NIfTI (.nii), uncompressed |
| **Modalities provided** | FLAIR (bias-corrected), T1c/T1gad (bias-corrected), tumor segmentation mask |
| **Shape** | `(240, 240, 155)` — 1 mm³ isotropic |
| **dtype** | int16 (FLAIR, T1c), uint8 (segmentation) |

**Labels (from `UCSF-PDGM-metadata.csv`):**

| Column | Description | Distribution |
|---|---|---|
| `ID` | Subject ID (`UCSF-PDGM-XXXX`) | 501 unique |
| `Sex` | M / F | — |
| `Age at MRI` | Years | Range 17–94 |
| `WHO CNS Grade` | 2 / 3 / 4 | G2: 56, G3: 43, G4: 402 |
| `Final pathologic diagnosis (WHO 2021)` | Free-text | Glioblastoma (majority), Astrocytoma, Oligodendroglioma |
| `MGMT status` | positive / negative / indeterminate / _(empty)_ | pos: 302, neg: 114, empty: 80, indet: 5 |
| `MGMT index` | Numeric (0–17) when available | Sparse |
| `1p/19q` | co-deletion / intact / relative co-deletion / _(empty)_ | — |
| `IDH` | wildtype / mutated (NOS) / specific mutations | wt: 398, mut: 103 |
| `1-dead 0-alive` | Event indicator | — |
| `OS` | Overall survival (days) | Range 6–4177 |
| `EOR` | Extent of resection (GTR / STR / biopsy) | — |
| `Biopsy prior to imaging` | Yes / No | — |
| `BraTS21 ID` | Cross-reference to BraTS 2021 | 298 of 501 have IDs |
| `BraTS21 Segmentation Cohort` | Training / Validation / _(empty)_ | — |
| `BraTS21 MGMT Cohort` | Training / Validation / _(empty)_ | — |

**Key observations:**
- **Only 2 imaging modalities** (FLAIR + T1c). No native T1, T2, or T1ce.
- MGMT status is missing for 80 subjects (mostly Grade 2/3 gliomas where MGMT
  testing is not standard of care) and indeterminate for 5.
- **298 subjects overlap with BraTS 2021** via `BraTS21 ID` — necessitates
  patient-level deduplication against RSNA-MICCAI.
- 10 IDH-mutant astrocytomas among the Grade 4 group (WHO 2021 reclassification).

### 1.2 RSNA-MICCAI Brain Tumor Radiogenomic Classification

| | |
|---|---|
| **Source** | [Kaggle Competition](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification) |
| **License** | Competition rules + BraTS data usage (CC BY-NC 4.0) |
| **Subjects** | 585 (training set with labels) |
| **Format** | DICOM (.dcm), one file per slice |
| **Modalities provided** | T1, T1ce, T2, FLAIR (4 sequences) |
| **Approx. shape** | `(240, 240, ~155)` — 1 mm³ isotropic (BraTS standard) |
| **dtype** | uint16 (DICOM standard) |

**Labels (from `train_labels.csv`):**

| Column | Description | Distribution |
|---|---|---|
| `BraTS21ID` | 5-digit subject ID (zero-padded) | 585 unique |
| `MGMT_value` | 0 = unmethylated, 1 = methylated | 0: 278, 1: 307 |

**Key observations:**
- **4 full modalities** — closest to Neuro-JEPA's pretraining distribution.
- **DICOM format** — slice-by-slice; each modality stored as ~150 separate
  `.dcm` files in `{subject}/{modality}/Image-{n}.dcm` structure.
- Balanced classes (278/307, ~48%/52%).
- No segmentation masks, no survival data, no IDH status — MGMT only.

---

## 2. Data Comparison Matrix

| | UCSF-PDGM-v3 | RSNA-MICCAI BraTS |
|---|---|---|
| **N (usable)** | 416 | 585 |
| **Task** | MGMT status (+/-) | MGMT_value (1/0) |
| **Class balance** | 302 pos / 114 neg (2.6:1) | 307 pos / 278 neg (1.1:1) |
| **Imaging modalities** | 2 (FLAIR + T1c) | 4 (T1 + T1ce + T2 + FLAIR) |
| **Shape** | 240×240×155 | 240×240×~155 (DICOM) |
| **Voxel** | 1 mm³ isotropic | 1 mm³ isotropic |
| **Format** | NIfTI (.nii) | DICOM (.dcm) |
| **Segmentation masks** | Yes (3-class tumor subregions) | No |
| **Survival data** | Yes (OS days + event) | No |
| **IDH status** | Yes | No |
| **Overlap with other** | 298 in BraTS21 | 298 UCSF-PDGM subjects in BraTS21 |

---

## 3. Modality Mapping to Neuro-JEPA Topology

Fixed topology: `['t1', 't2', 'flair']`

### UCSF-PDGM-v3 (2 available)

| Topology key | Source modality | Status |
|---|---|---|
| `t1` | T1c (T1gad, post-contrast T1) | Present |
| `t2` | — | **Missing** → imputed by `MeanImputeReconstructor` |
| `flair` | FLAIR (bias-corrected) | Present |

The `t2` slot is filled by `mean(t1_latent, flair_latent)` at the classifier
prefill breakpoint.  This is the exact use-case the `MeanImputeReconstructor`
was designed for.

### RSNA-MICCAI (4 available → 3 used)

| Topology key | Source modality | Status |
|---|---|---|
| `t1` | T1 (native, pre-contrast) | Present |
| `t2` | T2 | Present |
| `flair` | FLAIR | Present |

T1ce is **dropped** (no dedicated slot in the 3-modality topology).
Alternative: merge T1 + T1ce via early fusion or use T1ce as `t1` for
consistency with UCSF-PDGM's T1c-only protocol.  Decision deferred to
preprocessing config.

---

## 4. Subject Overlap & Deduplication

298 UCSF-PDGM subjects have `BraTS21 ID` values (e.g., `BraTS2021_00097`).
The RSNA-MICCAI competition uses zero-padded 5-digit IDs (e.g., `00097`).

**Overlap**: These are the **same patients** appearing in both datasets.
UCSF-PDGM contributed its imaging + clinical data to the BraTS 2021 challenge;
the RSNA-MICCAI competition drew its training cases from the BraTS 2021 pool.

### Deduplication Strategy

```
UCSF-PDGM BraTS21 ID:  "BraTS2021_00097"  →  strip prefix → "00097"
RSNA-MICCAI BraTS21ID: "00097"            →  direct match
```

**298 UCSF-PDGM subjects carry BraTS21 IDs.**  To produce truly disjoint
cohorts, matching subjects are **subtracted from UCSF-PDGM**.
RSNA-MICCAI is left intact — all 585 subjects are kept.

Rationale: RSNA-MICCAI is the primary training cohort (4 full modalities,
balanced labels).  Keeping it whole maximizes in-domain training signal.
UCSF-PDGM, already limited to 2 modalities, absorbs the subtraction;
subjects that overlap with BraTS are removed, leaving only the
institution-unique cases for out-of-domain evaluation.

```
RSNA-MICCAI after subtraction: 585 subjects (unchanged)
UCSF-PDGM after subtraction:   416 − |overlap| ≈ 118–203 subjects
```

### Split Design: In-Domain / Out-of-Domain

| Split | Cohort | N | Role |
|---|---|---|---|
| **train** | RSNA-MICCAI (4-modality, intact) | 400 | In-domain training |
| **val** | RSNA-MICCAI (4-modality, intact) | 85 | In-domain validation |
| **test** | RSNA-MICCAI (4-modality, intact) | 100 | In-domain held-out test |
| **test_ood** | UCSF-PDGM (2-modality, disjoint) | ~200 | Out-of-domain generalization test |

**In-domain** (RSNA-MICCAI, intact): Same acquisition protocol, 4 modalities
present.  Evaluates whether the model learns the MGMT signal when all
modalities are available.

**Out-of-domain** (UCSF-PDGM, disjoint subset): Different institution, only
2 modalities (FLAIR + T1c), different scanner vendors.  Evaluates (a)
cross-cohort generalization under distribution shift and (b) missing-modality
imputation (`t2` always imputed by `MeanImputeReconstructor`).

**Disjoint guarantee:** Any UCSF-PDGM subject whose `BraTS21 ID` (after
stripping the `BraTS2021_` prefix) matches an RSNA-MICCAI `BraTS21ID` is
removed from UCSF-PDGM.  The two cohorts share zero patients.

---

## 5. Curation Plan

### Phase 1 — Label Harmonization

**Target:** `mgmt_label ∈ {0, 1}` for both datasets.

| Dataset | Source column | Mapping |
|---|---|---|
| RSNA-MICCAI | `MGMT_value` | Direct: `0` → 0, `1` → 1 |
| UCSF-PDGM | `MGMT status` | `"negative"` → 0, `"positive"` → 1, `""` / `"indeterminate"` → **exclude** |

UCSF-PDGM exclusion: 80 empty + 5 indeterminate = **85 subjects excluded**.
Usable: **416 subjects** (302 pos, 114 neg).

### Phase 2 — DICOM → NIfTI Conversion (RSNA-MICCAI only)

For each subject × modality:

```
train/{subject}/{Modality}/Image-{1..N}.dcm
    ↓ pydicom: sort by SliceLocation / InstanceNumber
    ↓ stack into 3D numpy array (H, W, N_slices)
    ↓ nibabel: save as {subject}_{modality}.nii.gz
    ↓ verify affine = 1 mm³ isotropic RAS
```

Conversion script: `scripts/convert_brats_dicom_to_nifti.py` (to be built with data pipeline).

### Phase 3 — Intensity Preprocessing

Both datasets undergo the same MONAI transform chain:

1. **Load**: `LoadImaged(keys=[...], ensure_channel_first=True)` → `(1, D, H, W)`
2. **Resize to ROI**: `Resized(keys=[...], spatial_size=(96, 108, 96), mode="trilinear")`
3. **Intensity clip**: Percentile-based (0.5–99.5) per volume
4. **Normalize**: Min-max → [0, 1] or z-score per volume
5. **Cast**: `ToTensord(keys=[...], dtype=torch.float32)`

UCSF-PDGM volumes are already skull-stripped and bias-corrected.
RSNA-MICCAI volumes are skull-stripped (BraTS convention) but not bias-corrected.

### Phase 4 — Split Definition & Label CSV Format

Each dataset produces its own `labels.csv` with an identical 6-column schema.
The pipelines are **independent** — no shared DataModule, no combined CSV.

**Unified label CSV schema (both datasets):**

| Column | Type | Description |
|---|---|---|
| `subject_id` | str | Unique subject identifier |
| `mgmt_label` | int | 0 = unmethylated, 1 = methylated |
| `split` | str | `train` / `val` / `test` / `test_ood` |
| `t1_path` | str | Relative path to T1 (or T1c) NIfTI, or empty if missing |
| `t2_path` | str | Relative path to T2 NIfTI, or empty if missing |
| `flair_path` | str | Relative path to FLAIR NIfTI, or empty if missing |

**RSNA-MICCAI splits** (in-domain, 585 subjects intact):

```
train: 400 (68%)  — stratified on mgmt_label
val:   85 (15%)  — stratified on mgmt_label
test: 100 (17%)  — stratified on mgmt_label
```

**UCSF-PDGM splits** (out-of-domain, overlap subtracted):

```
test_ood: ~200   — usable subjects NOT in the BraTS21 overlap set
```

UCSF-PDGM subjects whose `BraTS21 ID` (after stripping `BraTS2021_` prefix)
matches any RSNA-MICCAI `BraTS21ID` are **permanently excluded**.  Only the
~118 subjects without BraTS21 IDs plus any BraTS21 subjects not present in
RSNA-MICCAI's `train_labels.csv` form the `test_ood` set.

**Example RSNA-MICCAI** `data/jepa/brats_radiogenomic/splits/labels.csv`:

```csv
subject_id,mgmt_label,split,t1_path,t2_path,flair_path
00000,1,train,00000_t1.nii.gz,00000_t2.nii.gz,00000_flair.nii.gz
00001,0,train,00001_t1.nii.gz,00001_t2.nii.gz,00001_flair.nii.gz
...
```

**Example UCSF-PDGM** `data/jepa/ucsf_pdgm/splits/labels.csv`:

```csv
subject_id,mgmt_label,split,t1_path,t2_path,flair_path
UCSF-PDGM-0004,0,test_ood,UCSF-PDGM-0004_T1gad_bias.nii,,UCSF-PDGM-0004_FLAIR_bias.nii
UCSF-PDGM-0010,1,test_ood,UCSF-PDGM-0010_T1gad_bias.nii,,UCSF-PDGM-0010_FLAIR_bias.nii
```

UCSF-PDGM's `t2_path` is always empty — the `MeanImputeReconstructor` fills
it at inference time.

### Phase 5 — Directory Structure

Each dataset has its own self-contained directory tree.  No shared or
combined splits — the pipelines are independent.

```
data/jepa/
├── brats_radiogenomic/                   # In-domain (RSNA-MICCAI, intact)
│   ├── raw/                              # DICOM directories (per subject/modality)
│   ├── nifti/                            # Converted .nii.gz files
│   │   ├── 00000_t1.nii.gz
│   │   ├── 00000_t2.nii.gz
│   │   ├── 00000_flair.nii.gz
│   │   └── ...
│   ├── preprocessed/                     # MONAI-cached, resized to 96×108×96
│   └── splits/
│       └── labels.csv                    # split ∈ {train, val, test}
│
└── ucsf_pdgm/                            # Out-of-domain (UCSF-PDGM, intact)
    ├── raw/                              # Original .nii files + metadata.csv
    ├── preprocessed/                     # MONAI-cached, resized
    │   ├── UCSF-PDGM-0004_t1.nii.gz
    │   ├── UCSF-PDGM-0004_flair.nii.gz
    │   └── ...
    └── splits/
        └── labels.csv                    # split = test_ood
```

### Phase 6 — Distinct Data Pipelines

Two independent DataModules, each reading its own `labels.csv`.  No shared
code path between datasets — they differ in modality count, file format
history (DICOM→NIfTI vs native NIfTI), and split structure.

**RSNA-MICCAI DataModule** (in-domain, 585 subjects intact):

```python
# configs/data/brats_radiogenomic.yaml
_target_: src.data.brats_radiogenomic.datamodule.BraTSRadiogenomicDataModule
labels_csv: data/jepa/brats_radiogenomic/splits/labels.csv
modality_keys: [t1, t2, flair]
batch_size: 1
num_workers: 0
# Reads split column — train/val/test partitions pre-computed.
```

**UCSF-PDGM DataModule** (out-of-domain, disjoint subset):

```python
# configs/data/ucsf_pdgm.yaml
_target_: src.data.ucsf_pdgm.datamodule.UCSFPDGMDataModule
labels_csv: data/jepa/ucsf_pdgm/splits/labels.csv
modality_keys: [t1, t2, flair]
batch_size: 1
num_workers: 0
# Single test_ood partition.  t2_path is always empty → None → imputed.
```

**Key design points:**
- Both DataModules share the same label CSV schema (6 columns) so the
  dataset-level parsing is identical, but the DataModules themselves are
  separate classes — each can evolve independently (e.g., UCSF-PDGM may
  later add segmentation mask loading, RSNA-MICCAI may add T1ce).
- Splits are pre-computed and baked into `labels.csv` — no runtime splitting.
- Missing modalities (empty path in CSV) → dataset returns `None` for that
  key → JEPA forward pass skips the encoder → `MeanImputeReconstructor`
  imputes at the classifier hook.
- In-domain eval: RSNA-MICCAI `test` split (100 subjects, intact), all 3
  modalities present.
- Out-of-domain eval: UCSF-PDGM `test_ood` split (~200 subjects, disjoint),
  only 2 modalities, `t2` always imputed.

---

## 6. Neuro-JEPA Compatibility Checklist

| Requirement | UCSF-PDGM | RSNA-MICCAI |
|---|---|---|
| 3D volumetric MRI | Yes | Yes (after DICOM stack) |
| MNI152 or MNI-aligned space | 1 mm³ isotropic, BraTS-standard | 1 mm³ isotropic, BraTS-standard |
| Single-channel per modality | Yes | Yes |
| Skull-stripped | Yes | Yes |
| Input resize to (96, 108, 96) | Trilinear resize | Trilinear resize |
| Patient-level splits | Yes (via metadata) | Yes (competition splits) |
| No same-patient leakage | Yes — overlapped subjects subtracted | Yes — kept intact as reference |
| Publicly downloadable | Yes (Kaggle) | Yes (Kaggle competition) |
| Analogous `labels.csv` schema | Yes — same 6 columns | Yes — same 6 columns |
| Independent data pipeline | `UCSFPDGMDataModule` | `BraTSRadiogenomicDataModule` |
| In-domain / out-of-domain eval | `test_ood` (~200, disjoint) | `train` / `val` / `test` (585, intact) |

---

## 7. Implementation Sequence

Two independent pipeline tracks, built in parallel.  The split generation
script runs first and produces both `labels.csv` files.

### Track A — RSNA-MICCAI (in-domain)

1. **DICOM conversion script** — `scripts/convert_brats_dicom_to_nifti.py`
2. **Dataset class** — `src/data/brats_radiogenomic/dataset.py`
3. **DataModule** — `src/data/brats_radiogenomic/datamodule.py`
4. **Transform config** — `configs/data/transform/brats_radiogenomic.yaml`
5. **Data config** — `configs/data/brats_radiogenomic.yaml`

### Track B — UCSF-PDGM (out-of-domain)

1. **Dataset class** — `src/data/ucsf_pdgm/dataset.py` (native NIfTI, no DICOM step)
2. **DataModule** — `src/data/ucsf_pdgm/datamodule.py`
3. **Transform config** — `configs/data/transform/ucsf_pdgm.yaml`
4. **Data config** — `configs/data/ucsf_pdgm.yaml`

### Shared

1. **Split generation script** — `scripts/generate_mgmt_splits.py`
   - Reads `train_labels.csv` (RSNA-MICCAI) and `UCSF-PDGM-metadata.csv`
   - Harmonizes MGMT labels → `{0, 1}` for both sources
   - Cross-references BraTS21 IDs, **subtracts** matching subjects from UCSF-PDGM
   - RSNA-MICCAI: stratified split → train/val/test (585 subjects, intact)
   - UCSF-PDGM: single partition → test_ood (~200 subjects, disjoint)
   - Writes `labels.csv` to each dataset's `splits/` directory (same 6-column schema)
2. **Model config** — `configs/model/hook_dag_mgmt.yaml` (reuses `hook_dag_jepa_multimodal.yaml`)
3. **Training config** — `configs/train_hook_dag_mgmt.yaml` (points to RSNA-MICCAI DataModule)
4. **OOD evaluation config** — `configs/eval_hook_dag_mgmt_ood.yaml` (points to UCSF-PDGM DataModule)

---

## 8. Sample Label Preview

### UCSF-PDGM-v3 (4 downloaded samples)

| Subject | Grade | Diagnosis | MGMT | IDH | BraTS21 ID |
|---|---|---|---|---|---|
| UCSF-PDGM-0004 | 4 | GBM, IDH-wt | negative | wildtype | BraTS2021_00097 |
| UCSF-PDGM-0010 | 4 | GBM, IDH-wt | positive | wildtype | BraTS2021_00087 |
| UCSF-PDGM-0011 | 4 | GBM, IDH-wt | positive | wildtype | BraTS2021_00068 |
| UCSF-PDGM-0021 | 4 | Astrocytoma, IDH-mutant | positive | mutated (NOS) | BraTS2021_00085 |

### RSNA-MICCAI (matching BraTS21 IDs where applicable)

| BraTS21ID | MGMT_value |
|---|---|
| 00097 | 0 |
| 00087 | 1 |
| 00068 | 1 |
| 00085 | 1 |

MGMT labels agree for all 4 cross-referenced subjects (UCSF-PDGM "negative" = RSNA-MICCAI "0", "positive" = "1").

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| UCSF-PDGM only has 2 modalities → `t2` always imputed | Accept as realistic missing-modality test case; `MeanImputeReconstructor` designed for this |
| DICOM→NIfTI conversion may lose affine metadata | Verify against BraTS NIfTI releases; validate voxel spacing post-conversion |
| 298-subject overlap → data leakage | Strict patient-level dedup; exclude overlapped subjects from test |
| Class imbalance in UCSF-PDGM (2.6:1) | Use weighted BCE loss or stratified sampling |
| UCSF-PDGM T1c vs RSNA-MICCAI T1 are different contrasts | Test impact by comparing T1-only vs T1c-only encoder outputs; consider domain adaptation if gap is large |
| No native T1 in UCSF-PDGM for `t1` slot | Accept T1c as `t1` proxy; post-contrast T1 is standard in glioma protocols |

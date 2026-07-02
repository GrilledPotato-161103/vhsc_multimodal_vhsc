# Neuro-JEPA: Complete Dataset Survey

## Overview

Neuro-JEPA was evaluated across **47 downstream tasks** spanning **12 public
research cohorts** and **3 large clinical cohorts** (NYU Langone, NYU Long
Island, MGH). All pretraining was done on ~1.55 million curated T1w, T2w, and
FLAIR scans from NYU Langone Health (internal — not publicly released).

This document surveys every dataset used in downstream evaluation, its access
terms, and compatibility with the Neuro-JEPA pipeline.

---

## JEPA Input Requirements

All datasets must satisfy these pipeline constraints for direct use:

| Requirement | Specification |
|---|---|
| Input shape | `[B, 1, 96, 108, 96]` (single-channel 3D volume) |
| Spatial registration | MNI152 space (T1w→MNI152 T1 1mm; T2w/FLAIR→MNI ICBM 152 T2) |
| File format | NIfTI (.nii.gz) with MONAI PersistentDataset caching |
| Data split | Patient-level (no same-patient leakage across train/val/test) |
| Modalities | Any subset of {T1w, T2w, FLAIR, DWI} — each encoded independently by shared ViT |

**Preprocessing pipeline**: Registration to MNI → defacing → NIfTI conversion →
MONAI `PersistentDataset` caching with `loading_transforms` (resampling to
96×108×96, intensity normalization).

---

## Dataset Access Tiers

### Tier 1: Truly Open — No DUA Required

Immediate download with free registration only. Best starting point for
reproducing Neuro-JEPA results.

#### UCSF-PDGM — Preoperative Diffuse Glioma MRI

| | |
|---|---|
| **Access** | [TCIA Collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) |
| **License** | CC BY 4.0 (commercial use permitted with attribution) |
| **Size** | 501 subjects, ~156 GB (NIfTI) |
| **Modalities** | T1, T1c, T2, FLAIR, DWI (b=1000), DSC perfusion |
| **Tasks** | IDH mutation (binary), Overall survival (TTE), MGMT methylation |
| **Train / Val / Test** | 246 / 124 / 125 (per modality) |
| **JEPA compatibility** | Full — all required modalities present; MNI registration scripts provided in Neuro-JEPA repo |
| **Notes** | Multimodal + unimodal splits provided. TTE endpoint uses OS (days) + event indicator. Download via NBIA Data Retriever. |

#### SOOP — Stroke Outcome Optimization Project

| | |
|---|---|
| **Access** | [OpenNeuro ds004889](https://openneuro.org/datasets/ds004889) |
| **License** | Public domain (no DUA required) |
| **Size** | 1,715 subjects (1,461 confirmed stroke, 1,106 with outcomes) |
| **Modalities** | T1w, FLAIR, DWI (TRACE + ADC) |
| **Tasks** | mRS binary (good vs poor functional outcome), NIHSS |
| **Train / Val / Test** | 388 / 129 / 130 (multimodal T1w+FLAIR) |
| **JEPA compatibility** | Good — T1w + FLAIR available. MNI registration required (scripts in repo). BIDS format. |
| **Notes** | Most diverse stroke dataset (~25% Black/African American). Includes GWTG-Stroke clinical data. |

#### ABIDE — Autism Brain Imaging Data Exchange

| | |
|---|---|
| **Access** | [NITRC fcon_1000](https://fcon_1000.projects.nitrc.org/indi/abide/) |
| **License** | CC BY-NC-SA (non-commercial research only) |
| **Size** | ABIDE I: 1,112 subjects (539 ASD, 573 controls); ABIDE II: ~1,000 additional |
| **Modalities** | T1w (MPRAGE), resting-state fMRI |
| **Tasks** | Autism diagnosis (binary) |
| **Train / Val / Test** | 659 / 220 / 220 |
| **JEPA compatibility** | Partial — T1w structural available. No T2w/FLAIR. 24+ sites with heterogeneous acquisition; MNI registration and harmonization needed. |
| **Notes** | Multi-site (24+ international). Preprocessed derivatives available via PCP. |

#### ADHD-200

| | |
|---|---|
| **Access** | [NITRC Neuro Bureau](http://neurobureau.projects.nitrc.org/ADHD200/Introduction.html) |
| **License** | BSD (non-commercial research) |
| **Size** | 973 subjects (585 TD, 362 ADHD, 26 unknown) from 8 sites |
| **Modalities** | T1w (MPRAGE), resting-state fMRI |
| **Tasks** | ADHD diagnosis (binary) |
| **Train / Val / Test** | 617 / 73 / 86 |
| **JEPA compatibility** | Partial — only T1w structural. Multi-site harmonization needed. Preprocessed connectomes available via PCP. |
| **Notes** | Ages 7–27. Phenotypic data includes IQ, medication status. Athena/NIAK/Burner preprocessing pipelines available. |

#### CNP — Consortium for Neuropsychiatric Phenomics

| | |
|---|---|
| **Access** | [OpenNeuro ds000030](https://openneuro.org/datasets/ds000030) |
| **License** | CC0 (Public Domain — no restrictions) |
| **Size** | ~272 subjects (130 healthy, 49 bipolar, 50 schizophrenia, 43 ADHD) |
| **Modalities** | T1w (MPRAGE), DWI (64-direction HARDI), task BOLD (5 tasks), resting fMRI |
| **Tasks** | Psychiatric diagnosis classification |
| **Train / Val / Test** | 132 / 66 / 67 |
| **JEPA compatibility** | Partial — T1w structural only. No T2w/FLAIR. BIDS format. |
| **Notes** | Preprocessed fMRIPrep derivatives available. Small sample but very well-characterized phenotypically. |

#### OpenBHB — Open Big Healthy Brains

| | |
|---|---|
| **Access** | [IEEE DataPort](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing) |
| **License** | CC BY-NC-SA 3.0 (also subject to most restrictive DUA of source cohorts) |
| **Size** | Train: 3,227; Val: 757; Test: 664 (private leaderboard) |
| **Modalities** | T1w (preprocessed: VBM CAT12, SBM FreeSurfer, Quasi-Raw) |
| **Tasks** | Brain age regression (chronological age prediction) |
| **Train / Val / Test** | 2,581 / 646 / 757 (public test) |
| **JEPA compatibility** | Requires re-preprocessing — OpenBHB distributes preprocessed derivatives (not raw images). Quasi-Raw format closest to JEPA input but still skull-stripped + linearly registered. |
| **Notes** | Aggregates 10 public cohorts (IXI, ABIDE, CoRR, GSP, etc.). Test set is private for challenge leaderboard. |

---

### Tier 2: DUA Required — Free for Academic Research

Requires a signed Data Use Agreement and/or institutional approval. Processing
time ranges from days to weeks.

#### OASIS3 — Open Access Series of Imaging Studies

| | |
|---|---|
| **Access** | [oasis-brains.org](https://www.oasis-brains.org/) via NITRC-IR |
| **License** | Custom DUA (non-commercial academic research only) |
| **Size** | 1,378 participants (755 CN, 622 AD-spectrum), 2,842 MR sessions |
| **Modalities** | T1w, T2w, FLAIR, ASL, SWI, TOF, resting BOLD, DTI; also PET (PIB, AV45, FDG) |
| **Tasks** | Alzheimer's Disease diagnosis (binary) |
| **Train / Val / Test** | 811-1,143 / 282-400 / 291-381 (varies by modality) |
| **JEPA compatibility** | Excellent — T1w, T2w, FLAIR all available. Longitudinal (up to 5 timepoints per subject). Gold standard for AD research. |
| **Notes** | Requires NITRC account + DUA acceptance. PET Avid tracers need 30-day pharma review before publication. |

#### ADNI — Alzheimer's Disease Neuroimaging Initiative

| | |
|---|---|
| **Access** | [LONI IDA](https://ida.loni.usc.edu/) |
| **License** | ADNI-specific DUA (requires institutional signature, annual reporting) |
| **Size** | ADNI-1: ~800 subjects; full ADNI (GO/2/3): 2,000+ |
| **Modalities** | T1w, T2w, FLAIR, DTI, resting fMRI, PET (FDG, amyloid, tau), ASL |
| **Tasks** | AD diagnosis (binary), Amyloid status, TTE (MCI→AD conversion time) |
| **Train / Val / Test** | ADNI1_AD: 1,135 / 253 / 244; ADNI Amyloid: 100-190 / 33-64 / 34-64; TTE: 125 / 42 / 42 |
| **JEPA compatibility** | Excellent — richest multimodal data of any public cohort. T1w, T2w, FLAIR all present. |
| **Notes** | Processing 1-2 weeks. Institutional signature required on DUA. Strict anti-redistribution policy. Publication policy requires annual reporting. |

#### PPMI — Parkinson's Progression Markers Initiative

| | |
|---|---|
| **Access** | [ppmi-info.org](https://www.ppmi-info.org/access-data-specimens/download-data) |
| **License** | PPMI DUA (annual reporting, data return obligation) |
| **Size** | 1,000+ subjects (PD, SWEDD, healthy controls, prodromal) |
| **Modalities** | T1w, T2w, FLAIR, DTI, resting fMRI, DaTSCAN SPECT |
| **Tasks** | PD diagnosis (binary), TTE (progression time) |
| **Train / Val / Test** | PD classification: 769-1,419 (varies by modality); TTE: 238-529 |
| **JEPA compatibility** | Excellent — T1w, T2w, FLAIR all present. Longitudinal design (multiple follow-ups). |
| **Notes** | Requires DUA + publications policy compliance. Must return derived data to PPMI. DaTSCAN imaging provides gold-standard dopaminergic confirmation. |

#### NACC — National Alzheimer's Coordinating Center

| | |
|---|---|
| **Access** | [naccdata.org](https://www.naccdata.org/data-request-process/) + [SCAN](https://scan.naccdata.org/) |
| **License** | NACC DUA (~3 business day approval) |
| **Size** | 12,000+ participants with structural MRI (via SCAN); 45,000+ with clinical data (UDS) |
| **Modalities** | T1w, T2w, FLAIR (via SCAN initiative); Amyloid + Tau PET |
| **Tasks** | AD diagnosis (binary), Amyloid status |
| **Train / Val / Test** | AD_T1w+T2w: 2,638 / 571 / 563 (multimodal) |
| **JEPA compatibility** | Excellent — T1w, T2w, FLAIR all available. Largest AD cohort. AWS S3 delivery. |
| **Notes** | MR images delivered via AWS S3 (Cyberduck/S3 Browser). SCAN provides defaced images + numerical summaries (FreeSurfer volumes). Multi-site (34 ADRCs). |

#### MCSA — Mayo Clinic Study of Aging

| | |
|---|---|
| **Access** | [GAAIN MCSA portal](https://www.gaaindata.org/partner/MCSA) or mcsaadrcdatasharing@mayo.edu |
| **License** | MCSA DUA (academic + industry researchers) |
| **Size** | ~5,925 clinical participants; ~1,802 with 3T MRI |
| **Modalities** | T1w (MPRAGE), FLAIR, DWI (41 dir, b=1000), PiB PET |
| **Tasks** | Cognitive impairment (Cog), Stroke, Hypertension, Dyslipidemia |
| **Train / Val / Test** | 1,733 / 543 / 597 |
| **JEPA compatibility** | Good — T1w + FLAIR available. DICOM format, defaced with mri_reface. |
| **Notes** | Population-based (not clinic-based). Ages 30-90. Rich cardiovascular/metabolic phenotyping. |

---

### Tier 3: Restricted Clinical Cohorts

Accessible with institutional agreements but not freely downloadable without
established research relationships.

#### BIND-MGH — Brain Imaging and Neurophysiology Database

| | |
|---|---|
| **Access** | [BDSP.io](https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/) |
| **License** | DUA + CITI human subjects certification + AWS account |
| **Size** | 38,945 patients, 1.8M brain scans (1.7M+ MRI across 1.5T/3T/7T) |
| **Modalities** | T1w, T2w, FLAIR, DWI, fMRI, SWI, PWI, MRA |
| **Tasks** | 17 brain pathology labels (multi-label): tumor types, edema, hematoma, infarct, MS, etc. |
| **Train / Val / Test** | 13,351 / 4,534 / 4,257 (multimodal, per modality) |
| **JEPA compatibility** | Excellent — richest single-cohort multimodal data. NIfTI/BIDS format. AWS-hosted. |
| **Notes** | Published Sep 2025. Sponsored by AWS Open Data program (free egress). Requires CITI training completion. Most recent and largest public clinical neuroimaging dataset. |

#### ICSPR — Ischemic Stroke (multimodal)

| | |
|---|---|
| **Access** | See notes below |
| **License** | Varies by sub-dataset |
| **Size** | T1w+FLAIR multimodal: 1,321 / 441 / 450 |
| **Modalities** | T1w, T2w, FLAIR, DWI |
| **Tasks** | 90-day mRS binary, Lesion type classification, Length of Stay > 8 days |
| **JEPA compatibility** | Partial — access details unclear for the specific multimodal splits used in Neuro-JEPA |
| **Notes** | The Neuro-JEPA README lists ICSPR as a "public" cohort but the exact dataset source is ambiguous. The multimodal splits in `datasets/multimodal/ICSPR/` reference T1w+FLAIR columns. The ATLAS stroke dataset (T1w only, INDI) is the most accessible public stroke MRI resource. |

---

### Tier 4: Internal Cohorts — No Public Access

Not downloadable. Results are reported for benchmarking purposes only.

| Cohort | Modalities | N (approximate) | Source |
|---|---|---|---|
| **NYU Langone** | T1w, T2w, FLAIR | 282,693 patients (pretraining), downstream subsets | NYU Langone Health clinical |
| **NYU Long Island** | T1w, T2w, FLAIR | Not disclosed | NYU Langone — Long Island hospitals |
| **MGH Clinical** | T1w, T2w, FLAIR | Subset of BIND-MGH used for 45 unimodal + 30 multimodal combinations | Massachusetts General Hospital |

---

## Dataset × Task × Modality Matrix

### Multimodal (Paired Modalities)

| Dataset | Modality Pair | Task | Train | Val | Test | Public? |
|---|---|---|---|---|---|---|
| UCSF-PDGM | T1w + FLAIR | IDH mutation | 246 | 124 | 125 | Yes (CC BY 4.0) |
| SOOP | T1w + FLAIR | mRS binary | 388 | 129 | 130 | Yes (Public Domain) |
| ICSPR | T1w + FLAIR | 90-day mRS / Lesion type / LoS | 1,321 | 441 | 450 | Partial |
| MCSA | T1w + FLAIR | Cognitive impairment | 1,733 | 543 | 597 | Yes (DUA) |
| NACC | T1w + T2w | AD diagnosis | 2,638 | 571 | 563 | Yes (DUA) |
| OASIS3 | T1w + T2w | AD diagnosis | 811 | 282 | 291 | Yes (DUA) |
| PPMI | T1w + T2w | PD diagnosis | 830 | 256 | 253 | Yes (DUA) |
| BIND-MGH | T1w+T2w+FLAIR | 17 brain pathologies | 13,351 | 4,534 | 4,257 | Yes (DUA+CITI) |

### Classification (Unimodal)

| Dataset | Modalities Used | Primary Task | Public? |
|---|---|---|---|
| ABIDE | T1w | Autism | Yes (CC BY-NC-SA) |
| ADHD-200 | T1w | ADHD | Yes (BSD) |
| ADNI-1 | T1w | AD | Yes (DUA) |
| ADNI Amyloid | T1w | Amyloid status | Yes (DUA) |
| BIND-MGH | T1w / T2w / FLAIR | 17 pathologies (per modality) | Yes (DUA+CITI) |
| CNP | T1w | Psychiatric Dx | Yes (CC0) |
| NACC | T1w / T2w / FLAIR | AD, Amyloid (per modality) | Yes (DUA) |
| OASIS3 | T1w / T2w / FLAIR | AD (per modality) | Yes (DUA) |
| PPMI | T1w / T2w / FLAIR | PD (per modality) | Yes (DUA) |
| SOOP | T1w / FLAIR | mRS (per modality) | Yes (Public Domain) |
| UCSF-PDGM | T1w / T2w / FLAIR / DWI | IDH mutation (per modality) | Yes (CC BY 4.0) |

### Time-to-Event

| Dataset | Modalities | Endpoint | Train | Val | Test | Public? |
|---|---|---|---|---|---|---|
| ADNI-1 | T1w | MCI→AD conversion (days) | 125 | 42 | 42 | Yes (DUA) |
| PPMI | T1w, FLAIR | PD progression (months) | 238-529 | 79-176 | 80-177 | Yes (DUA) |
| UCSF-PDGM | T1w, T2w, FLAIR | Overall survival (days) | 246 | 124 | 125 | Yes (CC BY 4.0) |

### Regression

| Dataset | Modalities | Target | Train | Val | Test | Public? |
|---|---|---|---|---|---|---|
| OpenBHB | T1w (derivatives) | Chronological age | 2,581 | 646 | 757 | Yes (CC BY-NC-SA 3.0) |

---

## Recommended Quick-Start Path

For reproducing Neuro-JEPA multimodal results with minimal administrative
overhead:

1. **UCSF-PDGM** (CC BY 4.0, no DUA) — IDH mutation classification + TTE
   survival. T1w, T2w, FLAIR, DWI all available. 501 subjects.

2. **SOOP** (Public Domain, no DUA) — mRS stroke outcome. T1w + FLAIR. 1,715
   subjects.

3. **CNP** (CC0, no DUA) — Psychiatric diagnosis classification. T1w only
   but good for unimodal baseline. 272 subjects.

4. **OASIS3** (DUA, ~1 week) — Best for AD classification. T1w + T2w + FLAIR.
   1,378 subjects with longitudinal follow-up. Rich clinical data.

5. **OpenBHB** (CC BY-NC-SA, no DUA) — Brain age regression. Largest public
   age-prediction benchmark. Note: preprocessed derivatives, needs
   re-preprocessing for JEPA.

All five together cover multimodal classification, unimodal classification,
survival analysis, and regression — spanning the full Neuro-JEPA evaluation
task taxonomy.

---

## Registration Templates

Neuro-JEPA uses two MNI templates for spatial normalization. Both are publicly
downloadable:

| Modality | Template | Download |
|---|---|---|
| T1w | MNI152 T1 1 mm brain (`MNI152_T1_1mm_brain.nii.gz`) | [Google Drive](https://drive.google.com/file/d/1H84c-gcge5FpNIpU7gJiFB9J8ayeF-mx/view) |
| T2w / FLAIR | MNI ICBM 152 T2 nonlinear asymmetric 09c (`mni_icbm152_t2_tal_nlin_asym_09c.nii`) | [Google Drive](https://drive.google.com/file/d/1PWjTucRtyEGs1X1yl6bjSMnSb7q3ya8w/view) |

Registration scripts are provided in `submodules/Neuro-JEPA/registration/`:
- `register_t1w.sh` — ANTs-based T1w → MNI152 T1 registration
- `register_t2w.sh` — ANTs-based T2w → MNI152 T2 registration
- `register_flair.sh` — ANTs-based FLAIR → MNI152 T2 registration (same template as T2w)

---

## License Summary

| Cohort | License | Commercial Use | Attribution | DUA Required |
|---|---|---|---|---|
| UCSF-PDGM | CC BY 4.0 | Yes | Yes | No |
| SOOP | Public Domain | Yes | Yes (cite paper) | No |
| CNP | CC0 | Yes | No (but cite) | No |
| ABIDE | CC BY-NC-SA | No | Yes | No (registration) |
| ADHD-200 | BSD | No (non-commercial) | Yes | No (registration) |
| OpenBHB | CC BY-NC-SA 3.0 | No | Yes | No (IEEE registration) |
| OASIS3 | Custom DUA | No | Yes + acknowledgment | Yes |
| ADNI | Custom DUA | No | Yes + annual report | Yes (institutional) |
| PPMI | Custom DUA | No | Yes + data return | Yes |
| NACC | Custom DUA | No | Yes | Yes (3-day) |
| MCSA | Custom DUA | No | Yes | Yes |
| BIND-MGH | Custom DUA | No | Yes | Yes + CITI |
| Neuro-JEPA Model Weights | CC BY-NC-ND 4.0 | No | Yes | Request via HF |

---

## Files Reference

| Path | Content |
|---|---|
| `submodules/Neuro-JEPA/datasets/unimodal/` | Patient-level splits for all unimodal tasks |
| `submodules/Neuro-JEPA/datasets/multimodal/` | Patient-level splits for all multimodal tasks |
| `submodules/Neuro-JEPA/registration/` | ANTs registration scripts for T1w, T2w, FLAIR |
| `submodules/Neuro-JEPA/configs/finetune/` | Finetune config templates (TTE, multimodal, unimodal) |
| `submodules/Neuro-JEPA/src/neurojepa/data/datasets.py` | Dataset classes (unimodal + SurvivalDataset) |
| `submodules/Neuro-JEPA/src/neurojepa/data/datasets_mm.py` | Multimodal dataset classes |
| `submodules/Neuro-JEPA/src/neurojepa/data/transforms.py` | `vit3d_transforms`, `loading_transforms` |
| `claude/JEPA_MULTIMODAL.md` | Multimodal architectures + benchmark results |
| `claude/BACKPROP_HOOK_INPLACE.md` | Two-phase hook DAG architecture |

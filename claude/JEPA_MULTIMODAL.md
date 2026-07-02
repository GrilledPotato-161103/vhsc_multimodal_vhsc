# Neuro-JEPA Multimodal Finetuning & Evaluation

## Overview

Neuro-JEPA is evaluated across both **unimodal** (single MRI contrast) and
**multimodal** (paired MRI contrasts) downstream tasks. The same shared ViT
backbone processes each modality independently; fusion happens either via a
late-fusion classifier, product-of-experts (PoE), or modality averaging.

The evaluation covers **classification** (diagnosis, mutation status, lesion
type), **regression** (brain age), and **time-to-event / survival analysis**.

---

## Evaluation Summary

| Category | Combinations | Metrics |
|---|---|---|
| Unimodal classification (public) | 41 dataset-task-modality | AUROC, AUPRC |
| Unimodal classification (NYU internal) | 30 combinations | AUROC, AUPRC |
| Unimodal classification (Long Island) | 30 combinations | AUROC, AUPRC |
| Unimodal classification (MGH) | 45 combinations | AUROC, AUPRC |
| Multimodal classification (public) | 12 combinations | AUROC, AUPRC |
| Multimodal classification (MGH) | 30 combinations | AUROC, AUPRC |
| Survival / TTE | 6 combinations | C-index |
| Brain age regression | 1 dataset (OpenBHB, n=757) | R², MAE, RMSE |

---

## Multimodal Fusion Architectures

Neuro-JEPA supports four fusion strategies, selected via `model.model_name`:

| `model_name` | Architecture | Description |
|---|---|---|
| `vit_late` | `MultiModalLateFusion` | Projection heads → bidirectional cross-attention → gated fusion → classifier |
| `vit_avg` | `nn.ModuleList` of `nn.Linear` | Independent linear heads per modality, logit-averaged at output |
| `vit_poe` | Per-modality `AttentiveClassifier` + PoE | Each modality outputs log-probabilities, multiplied element-wise (Product of Experts) |
| `vit_poe_joint` | PoE + joint `AttentiveClassifier` | PoE plus a joint head that attends to concatenated token sequences |
| `vit_mil` | `ClassifyThenAggregate` | Multi-instance learning: per-token classify, then aggregate across tokens |
| `vit_early` | `AttentiveClassifier` on concatenated tokens | Modality token sequences concatenated along sequence dim before cross-attention |

### Late Fusion (`vit_late`) — Primary Multimodal Method

```
                                     ┌── proj1 (ProjectionHead) ──┐
  f1 [B, N, 768] ───────────────────┤                              ├── cross_attn_1to2 ──┐
                                     └─────────────────────────────┘                      │
                                                                                          ├── mean pool ── gate ── classifier
                                     ┌── proj2 (ProjectionHead) ──┐                      │
  f2 [B, N, 768] ───────────────────┤                              ├── cross_attn_2to1 ──┘
                                     └─────────────────────────────┘

  gate = tanh(ReLU(Linear(cat(pool1, pool2))))
  fused = gate * pool1 + (1 - gate) * pool2
  return Linear(fused)  →  [B, num_classes]
```

From [submodules/Neuro-JEPA/src/neurojepa/models/cross_attn.py](../submodules/Neuro-JEPA/src/neurojepa/models/cross_attn.py)

### Product of Experts (`vit_poe` / `vit_poe_joint`)

```
  logits_1 = AttentiveClassifier_1(f1)   →  log_softmax →  [B, C]
  logits_2 = AttentiveClassifier_2(f2)   →  log_softmax →  [B, C]
  fused = logits_1 + logits_2            (sum of log-probabilities = PoE)
  pred = argmax(fused)

  vit_poe_joint adds:
    joint_logits = JointAttentiveClassifier(cat(f1, f2, dim=1))  →  [B, C]
    fused = logits_1 + logits_2 + log_softmax(joint_logits)
```

From [submodules/Neuro-JEPA/src/neurojepa/engines/finetune/clf_mm_poe.py](../submodules/Neuro-JEPA/src/neurojepa/engines/finetune/clf_mm_poe.py)

### Support for Missing Modalities

The multimodal dataset (`BaseMRIDataset`) natively handles missing modalities:
- Each modality has its own `PersistentDataset` cache
- `_col_indices_map` tracks which samples have valid images per modality
- Missing modalities get a **zero placeholder tensor** and are flagged via
  `__validity_mask__`
- **Random modality dropout** (`drop_prob=0.2`): during training, up to N-1
  modalities may be randomly dropped per sample, forcing the model to be
  robust to missing inputs

---

## Datasets — Multimodal (Paired Modalities)

All splits are by **patient ID** to prevent data leakage.

### Public Cohorts

#### BIND-MGH (Boston Image Neurology Database)
| Property | Value |
|---|---|
| Modalities | T1w, T2w, FLAIR (registration only; multimodal uses selected pairs) |
| Train / Val / Test | 13,351 / 4,534 / 4,257 |
| Tasks | 17 brain pathology labels (multi-label): Astrocytoma, Atrophy, Cyst, Edema, Enhancement, Hematoma, Infarct, Ischemic, Mass effect, Midline shift, Multiple Sclerosis, Schwannoma, Cancer, Glioblastoma multiforme, Gliosis |
| Source | Massachusetts General Hospital clinical cohort |

#### ICSPR (Ischemic Stroke)
| Property | Value |
|---|---|
| Modalities | T1w, T2w, FLAIR, DWI |
| Train / Val / Test | 1,321 / 441 / 450 (multimodal T1w+FLAIR) |
| Tasks | 90-day mRS binary (good vs poor outcome), Lesion type classification, Length of Stay > 8 days |
| Source | Public ischemic stroke registry |

#### MCSA (Mayo Clinic Study of Aging)
| Property | Value |
|---|---|
| Modalities | T1w, FLAIR |
| Train / Val / Test | 1,733 / 543 / 597 |
| Tasks | Cognitive impairment (Cog), Stroke, Hypertension, Dyslipidemia |
| Source | Mayo Clinic population-based study |

#### NACC (National Alzheimer's Coordinating Center)
| Property | Value |
|---|---|
| Modalities | T1w, T2w, FLAIR |
| Train / Val / Test (T1w+T2w) | 2,638 / 571 / 563 |
| Tasks | Alzheimer's Disease diagnosis (AD binary), Amyloid status |
| Source | NACC uniform dataset (multi-site US) |

#### OASIS3 (Open Access Series of Imaging Studies)
| Property | Value |
|---|---|
| Modalities | T1w, T2w, FLAIR |
| Train / Val / Test (T1w+T2w) | 811 / 282 / 291 |
| Tasks | Alzheimer's Disease diagnosis (AD binary) |
| Source | Knight ADRC, Washington University |

#### PPMI (Parkinson's Progression Markers Initiative)
| Property | Value |
|---|---|
| Modalities | T1w, T2w, FLAIR |
| Train / Val / Test (T1w+T2w) | 830 / 256 / 253 |
| Tasks | Parkinson's Disease diagnosis |
| Source | PPMI multi-site study |

#### SOOP (Stroke Outcome)
| Property | Value |
|---|---|
| Modalities | T1w, FLAIR |
| Train / Val / Test | 388 / 129 / 130 |
| Tasks | Modified Rankin Scale binary (gs_rankin_binary: good vs poor functional outcome) |
| Source | Public stroke cohort |

#### UCSF-PDGM (Preoperative Diffuse Glioma MRI)
| Property | Value |
|---|---|
| Modalities | T1w, T2w, FLAIR, DWI |
| Train / Val / Test (T1w+FLAIR) | 246 / 124 / 125 |
| Tasks | IDH mutation status (binary), Overall survival (TTE) |
| Source | UCSF public glioma cohort |

---

## Datasets — Unimodal (Single Modality)

Used for both independent unimodal benchmarks and as inputs to multimodal
fusion (each modality encoded by the same backbone independently).

### Classification Datasets

| Dataset | Modalities Available | Train | Val | Test | Tasks |
|---|---|---|---|---|---|
| **ABIDE** | T1w | 659 | 220 | 220 | Autism diagnosis |
| **ADHD-200** | T1w | 617 | 73 | 86 | ADHD diagnosis |
| **ADNI-1** | T1w, T2w, FLAIR | 1,135 | 253 | 244 | AD diagnosis |
| **ADNI Amyloid** | T1w, T2w, FLAIR | 100-190 | 33-64 | 34-64 | Amyloid PET status |
| **BIND-MGH** | T1w, T2w, FLAIR | 13,351 | 4,534 | 4,257 | 17 brain pathology labels |
| **CNP** | T1w | 132 | 66 | 67 | Psychiatric diagnosis |
| **ICSPR** | T1w, T2w, FLAIR, DWI | 1,364-1,581 | 456-523 | 461-539 | Stroke outcomes |
| **MCSA** | T1w, FLAIR | 1,733 | 543 | 597 | Cognition, stroke, vascular |
| **NACC** | T1w, T2w, FLAIR | 1,846-3,006 | 613-1,008 | 603-980 | AD, amyloid |
| **OASIS3** | T1w, T2w, FLAIR | 620-1,143 | 204-400 | 204-381 | AD diagnosis |
| **PPMI** | T1w, T2w, FLAIR | 769-1,419 | 271-478 | 256-468 | PD diagnosis |
| **SOOP** | T1w, FLAIR | 388 | 129 | 130 | mRS binary |
| **UCSF-PDGM** | T1w, T2w, FLAIR, DWI | 246 | 124 | 125 | IDH mutation |

### Regression Datasets

| Dataset | Modalities | Train | Val | Test | Target |
|---|---|---|---|---|---|
| **OpenBHB** | T1w | 2,581 | 646 | 757 | Chronological age (brain age gap) |

### Time-to-Event Datasets

| Dataset | Modalities | Train | Val | Test | Endpoint |
|---|---|---|---|---|---|
| **ADNI-1** | T1w | 125 | 42 | 42 | Time to AD conversion (MCI→AD) |
| **PPMI** | T1w, FLAIR | 238-529 | 79-176 | 80-177 | Time to PD progression |
| **UCSF-PDGM** | T1w, T2w, FLAIR | 246 | 124 | 125 | Overall survival |

---

## NYU / Long Island Internal Cohorts

Listed in the results table but splits are not provided in the public
repository (clinical data privacy constraints). These are large-scale
internal datasets from NYU Langone Health and Long Island hospitals.

| Cohort | Modalities | Combinations | Source |
|---|---|---|---|
| **NYU** | T1w, T2w, FLAIR | 30 modality-task pairs | NYU Langone clinical |
| **Long Island** | T1w, T2w, FLAIR | 30 modality-task pairs | NYU Langone - Long Island |

---

## Training Configuration (Multimodal Finetuning)

| Parameter | Value |
|---|---|
| Epochs | 15 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Peak LR | 1.5e-4 |
| LR schedule | Warmup 2 epochs, linear decay to 1.5e-8 |
| Weight decay | 0.01 → 0.01 (constant) |
| Mixed precision | bfloat16 |
| Batch size | 16 (multimodal), 64 (unimodal) |
| Samples per epoch | 5,000 (oversampling for small datasets) |
| Modality dropout | 0.2 probability of dropping each extra modality |
| MoE | 16 routed experts, 6 activated, layers [1,3,5,7,9,11] |

### Metrics

| Task Type | Primary Metric | Secondary Metrics |
|---|---|---|
| Binary classification | AUROC (macro + per-class) | AUPRC, Accuracy |
| Multi-class classification | AUROC (macro + per-class) | AUPRC, Accuracy |
| Multi-label classification | AUROC per label | AUPRC per label |
| Regression (brain age) | R² | MAE, RMSE (years) |
| Survival (TTE) | C-index | Cox PH loss |

---

## Benchmark Results

### Multimodal Classification (Public Cohorts, 12 combinations)

| Model | AUROC | AUPRC |
|---|---|---|
| VoCo-B | 0.743 [0.698, 0.789] | 0.562 [0.443, 0.673] |
| BrainIAC | 0.730 [0.693, 0.766] | 0.552 [0.428, 0.673] |
| NeuroVFM | 0.748 [0.684, 0.804] | 0.574 [0.449, 0.685] |
| **Neuro-JEPA** | **0.805 [0.759, 0.849]** | **0.637 [0.505, 0.749]** |

### Multimodal Classification (MGH Cohort, 30 combinations)

| Model | AUROC | AUPRC |
|---|---|---|
| VoCo-B | 0.729 [0.701, 0.757] | 0.241 [0.194, 0.288] |
| BrainIAC | 0.684 [0.662, 0.707] | 0.203 [0.160, 0.245] |
| NeuroVFM | 0.742 [0.721, 0.765] | 0.255 [0.205, 0.305] |
| **Neuro-JEPA** | **0.763 [0.739, 0.789]** | **0.295 [0.248, 0.343]** |

---

## Files Reference

| Path | Role |
|---|---|
| `scripts/finetune/mm.py` | Multimodal late-fusion finetuning entry point |
| `scripts/finetune/mm_poe.py` | Multimodal PoE fusion finetuning entry point |
| `scripts/finetune/tte.py` | Time-to-event finetuning entry point |
| `scripts/finetune/default.py` | Unimodal finetuning entry point |
| `src/neurojepa/engines/finetune/clf_mm.py` | Multimodal classification training loop |
| `src/neurojepa/engines/finetune/clf_mm_poe.py` | PoE multimodal training loop |
| `src/neurojepa/engines/finetune/tte.py` | Survival analysis training loop |
| `src/neurojepa/models/cross_attn.py` | `MultiModalLateFusion` (late fusion classifier) |
| `src/neurojepa/models/mm_classifier.py` | Simple additive fusion classifier (used in some configs) |
| `src/neurojepa/models/attentive_pooler.py` | `AttentiveClassifier` (cross-attention pooling head) |
| `src/neurojepa/models/mil.py` | `ClassifyThenAggregate` (MIL classifier) |
| `src/neurojepa/data/datasets_mm.py` | Multimodal dataset classes (`BaseMRIDataset`, `FinetuneDataset`) |
| `src/neurojepa/data/datasets.py` | Unimodal + `SurvivalDataset` |
| `src/neurojepa/loss/cox_loss.py` | Cox PH loss + C-index metrics |
| `configs/finetune/finetune_neurojepa_mm.yaml` | Multimodal finetune config template |
| `configs/finetune/finetune_neurojepa_tte.yaml` | TTE finetune config template |
| `datasets/multimodal/` | 8 dataset split CSVs (BIND-MGH, ICSPR, MCSA, NACC, OASIS3, PPMI, SOOP, UCSF-PDGM) |
| `datasets/unimodal/` | 12+ dataset split CSVs (all modalities) |

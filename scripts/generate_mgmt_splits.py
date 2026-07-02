#!/usr/bin/env python3
"""Generate unified ``labels.csv`` for two disjoint MGMT classification cohorts.

Reads raw metadata CSVs for RSNA-MICCAI BraTS Radiogenomic (in-domain) and
UCSF-PDGM-v3 (out-of-domain), harmonises MGMT labels to {0, 1}, detects
subject overlap via BraTS21 ID cross-referencing, **subtracts** overlapped
subjects from UCSF-PDGM, and writes one ``labels.csv`` per dataset.

Chronological order: BraTS first, then UCSF-PDGM references BraTS output.
Both CSVs share identical 6-column schema::

    subject_id,mgmt_label,split,t1_path,t2_path,flair_path

Usage::

    python scripts/generate_mgmt_splits.py              # strict (subtract overlap)
    python scripts/generate_mgmt_splits.py --include-overlap  # keep all 416 for testing
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

BRATS_RAW_LABELS = ROOT / "data" / "jepa" / "brats_radiogenomic" / "train_labels.csv"
BRATS_SPLITS_DIR = ROOT / "data" / "jepa" / "brats_radiogenomic" / "splits"
BRATS_OUT = BRATS_SPLITS_DIR / "labels.csv"

UCSF_META = ROOT / "data" / "jepa" / "ucsf_pdgm" / "UCSF-PDGM-metadata.csv"
UCSF_SPLITS_DIR = ROOT / "data" / "jepa" / "ucsf_pdgm" / "splits"
UCSF_OUT = UCSF_SPLITS_DIR / "labels.csv"

LABEL_COLUMNS = ["subject_id", "mgmt_label", "split", "t1_path", "t2_path", "flair_path"]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stratified_split(
    items: List[dict],
    label_key: str = "mgmt_label",
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = RANDOM_SEED,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Stratified three-way split without sklearn.

    Groups items by *label_key*, shuffles each group with a fixed seed,
    then slices by *ratios*.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[dict]] = {}
    for item in items:
        label = int(item[label_key])
        by_label.setdefault(label, []).append(item)

    train, val, test = [], [], []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_test = max(1, round(n * ratios[2]))
        n_val = max(1, round(n * ratios[1]))
        n_train = n - n_val - n_test
        train.extend(label_samples[:n_train])
        val.extend(label_samples[n_train:n_train + n_val])
        test.extend(label_samples[n_train + n_val:])
    return train, val, test


def parse_ucsf_id(raw_id: str) -> str:
    """Normalize ``UCSF-PDGM-004`` → ``UCSF-PDGM-0004`` (4-digit zero-padded)."""
    numeric = "".join(ch for ch in raw_id if ch.isdigit())
    if not numeric:
        raise ValueError(f"Cannot parse numeric ID from {raw_id!r}")
    return f"UCSF-PDGM-{int(numeric):04d}"


def strip_brats_prefix(raw: str) -> str:
    """Strip ``BraTS2021_`` prefix, return zero-padded 5-digit string."""
    s = raw.strip().replace("BraTS2021_", "")
    return s.zfill(5) if s else ""


def write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    """Write *rows* (list of dicts) to *path* with *columns* header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})
    print(f"  Wrote {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# Phase A — BraTS Radiogenomic (in-domain, intact)
# ---------------------------------------------------------------------------

def process_brats() -> Tuple[List[dict], set]:
    """Harmonise BraTS labels, stratified-split, write labels.csv.

    Returns
    -------
    rows : List[dict]
        All 585 rows (for downstream UCSF overlap detection).
    brats_ids : set
        Set of BraTS21ID strings (5-digit zero-padded).
    """
    print("=" * 60)
    print("Phase A — RSNA-MICCAI BraTS Radiogenomic (in-domain)")
    print("=" * 60)

    # --- Read raw labels ---
    rows: List[dict] = []
    with open(BRATS_RAW_LABELS, newline="") as f:
        for entry in csv.DictReader(f):
            sid = entry["BraTS21ID"].strip()
            mgmt = int(entry["MGMT_value"].strip())
            rows.append({
                "subject_id": sid,          # already 5-digit zero-padded
                "mgmt_label": mgmt,
                "t1_path": f"nifti/{sid}_t1.nii.gz",
                "t2_path": f"nifti/{sid}_t2.nii.gz",
                "flair_path": f"nifti/{sid}_flair.nii.gz",
            })
    print(f"  Raw subjects: {len(rows)}")

    # --- Stratified split ---
    train, val, test = stratified_split(rows, "mgmt_label")
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"

    all_rows = train + val + test
    assert len(all_rows) == len(rows), "Split lost subjects"

    write_csv(BRATS_OUT, all_rows, LABEL_COLUMNS)
    print(f"  Splits: train={len(train)}, val={len(val)}, test={len(test)}")
    label_counts = {"train": {}, "val": {}, "test": {}}
    for r in all_rows:
        label_counts[r["split"]][r["mgmt_label"]] = label_counts[r["split"]].get(r["mgmt_label"], 0) + 1
    for s in ("train", "val", "test"):
        print(f"    {s}: {label_counts[s]}")

    brats_ids: set = {r["subject_id"] for r in all_rows}
    return all_rows, brats_ids


# ---------------------------------------------------------------------------
# Phase B — UCSF-PDGM (out-of-domain, overlap subtracted)
# ---------------------------------------------------------------------------

def process_ucsf(brats_ids: set, include_overlap: bool = False) -> None:
    """Harmonise UCSF-PDGM labels, subtract BraTS overlap, write labels.csv.

    Parameters
    ----------
    brats_ids : set
        BraTS21ID set from the BraTS cohort (5-digit zero-padded strings).
    include_overlap : bool
        If True, keep all 416 usable subjects (for testing with sample data).
        Default False — subtracts overlapped subjects.
    """
    print()
    print("=" * 60)
    print("Phase B — UCSF-PDGM-v3 (out-of-domain, disjoint)")
    print("=" * 60)

    # --- Read & filter ---
    usable: List[dict] = []
    excluded_empty = 0
    excluded_indet = 0
    with open(UCSF_META, newline="") as f:
        for entry in csv.DictReader(f):
            mgmt_raw = entry.get("MGMT status", "").strip().lower()
            if mgmt_raw in ("", ""):
                excluded_empty += 1
                continue
            if mgmt_raw == "indeterminate":
                excluded_indet += 1
                continue
            # Harmonise label
            mgmt_label = 1 if mgmt_raw == "positive" else 0

            raw_id = entry["ID"].strip()
            norm_id = parse_ucsf_id(raw_id)

            brats_ref = strip_brats_prefix(entry.get("BraTS21 ID", ""))

            usable.append({
                "subject_id": norm_id,
                "mgmt_label": mgmt_label,
                "t1_path": f"{norm_id}_T1gad_bias.nii",
                "t2_path": "",                     # always missing
                "flair_path": f"{norm_id}_FLAIR_bias.nii",
                "_brats21_id": brats_ref,          # internal — for overlap check
            })

    print(f"  Raw subjects:    {len(usable) + excluded_empty + excluded_indet}")
    print(f"  Excluded empty:  {excluded_empty}")
    print(f"  Excluded indet:  {excluded_indet}")
    print(f"  Usable (tot):    {len(usable)}")
    pos = sum(1 for r in usable if r["mgmt_label"] == 1)
    neg = len(usable) - pos
    print(f"  Labels: pos={pos}, neg={neg}")

    # --- Overlap detection ---
    has_brats_id = [r for r in usable if r["_brats21_id"]]
    overlapped = [r for r in has_brats_id if r["_brats21_id"] in brats_ids]
    non_overlapped_in_brats = [r for r in has_brats_id if r["_brats21_id"] not in brats_ids]
    no_brats_id = [r for r in usable if not r["_brats21_id"]]

    print(f"\n  Has BraTS21 ID:     {len(has_brats_id)}")
    print(f"    -> In BraTS set:   {len(overlapped)}  (will subtract)")
    print(f"    -> Not in BraTS:   {len(non_overlapped_in_brats)}")
    print(f"  No BraTS21 ID:      {len(no_brats_id)}")

    if include_overlap:
        keep = usable
        print(f"\n  --include-overlap: keeping all {len(keep)} subjects")
    else:
        keep = no_brats_id + non_overlapped_in_brats
        print(f"\n  After subtraction: {len(keep)} subjects (test_ood)")

    # --- Assign split & write ---
    for r in keep:
        r["split"] = "test_ood"

    # Remove internal key before writing
    output = [{k: v for k, v in r.items() if k != "_brats21_id"} for r in keep]

    write_csv(UCSF_OUT, output, LABEL_COLUMNS)

    # --- Summary ---
    pos_kept = sum(1 for r in keep if r["mgmt_label"] == 1)
    neg_kept = len(keep) - pos_kept
    print(f"  test_ood labels: pos={pos_kept}, neg={neg_kept}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unified labels.csv for MGMT classification cohorts."
    )
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        help="Keep all 416 usable UCSF-PDGM subjects regardless of BraTS overlap "
             "(useful for testing with the 4 downloaded sample subjects).",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # Phase A — BraTS first (reference cohort)
    brats_rows, brats_ids = process_brats()

    # Phase B — UCSF-PDGM references BraTS output
    process_ucsf(brats_ids, include_overlap=args.include_overlap)

    print()
    print("Done.  Verification:")
    print(f"  BraTS labels:     {BRATS_OUT}  ({len(brats_rows)} rows)")
    ucsf_count = sum(1 for _ in open(UCSF_OUT)) - 1 if UCSF_OUT.exists() else 0
    print(f"  UCSF-PDGM labels: {UCSF_OUT}  ({ucsf_count} rows)")
    print()
    print("Run validation:")
    print("  python -c \"import csv; rows=list(csv.DictReader(open('data/jepa/brats_radiogenomic/splits/labels.csv'))); print(len(rows))\"")
    print("  python -c \"import csv; rows=list(csv.DictReader(open('data/jepa/ucsf_pdgm/splits/labels.csv'))); print(len(rows))\"")


if __name__ == "__main__":
    main()

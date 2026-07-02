"""Download 5-10 OpenBHB quasiraw samples from HuggingFace Hub for pipeline testing.

Usage:
    python scripts/download_openbhb_samples.py                    # download 5 train + 5 val
    python scripts/download_openbhb_samples.py --n_train 10 --n_val 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "benoit-dufumier/openBHB"
REPO_TYPE = "dataset"
OUT_DIR = _ROOT / "data" / "openbhb"


def main():
    parser = argparse.ArgumentParser(description="Download OpenBHB quasiraw samples")
    parser.add_argument("--n_train", type=int, default=5, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=5, help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Download metadata
    print("Downloading participants.tsv...")
    part_path = hf_hub_download(REPO_ID, "participants.tsv", repo_type=REPO_TYPE)
    df = pd.read_csv(part_path, sep="\t")

    # Get participant IDs per split
    train_ids = df[df.split == "train"]["participant_id"].astype(str).tolist()
    ext_ids = df[df.split == "external_test"]["participant_id"].astype(str).tolist()
    int_ids = df[df.split == "internal_test"]["participant_id"].astype(str).tolist()
    print(f"Available: {len(train_ids)} train, {len(ext_ids)} external_test, {len(int_ids)} internal_test")

    # Save full metadata locally
    out_meta = OUT_DIR / "participants.tsv"
    df.to_csv(out_meta, sep="\t", index=False)
    print(f"Saved metadata to {out_meta}")

    # List all quasiraw files on HF
    print("Listing HF repo files...")
    all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    quasiraw_files = [f for f in all_files if "quasiraw" in f and f.endswith(".npy")]

    # Map participant_id -> HF path
    id_to_path = {}
    for f in quasiraw_files:
        parts = f.split("/")
        for p in parts:
            if p.startswith("sub-"):
                pid = p.replace("sub-", "")
                id_to_path[pid] = f
                break

    # Also save full metadata
    df.to_csv(OUT_DIR / "participants.tsv", sep="\t", index=False)
    print(f"Saved participants.tsv to {OUT_DIR / 'participants.tsv'}")

    # Select and download
    rng = np.random.default_rng(args.seed)

    for split_name, pool_ids, out_subdir, n_samples in [
        ("train", train_ids, "train", args.n_train),
        ("val", ext_ids + int_ids, "val", args.n_val),
    ]:
        selected = rng.choice(pool_ids, size=min(n_samples, len(pool_ids)), replace=False)
        out_dir = OUT_DIR / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading {len(selected)} {split_name} samples...")
        metadata_rows = []
        for pid in sorted(selected):
            pid_str = str(int(pid))
            if pid_str not in id_to_path:
                print(f"  WARNING: {pid_str} not found in HF repo (no quasiraw file), skipping")
                continue
            hf_path = id_to_path[pid_str]
            local_path = hf_hub_download(REPO_ID, hf_path, repo_type=REPO_TYPE)
            # Copy to local dir
            local_name = f"sub-{pid_str}_preproc-quasiraw_T1w.npy"
            dest = out_dir / local_name
            import shutil
            shutil.copy2(local_path, dest)
            row = df[df.participant_id == int(pid_str)].iloc[0]
            metadata_rows.append({"participant_id": pid_str, "file": str(dest), "age": row["age"], "sex": row["sex"], "site": row["site"], "split": row["split"]})
            print(f"  {split_name}/{local_name}  age={row['age']:.1f}  sex={row['sex']}  site={int(row['site'])}")

        # Save split metadata
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_csv(OUT_DIR / f"{out_subdir}_metadata.csv", index=False)

    # Save download manifest
    manifest = {
        "repo_id": REPO_ID,
        "preprocess": "quasiraw",
        "description": "Quasi-raw T1w brain MRIs in MNI152 space, skull-stripped, saved as .npy arrays (1, 1, 182, 218, 182) float64",
        "n_train": min(args.n_train, len(train_ids)),
        "n_val": min(args.n_val, len(ext_ids) + len(int_ids)),
        "image_shape": [1, 1, 182, 218, 182],
        "target": "age (years)",
    }
    import json
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Data saved to {OUT_DIR}")
    print(f"  train/: {list(OUT_DIR.glob('train/*.npy'))}")
    print(f"  val/:   {list(OUT_DIR.glob('val/*.npy'))}")


if __name__ == "__main__":
    main()

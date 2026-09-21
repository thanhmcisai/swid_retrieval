# -*- coding: utf-8 -*-
"""Audit/build SmartWoodID manifest against the validated v3 split counts.

This is a lightweight guard for Colab/Drive recovery. It builds the manifest from
the current patch tree and split CSV, then checks that the split image/species
counts match the validated v3 run before downstream training/cache rebuilds use
it.
"""

import json
import os
from pathlib import Path

import pandas as pd

from . import config as D
from .manifest import build_swi_manifest


EXPECTED_V3 = {
    "meta-train": {"images": 124577, "species": 557},
    "meta-val": {"images": 18018, "species": 80},
    "meta-test": {"images": 33528, "species": 317},
}


def _candidate_patch_roots():
    env = os.environ.get("FINAL_DATASET_DIR")
    roots = []
    if env:
        roots.append(Path(env))
    roots.extend([
        D.FINAL_DATASET_DIR,
        D.C.PSI_DIR / "smartwoodid_kmeans_split",
        D.C.PSI_DIR,
        D.C.DATASETS_DIR / "PSI_smartwoodid",
    ])
    out = []
    seen = set()
    for r in roots:
        r = Path(r)
        if r in seen:
            continue
        seen.add(r)
        if (r / "train").exists() and (r / "test").exists():
            out.append(r)
    return out


def _candidate_split_csvs():
    env = os.environ.get("SPLIT_CSV")
    paths = []
    if env:
        paths.append(Path(env))
    paths.extend([
        D.SPLIT_CSV,
        D.C.ROOT_PATH / "smartwoodid_split.csv",
        D.C.PSI_DIR / "smartwoodid_split.csv",
    ])
    out = []
    seen = set()
    for p in paths:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            out.append(p)
    return out


def _summarize(manifest):
    rows = []
    for split in ["meta-train", "meta-val", "meta-test"]:
        items = manifest.get(split, [])
        rows.append({
            "split": split,
            "images": len(items),
            "species": len({sp for _, sp in items}),
        })
    return rows


def _check_split_csv(split_csv):
    df = pd.read_csv(split_csv)
    required = {"canonical_binomial", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{split_csv} missing columns: {missing}")
    counts = df.groupby("split")["canonical_binomial"].nunique().to_dict()
    expected_species = {k: v["species"] for k, v in EXPECTED_V3.items()}
    return counts, expected_species


def run(force=True, strict=True):
    patch_roots = _candidate_patch_roots()
    split_csvs = _candidate_split_csvs()
    if not patch_roots:
        raise FileNotFoundError(
            "No SmartWoodID patch root found. Expected a directory with train/ and test/; "
            "set FINAL_DATASET_DIR explicitly."
        )
    if not split_csvs:
        raise FileNotFoundError(
            "No smartwoodid_split.csv found; set SPLIT_CSV explicitly or upload it to ROOT_PATH/PSI_DIR."
        )
    patch_root = patch_roots[0]
    split_csv = split_csvs[0]
    print(f"Patch root: {patch_root}")
    print(f"Split CSV:  {split_csv}")
    split_species, expected_species = _check_split_csv(split_csv)
    print(f"Split CSV species counts: {split_species}")
    print(f"Expected v3 species counts: {expected_species}")

    manifest = build_swi_manifest(
        patch_dir=patch_root,
        split_csv=split_csv,
        manifest_path=D.MANIFEST_PATH,
        force=force,
    )
    rows = _summarize(manifest)
    print("Manifest counts:")
    ok = True
    for row in rows:
        exp = EXPECTED_V3[row["split"]]
        row_ok = row["images"] == exp["images"] and row["species"] == exp["species"]
        ok = ok and row_ok
        print(
            f"  {row['split']}: {row['images']} images/{row['species']} species "
            f"(expected {exp['images']}/{exp['species']}) {'OK' if row_ok else 'MISMATCH'}"
        )
    audit = {
        "patch_root": str(patch_root),
        "split_csv": str(split_csv),
        "manifest_path": str(D.MANIFEST_PATH),
        "counts": rows,
        "expected_v3": EXPECTED_V3,
        "status": "ok" if ok else "mismatch",
    }
    out_path = D.C.ROOT_PATH / "swi_manifest_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    print(f"Audit saved: {out_path}")
    if strict and not ok:
        raise RuntimeError("Manifest does not match validated v3 counts; do not train/rebuild cache yet.")
    return audit


if __name__ == "__main__":
    run()

# -*- coding: utf-8 -*-
"""Build swi_manifest.json — the bridge the eval/train package consumes.

Lifted from scratchpad.py:build_swi_manifest (L264-299). Scans the final patch
tree ONCE (FINAL_DATASET_DIR/{train,test}/<species>/scale_*/*.{jpg,png,jpeg}),
maps each species dir to its split via SPLIT_CSV (`canonical_binomial` → split,
spaces→underscores), and writes:
  {"meta-train": [[abs_path, species_dir], ...], "meta-val": ..., "meta-test": ...}
to config.MANIFEST_PATH (== what data.load_swi_manifest reads).

train/ dir holds meta-train ∪ meta-val species; test/ holds meta-test.
"""

import json

import pandas as pd

from . import config as D


def build_swi_manifest(patch_dir=None, split_csv=None, manifest_path=None, force=False):
    patch_dir = patch_dir or D.FINAL_DATASET_DIR
    split_csv = split_csv or D.SPLIT_CSV
    manifest_path = manifest_path or D.MANIFEST_PATH
    if manifest_path.exists() and not (force or D.FORCE_DATAPREP):
        print(f"✅ Manifest exists: {manifest_path}")
        return json.load(open(manifest_path))

    split_df = pd.read_csv(split_csv)
    sp_to_split = {row["canonical_binomial"].replace(" ", "_"): row["split"]
                   for _, row in split_df.iterrows()}

    manifest = {"meta-train": [], "meta-val": [], "meta-test": []}
    for subdir_name, subdir_splits in [("train", {"meta-train", "meta-val"}),
                                       ("test", {"meta-test"})]:
        scan_dir = patch_dir / subdir_name
        if not scan_dir.exists():
            print(f"  ⚠️ missing {scan_dir}")
            continue
        for sp_dir in sorted(scan_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            sp_split = sp_to_split.get(sp_dir.name)
            if sp_split not in subdir_splits:
                continue
            for scale_dir in sorted(sp_dir.iterdir()):
                if not scale_dir.is_dir() or not scale_dir.name.startswith("scale_"):
                    continue
                for ext in ("*.jpg", "*.png", "*.jpeg"):
                    for p in scale_dir.glob(ext):
                        manifest[sp_split].append([str(p), sp_dir.name])

    for split, items in manifest.items():
        print(f"  {split}: {len(items)} images, "
              f"{len({it[1] for it in items})} species")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"✅ Manifest saved: {manifest_path}")
    return manifest


def run():
    build_swi_manifest()


if __name__ == "__main__":
    run()

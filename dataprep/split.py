# -*- coding: utf-8 -*-
"""Species-level meta-train/val/test split (genus-stratified).

Lifted from old_version_claim_and_process_data/split_smartwoodid.py.
954 species → meta-train 60 / meta-val 10 / meta-test 30 (SEED=42):
  - ID species (overlap with public datasets, from ID_species_public.csv) are
    force-assigned to meta-test.
  - genera with ≥3 species: proportional within-genus; <3 species: whole-genus.
`canonical_binomial` is the FULL lowercase Species name (spaces kept) — the
manifest maps it to dir names via replace(" ","_"); data.ce_train_image_paths
replays the downstream image split over it, so this format must not change.
Output: SPLIT_CSV (+ summary).
"""

import random
from collections import defaultdict

import pandas as pd

from . import config as D


def species_to_canonical(name):
    if not name or pd.isna(name):
        return ""
    return str(name).strip().lower()


def load_id_species():
    if not D.ID_SPECIES_CSV.exists():
        print(f"⚠️ {D.ID_SPECIES_CSV} not found — no ID-species meta-test constraint.")
        return set()
    df = pd.read_csv(D.ID_SPECIES_CSV)
    ids = set(df["canonical_binomial"].dropna().str.lower())
    print(f"Loaded {len(ids)} ID species from {D.ID_SPECIES_CSV.name}")
    return ids


def allocate_proportional(items, ratios):
    """Split items into {split: [items]} by ratio (last split takes remainder)."""
    n = len(items)
    counts, remaining = {}, n
    for i, (split, ratio) in enumerate(ratios.items()):
        if i == len(ratios) - 1:
            counts[split] = remaining
        else:
            c = round(n * ratio)
            counts[split] = c
            remaining -= c
    result, idx = {}, 0
    for split in ratios:
        result[split] = items[idx:idx + counts[split]]
        idx += counts[split]
    return result


def run():
    if D.SPLIT_CSV.exists() and not D.FORCE_DATAPREP:
        print(f"✅ {D.SPLIT_CSV.name} exists → skip split")
        return
    random.seed(D.SEED)
    df = pd.read_csv(D.RAW_METADATA_CSV,
                     usecols=["Species", "Genus", "Family", "Twnumber", "identifier"])
    df["canonical_binomial"] = df["Species"].apply(species_to_canonical)
    df["image_path"] = df["identifier"].apply(lambda x: str(D.ORIGIN_IMAGE_DIR / f"{x}.jpg"))
    print(f"  images={len(df)} species={df['Species'].nunique()} genera={df['Genus'].nunique()}")

    id_species = load_id_species()
    species_info = {}
    for species, group in df.groupby("Species"):
        canonical = group["canonical_binomial"].iloc[0]
        canonical_2w = " ".join(canonical.split()[:2]) if canonical else ""
        species_info[species] = {"genus": group["Genus"].iloc[0],
                                 "is_id": canonical_2w in id_species if canonical else False}
    all_species = list(species_info)

    # Step 1: ID species → meta-test.
    split_assignment = {sp: "meta-test" for sp, info in species_info.items() if info["is_id"]}
    # Step 2: group remaining by genus.
    genus_to_species = defaultdict(list)
    for sp in all_species:
        if sp not in split_assignment:
            genus_to_species[species_info[sp]["genus"]].append(sp)
    large = {g: spp for g, spp in genus_to_species.items() if len(spp) >= 3}
    small = {g: spp for g, spp in genus_to_species.items() if len(spp) < 3}
    ratios = {"meta-train": D.TRAIN_RATIO, "meta-val": D.VAL_RATIO, "meta-test": D.TEST_RATIO}
    # Step 3: large genera proportional within-genus.
    for spp_list in large.values():
        random.shuffle(spp_list)
        for split, spp in allocate_proportional(spp_list, ratios).items():
            for sp in spp:
                split_assignment[sp] = split
    # Step 4: small genera as whole units.
    small_list = list(small.keys())
    random.shuffle(small_list)
    for split, genera in allocate_proportional(small_list, ratios).items():
        for genus in genera:
            for sp in small[genus]:
                split_assignment[sp] = split

    assert len(split_assignment) == len(all_species), "unassigned species remain"
    for sp, info in species_info.items():
        if info["is_id"]:
            assert split_assignment[sp] == "meta-test", f"ID species {sp} not in meta-test"

    df["split"] = df["Species"].map(split_assignment)
    cols = ["Species", "Genus", "Family", "Twnumber", "identifier",
            "canonical_binomial", "image_path", "split"]
    out = df[cols].sort_values(["split", "Species", "Twnumber"])
    D.SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(D.SPLIT_CSV, index=False)
    print(f"💾 {D.SPLIT_CSV.name} ({len(out)} rows)")

    summary = [{"split": s,
                "num_species": out[out["split"] == s]["Species"].nunique(),
                "num_images": int((out["split"] == s).sum()),
                "num_genera": out[out["split"] == s]["Genus"].nunique(),
                "num_families": out[out["split"] == s]["Family"].nunique(),
                "pct_species": f"{out[out['split']==s]['Species'].nunique()/len(all_species)*100:.1f}%"}
               for s in ["meta-train", "meta-val", "meta-test"]]
    pd.DataFrame(summary).to_csv(D.SPLIT_SUMMARY_CSV, index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    run()

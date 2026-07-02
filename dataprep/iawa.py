# -*- coding: utf-8 -*-
"""IAWA anatomical attribute matrix (Paper-2 multi-task supervision).

Lifted from old_version_claim_and_process_data/parse_iawa_attributes.py. Parses
the IAWA columns of the metadata CSV (p/a/v → 1.0/0.0/0.5, missing → NaN),
aggregates per species (mean then round to {0,0.5,1}), merges canonical_binomial
+ split, drops uninformative attributes (0% or 100% present) → 29 attrs.
Output: IAWA_MATRIX_CSV + IAWA_STATS_CSV.

NOTE: the current Paper-1 training subpackage (metric learning) does NOT consume
this; it is produced for completeness / future Paper-2 multi-task training.
"""

import pandas as pd

from . import config as D


def normalize_iawa_value(val):
    if pd.isna(val):
        return float("nan")
    v = str(val).strip().lower()
    return {"p": 1.0, "a": 0.0, "v": 0.5}.get(v, float("nan"))


def round_to_iawa(x):
    if pd.isna(x):
        return float("nan")
    if x <= 0.25:
        return 0.0
    if x >= 0.75:
        return 1.0
    return 0.5


def run():
    if D.IAWA_MATRIX_CSV.exists() and not D.FORCE_DATAPREP:
        print(f"✅ {D.IAWA_MATRIX_CSV.name} exists → skip iawa")
        return
    df = pd.read_csv(D.RAW_METADATA_CSV)
    cols = df.columns.tolist()
    iawa_cols = cols[cols.index("Herbarium voucher Botanic Garden Meise") + 1:cols.index("identifier")]
    print(f"  IAWA columns: {len(iawa_cols)}")

    df_iawa = df[["Species"] + iawa_cols].copy()
    for c in iawa_cols:
        df_iawa[c] = df_iawa[c].apply(normalize_iawa_value)
    df_species = df_iawa.groupby("Species")[iawa_cols].mean().map(round_to_iawa).reset_index()

    split_df = pd.read_csv(D.SPLIT_CSV, usecols=["Species", "canonical_binomial", "split"]).drop_duplicates("Species")
    df_species = df_species.merge(split_df, on="Species", how="left")

    presence = df_species[iawa_cols].apply(lambda col: (col == 1.0).sum() / col.notna().sum() * 100)
    uninformative = presence[(presence == 0) | (presence == 100)].index.tolist()
    if uninformative:
        print(f"  Dropping {len(uninformative)} uninformative attrs (0%/100%)")
        iawa_cols = [c for c in iawa_cols if c not in uninformative]

    df_species = df_species[["Species", "canonical_binomial", "split"] + iawa_cols]
    D.IAWA_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_species.to_csv(D.IAWA_MATRIX_CSV, index=False)
    print(f"💾 {D.IAWA_MATRIX_CSV.name} ({len(df_species)} species × {len(iawa_cols)} attrs)")

    stats = []
    for c in iawa_cols:
        vals = df_species[c]
        n_total = len(vals); n_nan = int(vals.isna().sum())
        n_present = int((vals == 1.0).sum()); n_absent = int((vals == 0.0).sum())
        n_var = int((vals == 0.5).sum()); denom = max(n_total - n_nan, 1)
        stats.append({"attribute": c, "n_total": n_total, "n_present": n_present,
                      "n_absent": n_absent, "n_variable": n_var, "n_missing": n_nan,
                      "completeness_pct": round((n_total - n_nan) / n_total * 100, 1),
                      "presence_rate": round(n_present / denom * 100, 1),
                      "variability_rate": round(n_var / denom * 100, 1)})
    pd.DataFrame(stats).to_csv(D.IAWA_STATS_CSV, index=False)
    print(f"💾 {D.IAWA_STATS_CSV.name}")


if __name__ == "__main__":
    run()

# -*- coding: utf-8 -*-
"""Standardize public wood datasets + ID/OOD classification vs SmartWoodID.

Lifted from old_version_claim_and_process_data/standardize_public_datasets.py.
Scans 9 public datasets, runs the 5-stage name standardization, resolves names
via GBIF (match → search), and labels each species ID (overlaps SmartWoodID) or
OOD. Outputs all_public_datasets_standardized.csv + ID/OOD_species_public.csv,
then expands those to per-image ID/OOD_images_expanded.csv via data.expand_public_csv.

GBIF_OFFLINE=1 (default) uses ONLY gbif_cache.json — never hits the network — so
a cache miss is recorded as "Not found (offline)" instead of stalling a run.
"""

import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

import pandas as pd

from . import config as D
from ..data import expand_public_csv

DATASET_CONFIGS = {
    "BD11": {"structure": "flat"}, "BFS46": {"structure": "flat"},
    "DTSR14": {"structure": "flat", "name_type": "polish"}, "FSDM41": {"structure": "flat"},
    "GOIMAI": {"structure": "flat"}, "PCA11": {"structure": "flat"},
    "VN26": {"structure": "magnification", "magnifications": ["x10", "x20", "x50"]},
    "WOODAUTH": {"structure": "flat"}, "WRD25": {"structure": "flat"},
}
POLISH_TO_LATIN = {
    "brzoza": "Betula pendula", "buk": "Fagus sylvatica", "dab": "Quercus robur",
    "grab": "Carpinus betulus", "jawor": "Acer pseudoplatanus", "jesion": "Fraxinus excelsior",
    "jodla": "Abies alba", "lipa": "Tilia cordata", "modrzew": "Larix decidua",
    "olsza": "Alnus glutinosa", "sosna": "Pinus sylvestris", "swierk": "Picea abies",
    "wiazy": "Ulmus glabra", "wierzba": "Salix alba",
}
SPELLING_CORRECTIONS = {
    "bertholethia excelsa": "bertholletia excelsa", "grevilea robusta": "grevillea robusta",
    "cedrelinga catenaeformis": "cedrelinga cateniformis", "cupresus lusitanica": "cupressus lusitanica",
    "cinnamomum czmphora": "cinnamomum camphora",
}
LOCAL_TO_SCIENTIFIC = {"nogal cafetero": "cordia alliodora", "cedro costeno": "cedrela odorata"}


def _correction_path():
    candidates = [
        D.C.ROOT_PATH / "dataset_label_corrections.json",
        Path(__file__).resolve().parents[2] / "dataset_label_corrections.json",
        Path.cwd() / "dataset_label_corrections.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_public_label_corrections():
    path = _correction_path()
    if not path.exists():
        print(f"  ⚠️ public label correction file not found: {path}")
        return {}, path
    payload = json.load(open(path, encoding="utf-8"))
    return payload, path


def _fsdm41_override_map(payload):
    if not D.APPLY_FSDM41_CORRECTION:
        return {}
    return (payload.get("FSDM41", {}) or {}).get("overrides", {}) or {}


# ── 5-stage name standardization ─────────────────────────────────────────────
def standardize_species_name(name):
    name = str(name).lower().strip()
    name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"_", " ", name)
    name = re.sub(r"^\d+-?", "", name)
    return name.strip()


def lookup_polish(name, dataset):
    if DATASET_CONFIGS.get(dataset, {}).get("name_type") == "polish":
        latin = POLISH_TO_LATIN.get(name)
        if latin:
            return latin.lower()
        print(f"  ⚠️ Unknown Polish name: '{name}' in {dataset}")
    return name


def strip_botanical_authority(name):
    name = name.strip()
    var_match = re.match(r"^(\w+)\s+(\w+)\s+(var\.|subsp\.)\s+(\w+)", name, re.IGNORECASE)
    if var_match:
        return f"{var_match.group(1)} {var_match.group(2)} {var_match.group(3).lower()} {var_match.group(4)}".lower()
    words = name.split()
    return f"{words[0]} {words[1]}".lower() if len(words) >= 2 else name.lower()


def correct_spelling(name):
    return SPELLING_CORRECTIONS.get(name, name)


def handle_variants(name):
    spp_match = re.match(r"^(\w+)\s+spp?\.?\s*(\d*)$", name, re.IGNORECASE)
    if spp_match:
        genus, num = spp_match.group(1), spp_match.group(2)
        return f"{genus} sp", f"spp.{' ' + num if num else ''}"
    num_match = re.match(r"^(\w+\s+\w+)\s+(\d+)$", name)
    if num_match:
        return num_match.group(1), f"specimen {num_match.group(2)}"
    return name, ""


def apply_local_names(name):
    return LOCAL_TO_SCIENTIFIC.get(name, name)


def preprocess_name(original_name, dataset):
    std = standardize_species_name(original_name)
    std = lookup_polish(std, dataset)
    std = strip_botanical_authority(std)
    std = apply_local_names(std)
    std = correct_spelling(std)
    query_name, variant = handle_variants(std)
    return query_name, variant, std


# ── GBIF (offline-by-cache) ──────────────────────────────────────────────────
def load_gbif_cache():
    if D.GBIF_CACHE.exists():
        return json.load(open(D.GBIF_CACHE, encoding="utf-8"))
    return {}


def save_gbif_cache(cache):
    D.GBIF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(D.GBIF_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_canonical_binomial(canonical_full, query_name=""):
    if not canonical_full:
        return ""
    q = query_name.strip().lower()
    if q.endswith(" sp") or q.endswith(" spp"):
        return f"{q.split()[0]} sp."
    words = canonical_full.strip().split()
    return f"{words[0]} {words[1]}".lower() if len(words) >= 2 else canonical_full.lower()


def get_gbif_info(name, cache):
    cache_key = name.strip().lower()
    if cache_key in cache:
        return cache[cache_key]
    result = {"input_name": name, "matched_name": None, "status": "Not found",
              "accepted_name": None, "canonical_full": None}
    if D.GBIF_OFFLINE:
        result["status"] = "Not found (offline)"
        cache[cache_key] = result
        return result
    import requests
    is_genus_only = len(name.strip().split()) == 1
    try:
        params = {"name": name, "kingdom": "Plantae"}
        if is_genus_only:
            params["rank"] = "GENUS"
        data = requests.get("https://api.gbif.org/v1/species/match", params=params, timeout=10).json()
        if "usageKey" in data and data.get("matchType") != "NONE":
            result["matched_name"] = data.get("scientificName")
            result["status"] = data.get("status")
            result["accepted_name"] = data.get("acceptedUsage", {}).get("scientificName") or data.get("scientificName")
            result["canonical_full"] = result["accepted_name"]
            cache[cache_key] = result
            return result
        sp = {"q": name, "limit": 1, "highertaxonKey": 6}
        if is_genus_only:
            sp["rank"] = "GENUS"
        s_data = requests.get("https://api.gbif.org/v1/species/search", params=sp, timeout=10).json()
        if s_data.get("results"):
            res = s_data["results"][0]
            result["matched_name"] = res.get("scientificName")
            result["status"] = res.get("taxonomicStatus")
            result["accepted_name"] = res.get("acceptedScientificName") or res.get("scientificName")
            result["canonical_full"] = result["accepted_name"]
    except Exception as e:  # noqa: BLE001
        result["status"] = f"Error: {e}"
    cache[cache_key] = result
    return result


# ── Filesystem scan ──────────────────────────────────────────────────────────
def count_images(folder):
    try:
        return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in D.IMAGE_EXTENSIONS)
    except Exception:  # noqa: BLE001
        return 0


def scan_dataset(name, cfg):
    records, root = [], D.PUBLIC_DATASETS_DIR / name
    if not root.exists():
        print(f"  ❌ not found: {root}")
        return records
    if cfg["structure"] == "magnification":
        for mag in cfg["magnifications"]:
            mp = root / mag
            if not mp.exists():
                continue
            for item in sorted(mp.iterdir()):
                if item.is_dir():
                    records.append({"dataset": name, "original_name": item.name,
                                    "folder_path": str(item), "image_count": count_images(item),
                                    "magnification": mag})
    else:
        for item in sorted(root.iterdir()):
            if item.is_dir():
                records.append({"dataset": name, "original_name": item.name,
                                "folder_path": str(item), "image_count": count_images(item),
                                "magnification": ""})
    return records


def _grouped(df_sub):
    return df_sub.groupby("canonical_binomial", as_index=False).agg({
        "dataset": list, "original_name": list, "folder_path": list,
        "image_count": "sum", "magnification": list,
        "gbif_status": "first", "gbif_accepted_name": "first",
    }).sort_values("canonical_binomial")


def _before_correction_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".before_public_correction" + path.suffix)


def _backup_existing(path):
    path = Path(path)
    backup = _before_correction_path(path)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
        print(f"  ↻ Backed up pre-correction file: {backup}")
    return backup


def _ensure_old_expanded_backup(species_backup, expanded_path):
    """Create the pre-correction expanded CSV if only the species CSV exists.

    This lets cache migration proceed on Colab even when the intermediate
    OOD_images_expanded.csv was not saved/uploaded. The expanded backup is
    generated from the pre-correction species CSV before the corrected expanded
    CSV overwrites the active filename.
    """
    species_backup = Path(species_backup)
    expanded_backup = _before_correction_path(expanded_path)
    if expanded_backup.exists() or not species_backup.exists():
        return expanded_backup
    print(f"  ↻ Creating pre-correction expanded CSV from {species_backup.name}")
    old_force = os.environ.get("FORCE_PUBLIC_EXPAND")
    os.environ["FORCE_PUBLIC_EXPAND"] = "0"
    try:
        expand_public_csv(species_backup, expanded_backup)
    finally:
        if old_force is None:
            os.environ.pop("FORCE_PUBLIC_EXPAND", None)
        else:
            os.environ["FORCE_PUBLIC_EXPAND"] = old_force
    return expanded_backup


def _canonical_from_corrected_label(label_name, dataset):
    q, _, _ = preprocess_name(label_name, dataset)
    return q.strip().lower()


def _is_woodauth_row(row):
    src = str(row.get("source_dataset", "")).upper()
    path = str(row.get("file_path", "")).replace("\\", "/").upper()
    return src == "WOODAUTH" or "/WOODAUTH/" in path


def _is_fsdm41_row(row):
    src = str(row.get("source_dataset", "")).upper()
    path = str(row.get("file_path", "")).replace("\\", "/").upper()
    return src == "FSDM41" or "/FSDM41/" in path


def _correct_expanded_public_df(df, fsdm41_overrides):
    df = df.copy()
    if "source_dataset" not in df.columns:
        df["source_dataset"] = ""
    if "source_original_name" not in df.columns:
        df["source_original_name"] = ""
    df["label_correction"] = ""
    df["corrected_name"] = ""

    keep = []
    for idx, row in df.iterrows():
        if D.EXCLUDE_WOODAUTH and _is_woodauth_row(row):
            keep.append(False)
            df.at[idx, "label_correction"] = "WOODAUTH_EXCLUDED"
            continue
        keep.append(True)
        if _is_fsdm41_row(row):
            original = str(row.get("source_original_name", ""))
            corrected = fsdm41_overrides.get(original)
            if corrected:
                df.at[idx, "label"] = _canonical_from_corrected_label(corrected, "FSDM41")
                df.at[idx, "label_correction"] = "FSDM41_PERMUTED_LABEL"
                df.at[idx, "corrected_name"] = corrected
    return df.loc[keep].reset_index(drop=True), df.loc[[not x for x in keep]].reset_index(drop=True)


def _infer_magnification_from_path(path):
    parts = [p.lower() for p in Path(str(path)).parts]
    for mag in ("x10", "x20", "x50"):
        if mag in parts:
            return mag
    return ""


def _species_csv_from_expanded(df, out_path):
    rows = []
    for label, g in df.groupby("label", sort=True):
        folder_paths = sorted({str(Path(p).parent) for p in g["file_path"].astype(str)})
        datasets = []
        originals = []
        magnifications = []
        image_count = 0
        for folder in folder_paths:
            sub = g[g["file_path"].astype(str).map(lambda p, f=folder: str(Path(p).parent) == f)]
            first = sub.iloc[0]
            datasets.append(str(first.get("source_dataset", "")))
            originals.append(str(first.get("source_original_name", "")))
            magnifications.append(_infer_magnification_from_path(folder))
            image_count += int(len(sub))
        rows.append({
            "canonical_binomial": label,
            "dataset": datasets,
            "original_name": originals,
            "folder_path": folder_paths,
            "image_count": image_count,
            "magnification": magnifications,
            "gbif_status": "v3_artifact",
            "gbif_accepted_name": "",
        })
    out = pd.DataFrame(rows).sort_values("canonical_binomial")
    out.to_csv(out_path, index=False)
    return out


def _run_v3_artifact_correction(corrections, corrections_path, fsdm41_overrides):
    """Correct public CSVs by migrating the audited v3 public artifacts.

    The original v3 pipeline already fixed the ID/OOD split using GBIF and the
    SmartWoodID reference list. For the public-label correction, do not rerun
    that taxonomy pipeline. Instead, preserve the v3 ID/OOD assignment, remove
    WoodAuth rows, and relabel FSDM41 rows by file path/source folder.
    """
    if not D.ID_SPECIES_CSV.exists() or not D.OOD_SPECIES_CSV.exists():
        raise FileNotFoundError(
            "v3 artifact correction requires ID_species_public.csv and "
            "OOD_species_public.csv under ROOT_PATH."
        )
    old_id_species = _backup_existing(D.ID_SPECIES_CSV)
    old_ood_species = _backup_existing(D.OOD_SPECIES_CSV)
    _backup_existing(D.ID_IMAGES_CSV)
    _backup_existing(D.OOD_IMAGES_CSV)
    old_id_expanded = _ensure_old_expanded_backup(old_id_species, D.ID_IMAGES_CSV)
    old_ood_expanded = _ensure_old_expanded_backup(old_ood_species, D.OOD_IMAGES_CSV)

    id_old = pd.read_csv(old_id_expanded)
    ood_old = pd.read_csv(old_ood_expanded)
    id_new, id_dropped = _correct_expanded_public_df(id_old, fsdm41_overrides)
    ood_new, ood_dropped = _correct_expanded_public_df(ood_old, fsdm41_overrides)

    print(
        "  v3 artifact source: "
        f"ID={len(id_old)} images/{id_old['label'].nunique()} species, "
        f"OOD={len(ood_old)} images/{ood_old['label'].nunique()} species"
    )
    print(
        "  corrections applied: "
        f"FSDM41 relabelled={(id_new['label_correction'].eq('FSDM41_PERMUTED_LABEL').sum() + ood_new['label_correction'].eq('FSDM41_PERMUTED_LABEL').sum())} images, "
        f"WoodAuth dropped={len(id_dropped) + len(ood_dropped)} images"
    )

    if id_new["label"].nunique() < 20:
        raise RuntimeError(
            f"Corrected ID split has only {id_new['label'].nunique()} species; "
            "refusing to overwrite v3 artifacts."
        )
    if len(ood_new) >= len(ood_old) and D.EXCLUDE_WOODAUTH:
        raise RuntimeError(
            "WoodAuth exclusion was requested but no OOD rows were dropped; "
            "check that v3 expanded CSVs include source_dataset/source paths."
        )
    overlap = set(id_new["label"].astype(str)) & set(ood_new["label"].astype(str))
    if overlap:
        raise RuntimeError(f"Corrected ID/OOD overlap: {sorted(overlap)[:10]}")

    id_new.to_csv(D.ID_IMAGES_CSV, index=False)
    ood_new.to_csv(D.OOD_IMAGES_CSV, index=False)
    id_species = _species_csv_from_expanded(id_new, D.ID_SPECIES_CSV)
    ood_species = _species_csv_from_expanded(ood_new, D.OOD_SPECIES_CSV)

    audit_rows = []
    for side, old_df, new_df, dropped_df in [
        ("ID", id_old, id_new, id_dropped),
        ("OOD", ood_old, ood_new, ood_dropped),
    ]:
        fsdm_mask = new_df.get("label_correction", pd.Series([], dtype=str)).eq("FSDM41_PERMUTED_LABEL")
        audit_rows.append({
            "item": f"{side}_v3_artifact_migration",
            "status": "corrected",
            "correction_file": str(corrections_path),
            "old_images": int(len(old_df)),
            "new_images": int(len(new_df)),
            "old_species": int(old_df["label"].nunique()),
            "new_species": int(new_df["label"].nunique()),
            "dropped_images": int(len(dropped_df)),
            "corrected_images": int(fsdm_mask.sum()),
            "note": "Preserved v3 ID/OOD split; corrected FSDM41 labels and removed WoodAuth rows only.",
        })
    if D.EXCLUDE_WOODAUTH:
        wa = (corrections.get("WOODAUTH", {}) or {})
        audit_rows.append({
            "item": "WOODAUTH",
            "status": "excluded",
            "correction_file": str(corrections_path),
            "old_images": int(len(id_dropped) + len(ood_dropped)),
            "new_images": 0,
            "old_species": int(wa.get("n_species", 0) or 0),
            "new_species": 0,
            "dropped_images": int(len(id_dropped) + len(ood_dropped)),
            "corrected_images": 0,
            "note": "Excluded because public release mixes transverse and longitudinal sections without per-image plane labels.",
        })
    pd.DataFrame(audit_rows).to_csv(D.C.PUBLIC_LABEL_AUDIT_CSV, index=False)
    print(f"  ✅ v3 artifact correction: {len(id_species)} ID / {len(ood_species)} OOD species")
    print(f"  ✅ Expanded corrected public CSVs: ID={len(id_new)} images, OOD={len(ood_new)} images")
    print(f"  ✅ Public label audit saved: {D.C.PUBLIC_LABEL_AUDIT_CSV}")
    return True


def run():
    corrections, corrections_path = _load_public_label_corrections()
    fsdm41_overrides = _fsdm41_override_map(corrections)
    if D.EXCLUDE_WOODAUTH or fsdm41_overrides:
        _run_v3_artifact_correction(corrections, corrections_path, fsdm41_overrides)
        return

    dataset_configs = dict(DATASET_CONFIGS)
    if D.EXCLUDE_WOODAUTH and "WOODAUTH" in dataset_configs:
        dataset_configs.pop("WOODAUTH")
        print("  WOODAUTH: excluded by public-label correction policy")

    if D.ID_SPECIES_CSV.exists() and D.OOD_SPECIES_CSV.exists() and not D.FORCE_DATAPREP:
        print(f"✅ {D.ID_SPECIES_CSV.name} + {D.OOD_SPECIES_CSV.name} exist → skip standardize")
    else:
        old_id_species = _backup_existing(D.ID_SPECIES_CSV)
        old_ood_species = _backup_existing(D.OOD_SPECIES_CSV)
        _backup_existing(D.ID_IMAGES_CSV)
        _backup_existing(D.OOD_IMAGES_CSV)
        _ensure_old_expanded_backup(old_id_species, D.ID_IMAGES_CSV)
        _ensure_old_expanded_backup(old_ood_species, D.OOD_IMAGES_CSV)

        all_records = []
        for ds, cfg in dataset_configs.items():
            recs = scan_dataset(ds, cfg)
            all_records.extend(recs)
            print(f"  {ds}: {len(recs)} folders")
        for rec in all_records:
            label_name = rec["original_name"]
            rec["label_correction"] = ""
            rec["corrected_name"] = ""
            if rec["dataset"] == "FSDM41" and label_name in fsdm41_overrides:
                rec["label_correction"] = "FSDM41_PERMUTED_LABEL"
                rec["corrected_name"] = fsdm41_overrides[label_name]
                label_name = rec["corrected_name"]
            q, v, std = preprocess_name(label_name, rec["dataset"])
            rec.update(query_name=q, variant=v, standardized_name=std)

        cache = load_gbif_cache()
        uncached = sorted({r["query_name"] for r in all_records if r["query_name"].strip().lower() not in cache})
        print(f"  GBIF: {len(uncached)} uncached queries (offline={D.GBIF_OFFLINE})")
        for name in uncached:
            get_gbif_info(name, cache)
            if not D.GBIF_OFFLINE:
                time.sleep(0.5)
        save_gbif_cache(cache)

        for rec in all_records:
            q = rec["query_name"].strip().lower()
            g = cache.get(q, {})
            rec["gbif_status"] = g.get("status", "Not found")
            rec["gbif_matched_name"] = g.get("matched_name")
            rec["gbif_accepted_name"] = g.get("accepted_name")
            rec["canonical_binomial"] = extract_canonical_binomial(g.get("canonical_full"), query_name=q)

        if D.SWID_GBIF_CSV.exists():
            swid = pd.read_csv(D.SWID_GBIF_CSV)
            swid_canon = {extract_canonical_binomial(c) for c in swid["canonical"].dropna().unique()}
            swid_source = str(D.SWID_GBIF_CSV)
        else:
            id_backup = _before_correction_path(D.ID_SPECIES_CSV)
            if not id_backup.exists():
                raise FileNotFoundError(
                    f"Missing {D.SWID_GBIF_CSV} and fallback {id_backup}; cannot determine public ID species."
                )
            id_prev = pd.read_csv(id_backup)
            swid_canon = set(id_prev["canonical_binomial"].dropna().astype(str).str.lower())
            swid_source = str(id_backup)
            print(f"  ⚠️  {D.SWID_GBIF_CSV.name} missing; using prior ID species CSV as ID/OOD reference")
        print(f"  SmartWoodID reference species: {len(swid_canon)} ({swid_source})")
        for rec in all_records:
            cb = rec["canonical_binomial"]
            rec["distribution"] = "ID" if cb and cb in swid_canon else "OOD"

        cols = ["dataset", "original_name", "standardized_name", "query_name", "variant",
                "canonical_binomial", "gbif_status", "gbif_matched_name", "gbif_accepted_name",
                "folder_path", "image_count", "magnification", "distribution",
                "label_correction", "corrected_name"]
        df_all = pd.DataFrame(all_records)[cols].sort_values(["dataset", "canonical_binomial"])
        D.PUBLIC_STD_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(D.PUBLIC_STD_CSV, index=False)
        df_id = df_all[df_all["distribution"] == "ID"]
        df_ood = df_all[df_all["distribution"] == "OOD"]
        if not df_id.empty:
            _grouped(df_id).to_csv(D.ID_SPECIES_CSV, index=False)
        if not df_ood.empty:
            _grouped(df_ood).to_csv(D.OOD_SPECIES_CSV, index=False)
        print(f"  ✅ {df_id['canonical_binomial'].nunique()} ID / {df_ood['canonical_binomial'].nunique()} OOD species")

        audit_rows = []
        fsdm = df_all[df_all["dataset"].eq("FSDM41")]
        audit_rows.append({
            "item": "FSDM41",
            "status": "corrected" if fsdm41_overrides else "not_corrected",
            "correction_file": str(corrections_path),
            "folders": int(len(fsdm)),
            "images": int(fsdm["image_count"].sum()) if not fsdm.empty else 0,
            "corrected_folders": int((fsdm["label_correction"] == "FSDM41_PERMUTED_LABEL").sum()) if not fsdm.empty else 0,
            "corrected_images": int(fsdm.loc[fsdm["label_correction"] == "FSDM41_PERMUTED_LABEL", "image_count"].sum()) if not fsdm.empty else 0,
            "note": "Folder paths unchanged; species labels corrected before canonicalization.",
        })
        if D.EXCLUDE_WOODAUTH:
            wa = (corrections.get("WOODAUTH", {}) or {})
            audit_rows.append({
                "item": "WOODAUTH",
                "status": "excluded",
                "correction_file": str(corrections_path),
                "folders": int(wa.get("n_species", 0) or 0),
                "images": int(wa.get("n_images_excluded", 0) or 0),
                "corrected_folders": 0,
                "corrected_images": 0,
                "note": "Excluded because public release mixes transverse and longitudinal sections without per-image plane labels.",
            })
        pd.DataFrame(audit_rows).to_csv(D.C.PUBLIC_LABEL_AUDIT_CSV, index=False)
        print(f"  ✅ Public label audit saved: {D.C.PUBLIC_LABEL_AUDIT_CSV}")

    # Per-image expansion (reuse the eval package's parser).
    if D.ID_SPECIES_CSV.exists():
        expand_public_csv(D.ID_SPECIES_CSV, D.ID_IMAGES_CSV)
    if D.OOD_SPECIES_CSV.exists():
        expand_public_csv(D.OOD_SPECIES_CSV, D.OOD_IMAGES_CSV)


if __name__ == "__main__":
    run()

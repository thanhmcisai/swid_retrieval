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
import re
import time
import unicodedata

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


def run():
    if D.ID_SPECIES_CSV.exists() and D.OOD_SPECIES_CSV.exists() and not D.FORCE_DATAPREP:
        print(f"✅ {D.ID_SPECIES_CSV.name} + {D.OOD_SPECIES_CSV.name} exist → skip standardize")
    else:
        all_records = []
        for ds, cfg in DATASET_CONFIGS.items():
            recs = scan_dataset(ds, cfg)
            all_records.extend(recs)
            print(f"  {ds}: {len(recs)} folders")
        for rec in all_records:
            q, v, std = preprocess_name(rec["original_name"], rec["dataset"])
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

        swid = pd.read_csv(D.SWID_GBIF_CSV)
        swid_canon = {extract_canonical_binomial(c) for c in swid["canonical"].dropna().unique()}
        print(f"  SmartWoodID reference species: {len(swid_canon)}")
        for rec in all_records:
            cb = rec["canonical_binomial"]
            rec["distribution"] = "ID" if cb and cb in swid_canon else "OOD"

        cols = ["dataset", "original_name", "standardized_name", "query_name", "variant",
                "canonical_binomial", "gbif_status", "gbif_matched_name", "gbif_accepted_name",
                "folder_path", "image_count", "magnification", "distribution"]
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

    # Per-image expansion (reuse the eval package's parser).
    if D.ID_SPECIES_CSV.exists():
        expand_public_csv(D.ID_SPECIES_CSV, D.ID_IMAGES_CSV)
    if D.OOD_SPECIES_CSV.exists():
        expand_public_csv(D.OOD_SPECIES_CSV, D.OOD_IMAGES_CSV)


if __name__ == "__main__":
    run()

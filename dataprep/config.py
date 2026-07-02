# -*- coding: utf-8 -*-
"""Data-prep paths + constants (env-overridable).

RAW inputs default under the eval package's dataset dirs (config.PSI_DIR /
DATASETS_DIR); OUTPUT paths reuse the eval package's config so the produced
manifest/split/CSVs land exactly where train/eval already read them.
"""

import os
from pathlib import Path

from .. import config as C


def _p(env, default):
    return Path(os.environ.get(env, str(default)))


# ── RAW inputs (on Drive) ────────────────────────────────────────────────────
RAW_IMAGE_DIR = _p("RAW_IMAGE_DIR", C.PSI_DIR / "smart_wood_id_dataset")        # crop input (full microscopy images)
PATCH_OUT_DIR = _p("PATCH_OUT_DIR", C.PSI_DIR / "smartwoodid_patches")          # crop output (image_id/scale_*/patch)
FINAL_DATASET_DIR = _p("FINAL_DATASET_DIR", C.PSI_DIR / "smartwoodid_kmeans_split")  # select_diverse output ({train,test})
ORIGIN_IMAGE_DIR = _p("ORIGIN_IMAGE_DIR", C.PSI_DIR / "smartwoodid_dataset_origin")  # for split image_path column
RAW_METADATA_CSV = _p("RAW_METADATA_CSV", C.PSI_DIR / "smartwoodid_metadata.csv")    # Species/Genus/.../IAWA cols
SWID_GBIF_CSV = _p("SWID_GBIF_CSV", C.PSI_DIR / "smartwoodid_gbif_species_check.csv")  # SWID canonical reference
PUBLIC_DATASETS_DIR = _p("PUBLIC_DATASETS_DIR", C.DATASETS_DIR)                 # 9 public wood datasets

# ── OUTPUTs (reuse eval-package locations) ───────────────────────────────────
SPLIT_CSV = C.SPLIT_CSV
SPLIT_SUMMARY_CSV = SPLIT_CSV.parent / "smartwoodid_split_summary.csv"
MANIFEST_PATH = C.MANIFEST_PATH
ID_SPECIES_CSV = C.ID_SPECIES_CSV
OOD_SPECIES_CSV = C.OOD_SPECIES_CSV
ID_IMAGES_CSV = C.ID_IMAGES_CSV
OOD_IMAGES_CSV = C.OOD_IMAGES_CSV
IAWA_MATRIX_CSV = _p("IAWA_MATRIX_CSV", C.ROOT_PATH / "smartwoodid_iawa_matrix.csv")
IAWA_STATS_CSV = _p("IAWA_STATS_CSV", C.ROOT_PATH / "smartwoodid_iawa_stats.csv")
PUBLIC_STD_CSV = _p("PUBLIC_STD_CSV", C.ROOT_PATH / "all_public_datasets_standardized.csv")
GBIF_CACHE = _p("GBIF_CACHE", C.ROOT_PATH / "gbif_cache.json")

# ── Constants (lifted from the source scripts) ───────────────────────────────
SEED = int(os.environ.get("DATAPREP_SEED", "42"))
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.60, 0.10, 0.30
BASE_SIZE = int(os.environ.get("PATCH_BASE_SIZE", "256"))
LINEAR_THRESH = 1024
GEOMETRIC_FACTOR = 1.5
SHARPNESS_THRESH = float(os.environ.get("SHARPNESS_THRESH", "5.0"))
ENTROPY_THRESH = float(os.environ.get("ENTROPY_THRESH", "3.5"))
BLACK_RATIO_THRESH = 0.5
CRACK_ASPECT_RATIO_THRESH = 5.0
MAX_PATCHES_PER_GROUP = int(os.environ.get("MAX_PATCHES_PER_GROUP", "100"))
SELECT_TEST_SIZE = 0.30  # image-level train/test dir split in select_diverse
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ── Flags ────────────────────────────────────────────────────────────────────
GBIF_OFFLINE = os.environ.get("GBIF_OFFLINE", "1") == "1"   # default: cache-only, no network
FORCE_DATAPREP = os.environ.get("FORCE_DATAPREP", "0") == "1"
RUN_DATAPREP_CROP = os.environ.get("RUN_DATAPREP_CROP", "1") == "1"
RUN_DATAPREP_SELECT = os.environ.get("RUN_DATAPREP_SELECT", "1") == "1"
RUN_DATAPREP_SPLIT = os.environ.get("RUN_DATAPREP_SPLIT", "1") == "1"
RUN_DATAPREP_IAWA = os.environ.get("RUN_DATAPREP_IAWA", "1") == "1"
RUN_DATAPREP_PUBLIC = os.environ.get("RUN_DATAPREP_PUBLIC", "1") == "1"
RUN_DATAPREP_MANIFEST = os.environ.get("RUN_DATAPREP_MANIFEST", "1") == "1"


def num_workers():
    return max(1, (os.cpu_count() or 2) - 1)

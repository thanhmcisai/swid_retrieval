# -*- coding: utf-8 -*-
"""Datasets, transforms, label canonicalization, manifest/CSV loading.

Lifted verbatim (only re-parameterized for the package) from
final_metric_learning_cea_2026.py:407-695. The image disk cache and dataset
classes are unchanged so extraction reproduces the validated embeddings exactly.
"""

import ast
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from . import config

cv2.setNumThreads(0)

CACHE_DIR = Path(os.environ.get("IMAGE_CACHE_DIR", "/content/cache_images"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Public-CSV folder paths were authored on a local machine; rewrite to Drive.
LOCAL_PREFIX = "/Users/admin/Downloads/Experimentals/datasets"
DRIVE_PREFIX = str(config.DATASETS_DIR)


def canonical_label(s):
    """Canonical species label: lowercase, spaces→underscores, strip."""
    return str(s).replace(" ", "_").lower().strip()


def get_transforms(img_size=224, augment=False):
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.LongestMaxSize(max_size=int(img_size * 1.1)),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _cache_one(src):
    """Copy one image from Drive into the local JPEG cache (md5-named to match
    CachedImageLoader._cache_path). Idempotent."""
    src = str(src)
    dst = CACHE_DIR / (hashlib.md5(src.encode()).hexdigest() + ".jpg")
    if dst.exists() and dst.stat().st_size > 0:
        return "hit"
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return "bad"
    ok = cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return "new" if ok else "bad"


def preload_image_cache(paths, max_workers=16, desc="Preloading images"):
    """Pre-warm the local image cache in parallel BEFORE extraction.

    Google Drive has high per-file latency; fetching all images once with a thread
    pool is far faster than the lazy per-image caching that happens through the
    DataLoader. Files land in /content/cache_images with the exact md5 names
    CachedImageLoader expects, so the subsequent extraction reads them locally.
    Idempotent: re-running only re-fetches missing/empty files.
    """
    paths = list(dict.fromkeys(str(p) for p in paths))
    print(f"Preloading {len(paths):,} images to {CACHE_DIR} ({max_workers} workers)...")
    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        def _tqdm(it, **k):
            return it
    stats = {"hit": 0, "new": 0, "bad": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_cache_one, p) for p in paths]
        for fut in _tqdm(as_completed(futures), total=len(futures), desc=desc):
            stats[fut.result()] += 1
    print(f"image cache: {stats} → {CACHE_DIR}")
    if stats["bad"]:
        print(f"⚠️  {stats['bad']} images could not be read (will retry lazily during extraction).")
    return stats


def collect_all_image_paths(root_path=None, include_swi=True, include_public=True):
    """Collect Drive image paths that may be read by the full paper pipeline.

    This mirrors the old one-cell Colab preloader, but keeps the logic inside the
    package. Returned paths are de-duplicated while preserving order. Public CSV
    paths are normalized from the original local-machine prefix to the active
    Drive dataset root when needed.
    """
    root = Path(root_path or config.ROOT_PATH)
    paths = []

    if include_swi:
        manifest_path = root / "swi_manifest.json"
        if manifest_path.exists():
            m = json.load(open(manifest_path))
            for split in ("meta-train", "meta-val", "meta-test"):
                paths.extend([p for p, _ in m.get(split, [])])
        else:
            print(f"⚠️  SWI manifest not found for image preload: {manifest_path}")

    if include_public:
        drive_prefix = str(root / "datasets")
        for csv_name in ("ID_images_expanded.csv", "OOD_images_expanded.csv"):
            csv_path = root / csv_name
            if not csv_path.exists():
                print(f"⚠️  public image CSV not found for image preload: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            if "file_path" not in df.columns:
                print(f"⚠️  public image CSV has no file_path column: {csv_path}")
                continue
            for p in df["file_path"].astype(str):
                paths.append(p.replace(LOCAL_PREFIX, drive_prefix))

    return list(dict.fromkeys(str(p) for p in paths))


def fix_public_species_csv_paths(root_path=None):
    """Rewrite legacy local public-dataset folder paths to the active ROOT_PATH."""
    root = Path(root_path or config.ROOT_PATH)
    datasets_dir = root / "datasets"
    changed = {}
    for csv_name in ("ID_species_public.csv", "OOD_species_public.csv"):
        src = root / csv_name
        if not src.exists():
            print(f"⚠️  {csv_name} not found")
            changed[csv_name] = False
            continue
        df = pd.read_csv(src)
        if "folder_path" not in df.columns:
            changed[csv_name] = False
            continue
        old_folder_path = df["folder_path"].astype(str)
        new_folder_path = old_folder_path.str.replace(
            LOCAL_PREFIX, str(datasets_dir), regex=False)
        if not old_folder_path.equals(new_folder_path):
            df["folder_path"] = new_folder_path
            df.to_csv(src, index=False)
            print(f"✅ Fixed paths in {csv_name}")
            changed[csv_name] = True
        else:
            print(f"✅ Paths already fixed in {csv_name}")
            changed[csv_name] = False
    return changed


def fix_public_expanded_csv_paths(root_path=None):
    """Rewrite legacy local public image paths in expanded per-image CSVs."""
    root = Path(root_path or config.ROOT_PATH)
    datasets_dir = root / "datasets"
    changed = {}
    for csv_name in ("ID_images_expanded.csv", "OOD_images_expanded.csv"):
        src = root / csv_name
        if not src.exists():
            changed[csv_name] = False
            continue
        df = pd.read_csv(src)
        if "file_path" not in df.columns:
            changed[csv_name] = False
            continue
        old_file_path = df["file_path"].astype(str)
        new_file_path = old_file_path.str.replace(
            LOCAL_PREFIX, str(datasets_dir), regex=False)
        if not old_file_path.equals(new_file_path):
            df["file_path"] = new_file_path
            df.to_csv(src, index=False)
            print(f"✅ Fixed paths in {csv_name}")
            changed[csv_name] = True
        else:
            print(f"✅ Paths already fixed in {csv_name}")
            changed[csv_name] = False
    return changed


def prepare_public_csvs(root_path=None):
    """One-call Colab setup for public ID/OOD CSVs.

    Ensures Drive paths in the species CSVs are valid, then creates the expanded
    per-image CSVs if they are missing. Existing expanded CSVs are reused.
    """
    root = Path(root_path or config.ROOT_PATH)
    fix_public_species_csv_paths(root)
    id_species = root / "ID_species_public.csv"
    ood_species = root / "OOD_species_public.csv"
    if id_species.exists():
        expand_public_csv(id_species, root / "ID_images_expanded.csv", root / "datasets")
    if ood_species.exists():
        expand_public_csv(ood_species, root / "OOD_images_expanded.csv", root / "datasets")
    fix_public_expanded_csv_paths(root)
    try:
        id_n = pd.read_csv(id_species).shape[0] if id_species.exists() else 0
        ood_n = pd.read_csv(ood_species).shape[0] if ood_species.exists() else 0
        print(f"ID species:  {id_n}")
        print(f"OOD species: {ood_n}")
    except Exception:
        pass


class CachedImageLoader:
    def __init__(self, use_ram_cache=False, jpeg_quality=95):
        self.ram_cache = {} if use_ram_cache else None
        self.jpeg_quality = jpeg_quality

    def _cache_path(self, path):
        h = hashlib.md5(path.encode()).hexdigest()
        return CACHE_DIR / f"{h}.jpg"

    def load(self, path):
        if self.ram_cache is not None and path in self.ram_cache:
            return self.ram_cache[path]
        cache_path = self._cache_path(path)
        if cache_path.exists():
            img = cv2.imread(str(cache_path))
            if img is None:
                cache_path.unlink(missing_ok=True)
                return self.load(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cache_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if self.ram_cache is not None:
            self.ram_cache[path] = img
        return img


class ManifestDataset(Dataset):
    """Fast dataset from manifest items [(path, species), ...]."""

    def __init__(self, items, transform=None, scales=None):
        self.transform = transform
        if scales is not None and scales != "all":
            scale_dirs = {f"scale_{s}" for s in scales}
            items = [it for it in items if Path(it[0]).parent.name in scale_dirs]
        species = sorted(set(it[1] for it in items))
        self.loader = CachedImageLoader(use_ram_cache=False)
        self.class_to_idx = {sp: i for i, sp in enumerate(species)}
        self.idx_to_class = {i: sp for sp, i in self.class_to_idx.items()}
        self.samples = [(it[0], self.class_to_idx[it[1]]) for it in items]
        print(f"  ManifestDataset: {len(self.class_to_idx)} classes, {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = self.loader.load(path)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


class CSVImageDataset(Dataset):
    """Load images from expanded CSV (file_path, label columns)."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.transform = transform
        self.loader = CachedImageLoader(use_ram_cache=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self.loader.load(str(row["file_path"]))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.class_to_idx[row["label"]]


def load_swi_manifest(manifest_path=None):
    """Load the SWI manifest JSON (built once by the training pipeline)."""
    manifest_path = Path(manifest_path or config.MANIFEST_PATH)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing SWI manifest: {manifest_path}. Build it once with the training "
            "pipeline (smartwoodid_experiments_full.py / build_swi_manifest)."
        )
    m = json.load(open(manifest_path))
    for s in ["meta-train", "meta-val", "meta-test"]:
        n_sp = len({it[1] for it in m[s]})
        print(f"  {s:12s}: {len(m[s]):6d} images, {n_sp:4d} species")
    return m


def full_swi_items(manifest):
    """All 954-species SWI pool = meta-train ∪ meta-val ∪ meta-test."""
    return manifest["meta-train"] + manifest["meta-val"] + manifest["meta-test"]


def ce_train_image_paths(manifest, seed=42, train_frac=0.65):
    """Reproduce CE-Full's deterministic TRAIN image partition.

    Exact replay of the split in train_ce_classifier
    (final_metric_learning_cea_2026.py:986-1000 / smartwoodid_experiments_full.py):
    image-level, stratified by (species, scale), 65/15/20, seed=42, over the fixed
    meta-train∪val∪test order. Returns the set of training image paths so the
    full-954 gallery can be restricted to exactly the images CE-Full learned from
    (a fairness robustness control). Assumes swi_manifest.json is unchanged since
    CE-Full was trained (its fingerprint is recorded in the cache meta).
    """
    all_items = full_swi_items(manifest)
    rng = np.random.RandomState(int(seed))
    groups = defaultdict(list)
    for path, species in all_items:
        scale = Path(path).parent.name
        groups[(species, scale)].append(path)
    train = set()
    for _key, paths in groups.items():
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        n_tr = max(1, int(train_frac * len(paths)))
        for i in idx[:n_tr]:
            train.add(paths[i])
    return train


def expand_public_csv(csv_path, cache_path, datasets_dir=None):
    """Parse folder_path list-strings → per-image CSV (file_path, label)."""
    cache = Path(cache_path)
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"✅ Cached: {cache} ({len(df)} images)")
        return df
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        for folder in ast.literal_eval(row["folder_path"]):
            folder = folder.replace(LOCAL_PREFIX, DRIVE_PREFIX)
            p = Path(folder)
            if not p.exists():
                continue
            for img in p.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    rows.append({"file_path": str(img), "label": row["canonical_binomial"]})
    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    print(f"✅ Expanded: {len(result)} images → {cache}")
    return result

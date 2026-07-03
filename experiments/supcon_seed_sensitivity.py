# -*- coding: utf-8 -*-
"""SupCon multi-seed OOD sensitivity.

SC-URD seed sensitivity is embedding-cache/lightweight. SupCon is different:
it is an image backbone checkpoint, so each seed requires training a ConvNeXt
SupCon model and extracting ID/OOD/SWI-gallery embeddings from images. This
module is therefore opt-in (`RUN_SUPCON_SEED_SENSITIVITY=1`).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import config
from ..data import CSVImageDataset, ManifestDataset, full_swi_items, get_transforms, load_swi_manifest
from ..models import _load_metric
from ..training import config as train_config
from ..training.common import extract_embeddings
from ..training.train_backbones import train_metric_model
from .ood_baselines import _auroc, _fpr95, _aupr
from .registry import _norm


def _csv_env(name, default, cast=int):
    out = []
    for item in str(os.environ.get(name, default)).split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def _distance_scores(query, gallery):
    return 1.0 - (_norm(query) @ _norm(gallery).T).max(axis=1)


def _id_species():
    df = pd.read_csv(config.ID_IMAGES_CSV)
    return set(df["label"].astype(str).str.replace(" ", "_").str.lower().str.strip())


def _gallery_items_id_only():
    id_sp = _id_species()
    manifest = load_swi_manifest()
    return [(p, sp) for p, sp in full_swi_items(manifest)
            if str(sp).replace(" ", "_").lower().strip() in id_sp]


def _extract_for_checkpoint(ckpt_path, device):
    batch = int(os.environ.get("SUPCON_EXTRACT_BATCH_SIZE", "128"))
    workers = config.num_workers()
    tf = get_transforms(224, augment=False)

    id_df = pd.read_csv(config.ID_IMAGES_CSV)
    ood_df = pd.read_csv(config.OOD_IMAGES_CSV)
    gal_items = _gallery_items_id_only()

    model, _ = _load_metric(ckpt_path, backbone="convnext_base", embedding_dim=512, device=device)
    id_loader = DataLoader(CSVImageDataset(id_df, tf), batch_size=batch, shuffle=False,
                           num_workers=workers, pin_memory=(device == "cuda"))
    ood_loader = DataLoader(CSVImageDataset(ood_df, tf), batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=(device == "cuda"))
    gal_loader = DataLoader(ManifestDataset(gal_items, tf), batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=(device == "cuda"))
    id_e, _ = extract_embeddings(model, id_loader, device)
    ood_e, _ = extract_embeddings(model, ood_loader, device)
    gal_e, _ = extract_embeddings(model, gal_loader, device)
    return id_e, ood_e, gal_e


def run(run_root, out_dir):
    """Train/evaluate SupCon seeds and write native_experiments outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("FORCE_SUPCON_SEED_SENSITIVITY", "0") == "1"
    rows_csv = out_dir / "supcon_seed_sensitivity.csv"
    summary_csv = out_dir / "supcon_seed_sensitivity_summary.csv"
    json_path = out_dir / "supcon_seed_sensitivity.json"
    if rows_csv.exists() and summary_csv.exists() and json_path.exists() and not force:
        print(f"  ✅ SupCon seed sensitivity exists: {summary_csv}")
        return json.load(open(json_path))

    device = config.resolve_device()
    manifest = load_swi_manifest()
    seeds = _csv_env("SUPCON_TRAIN_SEEDS", "42,43,44", int)
    use_existing_seed42 = os.environ.get("SUPCON_SEED42_USE_EXISTING", "1") == "1"
    rows = []

    for seed in seeds:
        if seed == 42 and use_existing_seed42 and (config.CKPT_DIR / "metric_convnext_base_supcon.pt").exists():
            ckpt_path = config.CKPT_DIR / "metric_convnext_base_supcon.pt"
            trained_name = "metric_convnext_base_supcon.pt"
        else:
            cfg = train_config.Config(backbone="convnext_base", loss="supcon", seed=int(seed))
            ckpt_name = f"metric_convnext_base_supcon_seed{int(seed)}.pt"
            ckpt_path = train_metric_model(cfg, manifest, device, ckpt_name=ckpt_name,
                                           force=os.environ.get("SUPCON_FORCE_RETRAIN", "0") == "1")
            trained_name = ckpt_name
        id_e, ood_e, gal_e = _extract_for_checkpoint(ckpt_path, device)
        sid = _distance_scores(id_e, gal_e)
        # Use the same OOD species-level test mask as the registry/extended suite.
        from .registry import ood_test_mask
        labels_ood = pd.read_csv(config.OOD_IMAGES_CSV)["label"].astype(str).str.replace(" ", "_").str.lower().str.strip().to_numpy()
        test_mask = ood_test_mask(labels_ood)
        sod = _distance_scores(ood_e, gal_e)[test_mask]
        rows.append({
            "method": "SupCon",
            "seed": int(seed),
            "checkpoint": trained_name,
            "gallery_scope": "id_only",
            "AUROC": _auroc(sid, sod),
            "FPR95": _fpr95(sid, sod),
            "AUPR": _aupr(sid, sod),
            "id_images": int(len(sid)),
            "ood_test_images": int(len(sod)),
            "gallery_images": int(len(gal_e)),
        })
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    summary = []
    for metric in ["AUROC", "FPR95", "AUPR"]:
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        summary.append({
            "method": "SupCon",
            "metric": metric,
            "mean": float(vals.mean()) if len(vals) else None,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else None,
            "min": float(vals.min()) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
            "n_seeds": int(len(vals)),
        })
    df.to_csv(rows_csv, index=False)
    pd.DataFrame(summary).to_csv(summary_csv, index=False)
    payload = {"protocol": {"gallery_scope": "id_only", "seeds": seeds,
                            "seed42_existing_checkpoint": bool(use_existing_seed42)},
               "rows": rows, "summary": summary}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"saved {rows_csv}")
    print(f"saved {summary_csv}")
    print(f"saved {json_path}")
    return payload

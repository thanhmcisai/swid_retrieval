# -*- coding: utf-8 -*-
"""Create a full-954 cache by replacing only CE-Full embeddings/logits.

This is the intended path after retraining ``ce_954sp_convnext_base.pt`` while
all non-CE embeddings already exist in a corrected full-954 cache. It avoids
rebuilding ArcFace/DINOv2/CLIP/SupCon features and therefore does not require
missing non-CE checkpoints.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import config
from ..data import (CSVImageDataset, ManifestDataset, canonical_label,
                    ce_train_image_paths, full_swi_items, get_transforms,
                    load_swi_manifest)
from ..models import CEClassifier
from .extract import extract_embeddings, extract_embeddings_raw


CE_KEYS = (
    "embs_swi_ce_full_norm",
    "logits_swi_ce_full",
    "embs_swi_ce_full_raw",
    "embs_id_ce_full_norm",
    "logits_id_ce_full",
    "embs_id_ce_full_raw",
    "embs_ood_ce_full_norm",
    "logits_ood_ce_full",
    "embs_ood_ce_full_raw",
)


def _source_cache_path():
    name = os.environ.get(
        "CE_UPDATE_SOURCE_CACHE_NAME",
        os.environ.get("SOURCE_FULL954_CACHE_NAME", "embedding_cache_full954_v4_corrected_public.npz"),
    )
    return config.ROOT_PATH / name


def _load_ce(device):
    ckpt = torch.load(config.CKPT_CE_FULL, map_location=device)
    n_classes = ckpt["metrics"]["n_classes"]
    model = CEClassifier("convnext_base", n_classes=n_classes, embedding_dim=512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def _loader(ds):
    batch_size = int(os.environ.get("CE_CACHE_BATCH_SIZE", "64"))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


def _extract_ce_for_dataset(model, ds, device, side):
    loader = _loader(ds)
    norm, labels, logits = extract_embeddings(model, loader, device, return_logits=True)
    raw = extract_embeddings_raw(model, loader, device)
    return {
        f"embs_{side}_ce_full_norm": norm,
        f"logits_{side}_ce_full": logits,
        f"embs_{side}_ce_full_raw": raw,
    }, labels


def _load_meta(source):
    meta_path = source.with_name(source.name.replace(".npz", "_meta.json"))
    if meta_path.exists():
        try:
            return json.load(open(meta_path))
        except Exception:
            return {}
    return {}


def run():
    device = config.resolve_device()
    source = _source_cache_path()
    target = config.FULL954_CACHE_PATH
    if not source.exists():
        raise FileNotFoundError(f"Missing CE update source cache: {source}")
    if not config.CKPT_CE_FULL.exists():
        raise FileNotFoundError(f"Missing CE-Full checkpoint: {config.CKPT_CE_FULL}")
    if target.exists() and not config.FORCE_REBUILD_FULL954:
        print(f"✅ CE-updated cache already exists: {target}")
        return target

    print(f"CE-only cache update source: {source}")
    print(f"CE-only cache update target: {target}")
    src = np.load(source, allow_pickle=False)
    out = {k: src[k] for k in src.files}

    manifest = load_swi_manifest()
    swi_items = full_swi_items(manifest)
    swi_ds = ManifestDataset(swi_items, transform=get_transforms(224, augment=False))

    model, ckpt = _load_ce(device)
    print(f"Loaded CE-Full checkpoint: {config.CKPT_CE_FULL.name} "
          f"(epoch={ckpt.get('epoch')}, val_acc={ckpt.get('metrics', {}).get('val_acc')})")

    repl, _ = _extract_ce_for_dataset(model, swi_ds, device, "swi")
    out.update(repl)
    n_swi = len(swi_ds)

    for side, csv_path in (("id", config.ID_IMAGES_CSV), ("ood", config.OOD_IMAGES_CSV)):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing corrected public CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        ds = CSVImageDataset(df, transform=get_transforms(224, augment=False))
        repl, labels = _extract_ce_for_dataset(model, ds, device, side)
        out.update(repl)
        label_key = f"labels_{side}_dinov2"
        if label_key in out and len(out[label_key]) != len(labels):
            raise ValueError(f"{label_key} has {len(out[label_key])} rows, CE extracted {len(labels)}")
        print(f"✅ Replaced CE-Full {side.upper()}: {len(labels)} rows")

    # Recompute the CE train mask from the current manifest; it should be
    # identical in length to the source cache labels and independent of CE weights.
    ce_seed = int(os.environ.get("CE_TRAIN_SEED", "42"))
    ce_train_set = ce_train_image_paths(manifest, seed=ce_seed)
    swi_paths = [p for p, _ in swi_items]
    out["swi_in_ce_train"] = np.array([p in ce_train_set for p in swi_paths], dtype=bool)

    for key in CE_KEYS:
        if key not in out:
            raise KeyError(f"Missing CE key after update: {key}")
    if "labels_swi_dinov2" in out and len(out["labels_swi_dinov2"]) != n_swi:
        raise ValueError(f"labels_swi_dinov2 length {len(out['labels_swi_dinov2'])} != {n_swi}")
    if len(out["swi_in_ce_train"]) != n_swi:
        raise ValueError(f"swi_in_ce_train length {len(out['swi_in_ce_train'])} != {n_swi}")

    np.savez_compressed(target, **out)

    meta = _load_meta(source)
    meta.update({
        "artifact": "embedding_cache_full954_ce_updated",
        "source_cache": str(source),
        "ce_full_checkpoint": str(config.CKPT_CE_FULL),
        "ce_full_checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "ce_full_checkpoint_val_acc": ckpt.get("metrics", {}).get("val_acc"),
        "ce_full_keys_recomputed": sorted(CE_KEYS),
        "public_queries_reextracted_for_ce_full": True,
        "note": "All non-CE keys copied from source corrected full-954 cache; CE-Full SWI/ID/OOD embeddings and logits recomputed from the active CE checkpoint.",
    })
    with open(config.FULL954_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"✅ Wrote CE-updated full-954 cache: {target}")
    print(f"✅ Wrote CE-updated cache meta: {config.FULL954_META_PATH}")
    return target


if __name__ == "__main__":
    run()

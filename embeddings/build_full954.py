# -*- coding: utf-8 -*-
"""Build the full-954 SmartWoodID gallery embedding cache.

THE unblocking step. embedding_cache_v3.npz holds SWI embeddings for only the
~317-species meta-test split, so the corrected `full_swi` protocol crashes the
`>= 900 species` assertion. This module re-extracts the SWI-gallery side over all
954 species (meta-train ∪ meta-val ∪ meta-test) for every backbone, COPIES the
unchanged public-ID / public-OOD query embeddings from the old cache, and writes
embedding_cache_full954_v3.npz (+ provenance sidecar). The old cache and all
paper_reframe outputs are left untouched.

Idempotent: skips if a valid full-954 cache already exists (FORCE_REBUILD_FULL954=1
to override). Resumable: with SAVE_PARTIAL=1 each backbone's SWI embeddings are
checkpointed to a scratch .npz so a Colab disconnect can resume.
"""

import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .. import config
from ..data import (ManifestDataset, get_transforms, canonical_label,
                    load_swi_manifest, full_swi_items, ce_train_image_paths,
                    preload_image_cache, collect_all_image_paths)
from ..models import build_models
from . import cache_schema
from .extract import (extract_embeddings, extract_embeddings_raw,
                      extract_clip_embeddings, extract_dinov2_embeddings)


def _cache_is_valid(cache_path, meta_path, min_species):
    if not (Path(cache_path).exists() and Path(meta_path).exists()):
        return False
    try:
        meta = json.load(open(meta_path))
        return int(meta.get("swi_full_species", 0)) >= min_species
    except Exception:
        return False


def _load_partial(partial_path):
    if config.SAVE_PARTIAL and Path(partial_path).exists():
        try:
            p = np.load(partial_path, allow_pickle=False)
            data = {k: p[k] for k in p.files}
            print(f"↻ Resuming from partial cache: {len(data)} SWI keys present")
            return data
        except Exception as e:
            print(f"⚠️  Could not read partial cache ({e}); starting fresh")
    return {}


def _save_partial(swi_data, partial_path):
    if config.SAVE_PARTIAL:
        np.savez(partial_path, **swi_data)


def build(device=None, force=None):
    device = device or config.resolve_device()
    force = config.FORCE_REBUILD_FULL954 if force is None else force

    if not force and _cache_is_valid(config.FULL954_CACHE_PATH, config.FULL954_META_PATH,
                                     config.MIN_FULL_GALLERY_SPECIES):
        print(f"✅ Full-954 cache already valid: {config.FULL954_CACHE_PATH} (skip; FORCE_REBUILD_FULL954=1 to rebuild)")
        return config.FULL954_CACHE_PATH

    if not config.SOURCE_EMB_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Source meta-test cache not found: {config.SOURCE_EMB_CACHE_PATH}. "
            "It supplies the unchanged public-ID/OOD query embeddings.")

    # ── Datasets: the only re-extraction is the 954-species SWI union ───────
    manifest = load_swi_manifest()
    items = full_swi_items(manifest)
    n_total_species = len({it[1] for it in items})
    print(f"\nFull SWI pool: {len(items)} images, {n_total_species} species")
    if n_total_species < config.MIN_FULL_GALLERY_SPECIES:
        raise RuntimeError(
            f"Manifest yields only {n_total_species} species (< {config.MIN_FULL_GALLERY_SPECIES}). "
            "Check swi_manifest.json covers all three splits.")

    swi_ds = ManifestDataset(items, transform=get_transforms(224, augment=False))

    # Pre-warm the local image cache in parallel (Drive is slow per-file). When
    # PRELOAD_ALL_IMAGES=1, include public ID/OOD image CSVs as well as SWI; this
    # helps later saliency/timing/cost steps avoid lazy Drive reads. If the
    # orchestrator already did the global prewarm, skip this local prewarm.
    if (
        os.environ.get("PRELOAD_IMAGE_CACHE", "1") == "1"
        and os.environ.get("SWID_IMAGE_PRELOAD_DONE", "0") != "1"
    ):
        if os.environ.get("PRELOAD_ALL_IMAGES", "0") == "1":
            preload_paths = collect_all_image_paths(config.ROOT_PATH, include_swi=True, include_public=True)
            preload_desc = "Preloading SWI + public ID/OOD images"
        else:
            preload_paths = [s[0] for s in swi_ds.samples]
            preload_desc = "Preloading SWI gallery images"
        preload_image_cache(preload_paths,
                            max_workers=int(os.environ.get("PRELOAD_WORKERS", "16")),
                            desc=preload_desc)

    swi_loader = DataLoader(swi_ds, batch_size=64, num_workers=config.num_workers(),
                            pin_memory=True, persistent_workers=True, prefetch_factor=4)
    swi_items = [(p, canonical_label(swi_ds.idx_to_class[l])) for p, l in swi_ds.samples]

    swi_data = _load_partial(config.PARTIAL_CACHE_PATH)
    have = lambda *ks: all(k in swi_data for k in ks)

    print("\n▶ Loading models (no training; checkpoints from Drive)...")
    models, tfs = build_models(device=device, need_clip=True, need_dinov2=True)

    # ── Re-extract each SWI backbone over the 954 union (resumable) ─────────
    if not have("embs_swi_imagenet"):
        swi_data["embs_swi_imagenet"], _ = extract_embeddings(models["imagenet"], swi_loader, device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)
    if not have("embs_swi_arc"):
        swi_data["embs_swi_arc"], _ = extract_embeddings(models["arc"], swi_loader, device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)
    if not have("embs_swi_arc954"):
        swi_data["embs_swi_arc954"], _ = extract_embeddings(models["arc954"], swi_loader, device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)
    if not have("embs_swi_proto"):
        swi_data["embs_swi_proto"], _ = extract_embeddings(models["proto"], swi_loader, device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)

    if not have("embs_swi_ce_full_norm", "logits_swi_ce_full", "embs_swi_ce_full_raw"):
        norm, _, logits = extract_embeddings(models["ce"], swi_loader, device, return_logits=True)
        swi_data["embs_swi_ce_full_norm"] = norm
        swi_data["logits_swi_ce_full"] = logits
        swi_data["embs_swi_ce_full_raw"] = extract_embeddings_raw(models["ce"], swi_loader, device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)
    if not have("embs_swi_ce_narrow_norm", "logits_swi_ce_narrow", "embs_swi_ce_narrow_raw"):
        norm, _, logits = extract_embeddings(models["ce_narrow"], swi_loader, device, return_logits=True)
        swi_data["embs_swi_ce_narrow_norm"] = norm
        swi_data["logits_swi_ce_narrow"] = logits
        swi_data["embs_swi_ce_narrow_raw"] = extract_embeddings_raw(models["ce_narrow"], swi_loader, device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)

    for tag in cache_schema.VARIANT_TAGS:
        key = f"embs_swi_{tag}"
        if not have(key):
            swi_data[key], _ = extract_embeddings(models[tag], swi_loader, device)
            _save_partial(swi_data, config.PARTIAL_CACHE_PATH)

    if not have("embs_swi_clip", "labels_swi_clip"):
        swi_data["embs_swi_clip"], swi_data["labels_swi_clip"] = extract_clip_embeddings(
            models["clip"], swi_items, tfs["clip"], device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)
    if not have("embs_swi_dinov2", "labels_swi_dinov2"):
        swi_data["embs_swi_dinov2"], swi_data["labels_swi_dinov2"] = extract_dinov2_embeddings(
            models["dinov2"], swi_items, tfs["dinov2"], device)
        _save_partial(swi_data, config.PARTIAL_CACHE_PATH)

    # Sanity: every SWI row count must match the dinov2 (gallery-label) row count.
    n_rows = len(swi_data["labels_swi_dinov2"])
    for k in cache_schema.SWI_KEYS:
        if k not in swi_data:
            raise KeyError(f"SWI key not extracted: {k}")
        if len(swi_data[k]) != n_rows:
            raise ValueError(f"Row misalignment: {k} has {len(swi_data[k])} rows, expected {n_rows}.")
    n_species = len(set(canonical_label(x) for x in swi_data["labels_swi_dinov2"]))
    print(f"\nSWI gallery (954): {n_rows} images, {n_species} species")

    # ── CE-train robustness mask (computed here, no second extraction) ──────
    # Tag which gallery rows are in CE-Full's exact training partition, so a
    # fairness robustness pass can restrict the gallery to only the images
    # CE-Full learned from (GALLERY_SCOPE=ce_train), aligned row-for-row.
    ce_seed = int(os.environ.get("CE_TRAIN_SEED", "42"))
    ce_train_set = ce_train_image_paths(manifest, seed=ce_seed)
    swi_paths_in_order = [s[0] for s in swi_ds.samples]
    swi_in_ce_train = np.array([p in ce_train_set for p in swi_paths_in_order], dtype=bool)
    if len(swi_in_ce_train) != n_rows:
        raise ValueError(f"ce_train mask length {len(swi_in_ce_train)} != gallery rows {n_rows}.")
    labels_canon = np.array([canonical_label(x) for x in swi_data["labels_swi_dinov2"]])
    n_ce_species = len(set(labels_canon[swi_in_ce_train].tolist()))
    print(f"CE-train robustness mask: {int(swi_in_ce_train.sum())}/{n_rows} gallery images "
          f"({n_ce_species} species) are in CE-Full's train partition (seed={ce_seed})")

    # ── Copy unchanged ID/OOD query embeddings from the source cache ────────
    src = np.load(config.SOURCE_EMB_CACHE_PATH, allow_pickle=False)
    copied = {k: src[k] for k in src.files if cache_schema.is_copy_key(k)}
    if not copied:
        raise KeyError(f"No id/ood query keys found in {config.SOURCE_EMB_CACHE_PATH}.")
    print(f"Copied {len(copied)} unchanged ID/OOD query keys from {config.SOURCE_EMB_CACHE_NAME}")

    # ── Hygiene: public-OOD species must be disjoint from EVERY SWI split ────
    # (Otherwise a "novel" OOD species was actually a metric-training species.)
    if "labels_ood_dinov2" in src.files:
        ood_species = {canonical_label(x) for x in src["labels_ood_dinov2"]}
        for split in ("meta-train", "meta-val", "meta-test"):
            split_species = {canonical_label(it[1]) for it in manifest[split]}
            overlap = ood_species & split_species
            if overlap:
                raise RuntimeError(
                    f"OOD leakage: {len(overlap)} public-OOD species also appear in "
                    f"SWI {split} (e.g. {sorted(overlap)[:5]}). OOD must be disjoint "
                    f"from all SWI splits."
                )
        print(f"✅ OOD disjointness verified: {len(ood_species)} OOD species disjoint from all 3 SWI splits")

    # ── Write the full-954 cache + provenance ───────────────────────────────
    out = {**{k: swi_data[k] for k in cache_schema.SWI_KEYS},
           "swi_in_ce_train": swi_in_ce_train, **copied}
    np.savez_compressed(config.FULL954_CACHE_PATH, **out)
    meta = cache_schema.build_meta(n_rows, n_species, list(copied.keys()),
                                   cache_schema.SWI_KEYS, config.SOURCE_EMB_CACHE_NAME)
    meta["ce_train_robustness"] = {
        "swi_in_ce_train_key": "swi_in_ce_train",
        "ce_train_images": int(swi_in_ce_train.sum()),
        "ce_train_species": int(n_ce_species),
        "ce_train_seed": ce_seed,
        "note": "Boolean mask over the 954-row SWI gallery marking CE-Full's exact "
                "train partition; use GALLERY_SCOPE=ce_train for the fairness control.",
    }
    with open(config.FULL954_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"✅ Wrote {config.FULL954_CACHE_PATH}  ({len(out)} keys)")
    print(f"✅ Wrote {config.FULL954_META_PATH}")

    # Optionally drop the partial scratch now that the full cache is complete.
    if config.SAVE_PARTIAL and Path(config.PARTIAL_CACHE_PATH).exists():
        try:
            Path(config.PARTIAL_CACHE_PATH).unlink()
        except Exception:
            pass
    return config.FULL954_CACHE_PATH


def project_scurd_full_pool(device=None):
    """Project the full-954 SWI DINOv2 gallery through SC-URD and store as
    `swi_pool` in the projected cache, so SC-URD stays aligned with the 954
    gallery in both the variance engine and the migrated review taxonomy.

    Mirrors variance_retrieval_evidence_colab.py:446-463 (which projects on the
    fly); persisting it once avoids re-projecting and prevents a stale
    meta-test-length swi_pool from tripping the alignment guard.
    """
    device = device or config.resolve_device()
    proj = config.SCURD_PROJ_CACHE
    if not config.SCURD_MAIN_CKPT.exists():
        print(f"⚠️  SC-URD checkpoint missing ({config.SCURD_MAIN_CKPT}); skipping swi_pool projection.")
        return None
    if not config.FULL954_CACHE_PATH.exists():
        raise FileNotFoundError("Full-954 cache must be built before SC-URD pool projection.")

    from ..scurd import load_scurd_model, project_np
    full = np.load(config.FULL954_CACHE_PATH, allow_pickle=False)
    swi_dinov2 = full["embs_swi_dinov2"]

    existing = {}
    if proj.exists():
        sc = np.load(proj, allow_pickle=False)
        existing = {k: sc[k] for k in sc.files}
        if "swi_pool" in existing and len(existing["swi_pool"]) == len(swi_dinov2):
            print(f"✅ SC-URD swi_pool already aligned ({len(swi_dinov2)} rows); skip.")
            return proj

    model, _ = load_scurd_model(config.SCURD_MAIN_CKPT, in_dim=swi_dinov2.shape[1], device=device)
    print(f"Projecting full-954 SWI DINOv2 through SC-URD ({len(swi_dinov2)} rows)...")
    existing["swi_pool"] = project_np(model, swi_dinov2, device)
    proj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(proj, **existing)
    print(f"✅ Saved SC-URD swi_pool into {proj}")
    return proj


def main():
    build()
    # SC-URD full-gallery embeddings: by default we do NOT persist `swi_pool`,
    # because the projected cache lives under the original paper_reframe dir and
    # we keep that untouched. The variance and review scripts already project
    # SC-URD from its checkpoint over the full-954 DINOv2 gallery on the fly
    # (an aligned, correct fallback). Set SCURD_PERSIST_SWI_POOL=1 to cache it.
    import os
    if os.environ.get("SCURD_PERSIST_SWI_POOL", "0") == "1":
        project_scurd_full_pool()
    else:
        print("ℹ️  Skipping swi_pool persistence (SCURD_PERSIST_SWI_POOL=0); "
              "SC-URD will project the full gallery from its checkpoint at eval time.")


if __name__ == "__main__":
    main()

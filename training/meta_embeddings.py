# -*- coding: utf-8 -*-
"""DINOv2 meta-embedding extraction for the research suite (URD-v2 / SC-URD).

Lifted from smartwoodid_experiments_full.py:5577-5636. Produces the frozen-DINOv2
weak/strong embedding cache the research heads train on:

  RESEARCH_DIR/urd_v2_meta_dinov2_embeddings_v2.npz
    keys: train_weak, train_strong, train_labels, val_weak, val_labels

The OOD DINOv2 embeddings + top-50 OOD species (needed by SC-URD Phase 2) are read
from the already-built embedding cache (embs_ood_dinov2 / labels_ood_dinov2 keys),
not re-extracted — they are external query images independent of the SWI gallery.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from ..data import canonical_label
from .. import config as C


class _DINOItemDataset(Dataset):
    def __init__(self, items, transform):
        self.items = [(p, canonical_label(sp)) for p, sp in items]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def dino_strong_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(518, scale=(0.45, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.35),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def extract_dino_dataset(model_dinov2, items, transform, device, desc="DINOv2",
                         batch_size=48, num_workers=None):
    num_workers = C.num_workers() if num_workers is None else num_workers
    ds = _DINOItemDataset(items, transform)
    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(ds, **kwargs)
    all_emb, all_lbl = [], []
    model_dinov2.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=desc, leave=False):
            emb = model_dinov2(imgs.to(device)).float()
            emb = F.normalize(emb, dim=1).cpu().numpy()
            all_emb.append(emb)
            all_lbl.extend([canonical_label(x) for x in lbls])
    return np.concatenate(all_emb, axis=0), np.array(all_lbl)


def _extract_backbone_224(model, items, augment, device, desc="backbone"):
    """Embed `items` (in order) through a 224px ConvNeXt/ViT EmbeddingModel using the
    package's albumentations transforms; returns embeddings aligned to items order."""
    from ..data import ManifestDataset, get_transforms
    from torch.utils.data import DataLoader
    from .common import extract_embeddings
    nw = C.num_workers()
    ds = ManifestDataset(items, transform=get_transforms(224, augment=augment))
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    embs, _ = extract_embeddings(model, loader, device)  # order preserved (shuffle=False)
    return embs


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _dino_halves(dino_cache, cache_path, train_labels, models, transforms_out,
                 train_items, val_items, device):
    """DINOv2 weak/strong(train) + weak(val) for fusion — REUSE the existing DINOv2 meta
    cache when present (same manifest order → rows align), else extract via DINOv2."""
    dc = Path(dino_cache) if dino_cache else (Path(cache_path).parent / "urd_v2_meta_dinov2_embeddings_v2.npz")
    if dc.exists():
        d = np.load(dc, allow_pickle=False)
        if {"train_weak", "train_strong", "val_weak", "train_labels"}.issubset(set(d.files)) and \
           list(map(str, d["train_labels"])) == list(map(str, train_labels)):
            print(f"  ♻️  Reusing DINOv2 meta halves from {dc.name} (no DINOv2 image pass)")
            return d["train_weak"], d["train_strong"], d["val_weak"]
        print(f"  ⚠️ DINOv2 meta cache {dc.name} misaligned/incomplete; extracting DINOv2 fresh.")
    if "dinov2" not in models:
        raise FileNotFoundError(
            f"fusion base needs DINOv2 meta halves: {dc} not found and no DINOv2 model passed. "
            "Build urd_v2_meta_dinov2_embeddings_v2.npz first (it is the main SC-URD meta cache).")
    dino, dtf = models["dinov2"], transforms_out["dinov2"]
    dw, _ = extract_dino_dataset(dino, train_items, dtf, device, "dino weak")
    ds_, _ = extract_dino_dataset(dino, train_items, dino_strong_transform(), device, "dino strong")
    dvw, _ = extract_dino_dataset(dino, val_items, dtf, device, "dino val")
    return dw, ds_, dvw


def extract_base_meta(base, cache_path, manifest, models, transforms_out, device,
                      force=False, dino_cache=None):
    """Weak/strong meta-embeddings for an ALTERNATE SC-URD base (the backbone ablation).

    base ∈ {'arcface557' (ConvNeXt ArcFace-557, 512-d), 'fusion' (ArcFace⊕DINOv2, 1280-d)}.
    Mirrors the DINOv2 weak(eval)/strong(aug) scheme so train_sc_urd(meta, …) works
    unchanged (it derives in_dim from the cache). Saves scurd_meta_<base>_v2.npz.

    Speed: (1) the SWI meta images are pre-warmed Drive→local once (16 threads) so the
    224px ArcFace passes read from the local disk cache; (2) for 'fusion', the DINOv2
    half is REUSED from the already-built DINOv2 meta cache (dino_cache) instead of
    re-running DINOv2 over the images — eliminating the two slow 518px passes. The
    saved npz is the resume point (next run loads it, no re-extraction).
    """
    from ..data import preload_image_cache
    cache_path = Path(cache_path)
    required = {"train_weak", "train_strong", "train_labels", "val_weak", "val_labels"}
    if cache_path.exists() and not force:
        c = np.load(cache_path, allow_pickle=False)
        if required.issubset(set(c.files)):
            print(f"  ✅ Loaded {base} meta cache: {cache_path.name}")
            return {k: c[k] for k in c.files}
    train_items = [(p, canonical_label(sp)) for p, sp in manifest["meta-train"]]
    val_items = [(p, canonical_label(sp)) for p, sp in manifest["meta-val"]]
    train_labels = np.array([sp for _, sp in train_items])
    val_labels = np.array([sp for _, sp in val_items])

    # Pre-warm the local image cache once (Drive → /content) for all meta images.
    preload_image_cache([p for p, _ in train_items] + [p for p, _ in val_items],
                        desc=f"Pre-cache meta images ({base})")

    arc = models["arc"]
    aw = _extract_backbone_224(arc, train_items, False, device, "arc weak")
    as_ = _extract_backbone_224(arc, train_items, True, device, "arc strong")
    avw = _extract_backbone_224(arc, val_items, False, device, "arc val")
    if base == "arcface557":
        tw, ts, vw = aw, as_, avw
    elif base == "fusion":
        dw, ds_, dvw = _dino_halves(dino_cache, cache_path, train_labels, models, transforms_out,
                                    train_items, val_items, device)
        tw = _l2(np.concatenate([_l2(aw), _l2(dw)], axis=1))
        ts = _l2(np.concatenate([_l2(as_), _l2(ds_)], axis=1))
        vw = _l2(np.concatenate([_l2(avw), _l2(dvw)], axis=1))
    else:
        raise ValueError(f"unknown base {base}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, train_weak=tw, train_strong=ts, train_labels=train_labels,
                        val_weak=vw, val_labels=val_labels)
    print(f"  ✅ Saved {base} meta cache ({tw.shape}) → {cache_path.name}")
    return {"train_weak": tw, "train_strong": ts, "train_labels": train_labels,
            "val_weak": vw, "val_labels": val_labels}


def load_or_extract_meta_embeddings(cache_path, manifest, model_dinov2, dinov2_transform,
                                    device, force=False):
    """Build (or load) the weak/strong DINOv2 meta-train + weak meta-val cache."""
    cache_path = Path(cache_path)
    required = {"train_weak", "train_strong", "train_labels", "val_weak", "val_labels"}
    if cache_path.exists() and not force:
        try:
            c = np.load(cache_path, allow_pickle=False)
            if required.issubset(set(c.files)):
                print(f"  ✅ Loaded URD meta embedding cache: {cache_path}")
                return {k: c[k] for k in c.files}
            print(f"  ⚠️ meta cache missing keys {sorted(required - set(c.files))}; rebuilding.")
        except Exception as e:
            print(f"  ⚠️ meta cache stale ({e}); rebuilding.")
    print(f"  Extracting URD weak/strong DINOv2 meta embeddings → {cache_path}")
    train_items = [(p, canonical_label(sp)) for p, sp in manifest["meta-train"]]
    val_items = [(p, canonical_label(sp)) for p, sp in manifest["meta-val"]]
    train_weak, train_labels = extract_dino_dataset(
        model_dinov2, train_items, dinov2_transform, device, "URD train weak")
    train_strong, train_labels_s = extract_dino_dataset(
        model_dinov2, train_items, dino_strong_transform(), device, "URD train strong")
    val_weak, val_labels = extract_dino_dataset(
        model_dinov2, val_items, dinov2_transform, device, "URD val weak")
    assert np.array_equal(train_labels, train_labels_s)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, train_weak=train_weak, train_strong=train_strong,
                        train_labels=train_labels, val_weak=val_weak, val_labels=val_labels)
    print(f"  ✅ Saved meta embedding cache: {cache_path}")
    return {"train_weak": train_weak, "train_strong": train_strong, "train_labels": train_labels,
            "val_weak": val_weak, "val_labels": val_labels}


def load_ood_dinov2(cache_path=None):
    """Load raw DINOv2 OOD embeddings + canonical labels from the embedding cache.

    These are the external public-OOD query images (key embs_ood_dinov2 /
    labels_ood_dinov2), copied verbatim into the full-954 cache. SC-URD Phase 2
    trains on the non-top50 subset of these to improve open-set enrollment.
    """
    for p in [cache_path, C.FULL954_CACHE_PATH, C.SOURCE_EMB_CACHE_PATH]:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        c = np.load(p, allow_pickle=False)
        if "embs_ood_dinov2" in c.files and "labels_ood_dinov2" in c.files:
            labels = np.array([canonical_label(x) for x in c["labels_ood_dinov2"]])
            print(f"  ✅ Loaded OOD DINOv2 embeddings from {p.name}: "
                  f"{c['embs_ood_dinov2'].shape[0]} imgs, {len(set(labels.tolist()))} species")
            return np.asarray(c["embs_ood_dinov2"], dtype=np.float32), labels
    raise FileNotFoundError(
        "No cache with embs_ood_dinov2/labels_ood_dinov2 found (looked at FULL954 + source). "
        "Build the embedding cache first (swid_retrieval.embeddings.build_full954).")


def select_top50_ood(ood_labels, n=50):
    """Top-N OOD species by image count (matches monolith ood_counts.head(50))."""
    labels = np.asarray(ood_labels)
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    top = [str(uniq[i]) for i in order[:n]]
    print(f"  Top-{n} OOD species selected (min count {int(counts[order[min(n, len(order))-1]])})")
    return top

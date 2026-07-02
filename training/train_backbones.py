# -*- coding: utf-8 -*-
"""Image-backbone training (GPU-heavy) — lifted from smartwoodid_experiments_full.py.

Reproduces the validated checkpoints the eval loads:
  - metric_convnext_base_arcface.pt        (ArcFace-557, meta-train)        L612
  - metric_convnext_base_protonet_v2.pt    (ProtoNet-557, meta-train)       L612
  - ce_954sp_convnext_base.pt              (CE-Full, 954sp, 65/15/20)       L700
  - ce_narrow_557sp_convnext_base.pt       (CE-Narrow, 557sp, 80/20)        L1043
  - arcface_full954.pt                     (ArcFace-954, 95/5 per-species)  L1204
  - metric_<bb>_<loss>.pt   × 3 variants   (backbone × loss ablation)       L1317

All functions are SKIP-IF-EXISTS (return the path when the .pt is already present
and marked complete) so a reproduction run never clobbers the validated weights.
Pass force=True (FORCE_RETRAIN=1) to retrain and overwrite — only do this when
CKPT_DIR points at a throwaway directory.

Backbones are built `pretrained=True` HERE (ImageNet init for training); the eval
loaders in ..models build them `pretrained=False` and load these state dicts.
"""

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..data import ManifestDataset, CachedImageLoader, get_transforms
from ..models import EmbeddingModel, CEClassifier
from .common import set_seed, save_checkpoint, compute_recall_at_k, extract_embeddings
from .losses import build_loss
from .samplers import PKSampler


def _build_embedding_model(backbone, embedding_dim, pretrained, device):
    return EmbeddingModel(backbone, embedding_dim, pretrained=pretrained).to(device)


def _is_complete(ckpt, total_epochs):
    """Replicates the monolith short-circuit: complete, past total, or legacy (no flag)."""
    mets = ckpt.get("metrics", {})
    return (mets.get("training_complete", False)
            or ckpt.get("epoch", 0) >= total_epochs
            or "training_complete" not in mets)


def _stamp_complete(ckpt_path, total_epochs, best_epoch):
    """Bump epoch→total_epochs and set training_complete so future runs short-circuit."""
    fc = torch.load(ckpt_path, map_location="cpu")
    fc["epoch"] = total_epochs
    fc.setdefault("metrics", {})
    fc["metrics"]["best_epoch"] = best_epoch
    fc["metrics"]["total_epochs"] = total_epochs
    fc["metrics"]["training_complete"] = True
    torch.save(fc, ckpt_path)
    return fc


class _CEDataset(Dataset):
    """CE dataset over [(path, species)] with a fixed species→index map."""

    def __init__(self, items, sp_to_idx, transform=None):
        self.items = items
        self.class_to_idx = sp_to_idx
        self.transform = transform
        self.loader = CachedImageLoader(use_ram_cache=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, species = self.items[i]
        img = self.loader.load(path)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.class_to_idx[species]


# ══════════════════════════════════════════════════════════════════════════
# Metric models — ArcFace / SupCon / ProtoNet on meta-train (L612-697)
# ══════════════════════════════════════════════════════════════════════════
def train_metric_model(cfg, manifest, device, ckpt_name=None, force=False):
    """Train ArcFace / SupCon / ProtoNet on meta-train, val on meta-val."""
    set_seed(cfg.seed)
    ckpt_name = ckpt_name or f"{cfg.run_name}.pt"
    ckpt_path = cfg.checkpoint_path / ckpt_name

    if ckpt_path.exists() and not force:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"✅ {ckpt_name} exists — epoch={ckpt.get('epoch')}, "
              f"R@1={ckpt.get('metrics',{}).get('R@1','?')} → skip")
        return ckpt_path

    print(f"🚀 Training {cfg.loss.upper()} ({cfg.backbone}) | P={cfg.P} K={cfg.K} batch={cfg.P*cfg.K}")
    train_ds = ManifestDataset(manifest["meta-train"],
                               transform=get_transforms(cfg.img_size, augment=True))
    val_ds = ManifestDataset(manifest["meta-val"],
                             transform=get_transforms(cfg.img_size, augment=False))

    n_classes = len(train_ds.class_to_idx)
    sampler = PKSampler(train_ds.get_labels(), P=cfg.P, K=cfg.K)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_val,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    model = _build_embedding_model(cfg.backbone, cfg.embedding_dim, cfg.pretrained, device)
    criterion = build_loss(cfg.loss, n_classes, cfg.embedding_dim, cfg, device)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    opt_groups = [
        {"params": backbone_params, "lr": cfg.backbone_lr},
        {"params": head_params, "lr": cfg.lr},
    ]
    loss_params = list(criterion.parameters())
    if loss_params:
        opt_groups.append({"params": loss_params, "lr": cfg.lr})

    optimizer = torch.optim.AdamW(opt_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    best_r1, best_epoch = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        for p in backbone_params:
            p.requires_grad_(epoch > cfg.warmup_epochs)
        model.train(); criterion.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        val_emb, val_lbl = extract_embeddings(model, val_loader, device)
        metrics = compute_recall_at_k(val_emb, val_emb, val_lbl, val_lbl, ks=[1, 5])
        r1 = metrics["R@1"]
        phase = "warmup" if epoch <= cfg.warmup_epochs else "train"
        print(f"Ep {epoch:3d}/{cfg.epochs} [{phase}] loss={total_loss/n_batches:.4f} "
              f"| val R@1={r1:.4f} R@5={metrics['R@5']:.4f}")

        if r1 > best_r1:
            best_r1, best_epoch = r1, epoch
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_path,
                            config={"backbone": cfg.backbone, "loss": cfg.loss,
                                    "embedding_dim": cfg.embedding_dim,
                                    "best_epoch": epoch,
                                    "total_epochs": cfg.epochs})
            print(f"  ✅ Best (R@1={best_r1:.4f})")

    if ckpt_path.exists():
        _stamp_complete(ckpt_path, cfg.epochs, best_epoch)
    print(f"\n✅ {cfg.loss} training done | best R@1={best_r1:.4f} @ ep {best_epoch} / {cfg.epochs}")
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# CE-Full — 954sp, 65/15/20 stratified by (species, scale)  (L700-855)
# ══════════════════════════════════════════════════════════════════════════
def train_ce_classifier(cfg, manifest, device, ckpt_name="ce_954sp_convnext_base.pt", force=False):
    """Train CE on ALL 954sp, 65/15/20 stratified by species×scale."""
    set_seed(cfg.seed)
    ckpt_path = cfg.checkpoint_path / ckpt_name

    resume_epoch, resume_ckpt = 0, None
    if ckpt_path.exists():
        resume_ckpt = torch.load(ckpt_path, map_location="cpu")
        resume_epoch = resume_ckpt.get("epoch", 0)
        resume_acc = resume_ckpt.get("metrics", {}).get("val_acc", 0)
        n_cls = resume_ckpt.get("metrics", {}).get("n_classes", "?")
        if not force and _is_complete(resume_ckpt, cfg.epochs):
            print(f"✅ {ckpt_name} EXISTS — epoch={resume_epoch}, "
                  f"val_acc={resume_acc:.4f}, n_classes={n_cls} → skip training")
            return ckpt_path
        if force:
            resume_ckpt, resume_epoch = None, 0
        else:
            print(f"⏩ RESUME from epoch {resume_epoch} (val_acc={resume_acc:.4f})")

    # Build data splits (deterministic with seed) — stratified by (species, scale).
    all_items = manifest["meta-train"] + manifest["meta-val"] + manifest["meta-test"]
    rng = np.random.RandomState(cfg.seed)
    groups = defaultdict(list)
    for path, species in all_items:
        groups[(species, Path(path).parent.name)].append((path, species))

    ce_train, ce_val, ce_test = [], [], []
    for _key, items in groups.items():
        idx = list(range(len(items))); rng.shuffle(idx)
        n_tr = max(1, int(0.65 * len(items)))
        n_va = max(0, int(0.15 * len(items)))
        ce_train.extend([items[i] for i in idx[:n_tr]])
        ce_val.extend([items[i] for i in idx[n_tr:n_tr + n_va]])
        ce_test.extend([items[i] for i in idx[n_tr + n_va:]])
    print(f"  Train: {len(ce_train)} | Val: {len(ce_val)} | Test: {len(ce_test)}")

    all_species = sorted(set(it[1] for it in all_items))
    n_classes = len(all_species)
    sp_to_idx = {sp: i for i, sp in enumerate(all_species)}
    print(f"  n_classes: {n_classes} | batch_size={cfg.batch_size_ce}")

    train_ds = _CEDataset(ce_train, sp_to_idx, get_transforms(224, augment=True))
    val_ds = _CEDataset(ce_val, sp_to_idx, get_transforms(224, augment=False))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size_ce, shuffle=True,
                              persistent_workers=True, prefetch_factor=4,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_val, shuffle=False,
                            persistent_workers=True, prefetch_factor=4,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = CEClassifier(cfg.backbone, n_classes=n_classes,
                         embedding_dim=cfg.embedding_dim, pretrained=True).to(device)

    backbone_params = list(model.backbone.parameters())
    other_params = list(model.head.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.backbone_lr},
        {"params": other_params, "lr": 1e-4},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_epoch = 0.0, 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        best_acc = resume_ckpt.get("metrics", {}).get("val_acc", 0)
        best_epoch = resume_epoch
        for _ in range(resume_epoch):
            scheduler.step()
        print(f"  ✅ Loaded model+optimizer from epoch {resume_epoch}, best_acc={best_acc:.4f}")

    start_epoch = resume_epoch + 1
    if start_epoch > cfg.epochs:
        print("  Nothing to train.")
        return ckpt_path
    print(f"  Training epochs {start_epoch} → {cfg.epochs}...")

    for epoch in range(start_epoch, cfg.epochs + 1):
        for p in backbone_params:
            p.requires_grad_(epoch > cfg.warmup_epochs)
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = criterion(model(imgs), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        model.eval()
        correct, total_v = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                correct += (model(imgs.to(device)).cpu().argmax(1) == lbls).sum().item()
                total_v += len(lbls)
        val_acc = correct / total_v
        phase = "warmup" if epoch <= cfg.warmup_epochs else "train"
        print(f"Ep {epoch:3d}/{cfg.epochs} [{phase}] loss={total_loss/n_batches:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            save_checkpoint(model, optimizer, epoch,
                            {"val_acc": val_acc, "n_classes": n_classes,
                             "best_epoch": epoch, "total_epochs": cfg.epochs},
                            ckpt_path, config={"backbone": cfg.backbone},
                            ce_species_list=all_species)

    _stamp_complete(ckpt_path, cfg.epochs, best_epoch)
    print(f"\n✅ CE done | best val_acc={best_acc:.4f} @ ep {best_epoch} / {cfg.epochs}")
    test_path = cfg.checkpoint_path / "ce_test_split.json"
    json.dump({"test": [[p, s] for p, s in ce_test]}, open(test_path, "w"))
    print(f"   Test split saved: {test_path}")
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# CE-Narrow — meta-train (557sp), 80/20 internal split  (L1043-1177)
# ══════════════════════════════════════════════════════════════════════════
def train_ce_narrow(cfg, manifest, device, ckpt_name="ce_narrow_557sp_convnext_base.pt", force=False):
    """Train CE on meta-train (557 species) only; 80/20 stratified by (species, scale)."""
    ckpt_path = cfg.checkpoint_path / ckpt_name

    resume_epoch, resume_ckpt = 0, None
    if ckpt_path.exists():
        resume_ckpt = torch.load(ckpt_path, map_location="cpu")
        resume_epoch = resume_ckpt.get("epoch", 0)
        resume_acc = resume_ckpt.get("metrics", {}).get("val_acc", 0)
        n_cls = resume_ckpt.get("metrics", {}).get("n_classes", "?")
        if not force and _is_complete(resume_ckpt, cfg.epochs):
            print(f"✅ {ckpt_name} EXISTS — epoch={resume_epoch}, "
                  f"val_acc={resume_acc:.4f}, n_classes={n_cls} → skip training")
            return ckpt_path
        if force:
            resume_ckpt, resume_epoch = None, 0
        else:
            print(f"⏩ RESUME from epoch {resume_epoch} (val_acc={resume_acc:.4f})")

    set_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    groups = defaultdict(list)
    for path, species in manifest["meta-train"]:
        groups[(species, Path(path).parent.name)].append((path, species))

    ce_train, ce_val = [], []
    for _key, items in groups.items():
        idx = list(range(len(items))); rng.shuffle(idx)
        n_tr = max(1, int(0.80 * len(items)))
        ce_train.extend([items[i] for i in idx[:n_tr]])
        ce_val.extend([items[i] for i in idx[n_tr:]])
    print(f"  CE-Narrow Train: {len(ce_train)} | Val: {len(ce_val)}")

    all_species = sorted(set(it[1] for it in manifest["meta-train"]))
    n_classes = len(all_species)
    sp_to_idx = {sp: i for i, sp in enumerate(all_species)}
    print(f"  n_classes: {n_classes} | batch_size={cfg.batch_size_ce}")

    train_ds = _CEDataset(ce_train, sp_to_idx, get_transforms(224, augment=True))
    val_ds = _CEDataset(ce_val, sp_to_idx, get_transforms(224, augment=False))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size_ce, shuffle=True,
                              persistent_workers=True, prefetch_factor=4,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_val, shuffle=False,
                            persistent_workers=True, prefetch_factor=4,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = CEClassifier(cfg.backbone, n_classes=n_classes,
                         embedding_dim=cfg.embedding_dim, pretrained=True).to(device)

    backbone_params = list(model.backbone.parameters())
    other_params = list(model.head.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.backbone_lr},
        {"params": other_params, "lr": 1e-4},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_epoch = 0.0, 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        best_acc = resume_ckpt.get("metrics", {}).get("val_acc", 0)
        best_epoch = resume_epoch
        for _ in range(resume_epoch):
            scheduler.step()

    start_epoch = resume_epoch + 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        for p in backbone_params:
            p.requires_grad_(epoch > cfg.warmup_epochs)
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = criterion(model(imgs), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        model.eval()
        correct, total_v = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                correct += (model(imgs.to(device)).cpu().argmax(1) == lbls).sum().item()
                total_v += len(lbls)
        val_acc = correct / total_v
        phase = "warmup" if epoch <= cfg.warmup_epochs else "train"
        print(f"Ep {epoch:3d}/{cfg.epochs} [{phase}] loss={total_loss/n_batches:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            save_checkpoint(model, optimizer, epoch,
                            {"val_acc": val_acc, "n_classes": n_classes,
                             "best_epoch": epoch, "total_epochs": cfg.epochs},
                            ckpt_path, config={"backbone": cfg.backbone},
                            ce_species_list=all_species)

    if ckpt_path.exists():
        _stamp_complete(ckpt_path, cfg.epochs, best_epoch)
    print(f"\n✅ CE-Narrow done | best val_acc={best_acc:.4f} @ ep {best_epoch} / {cfg.epochs}")
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# ArcFace-954 — all 954 species, 95/5 per-species split  (L1204-1299)
# ══════════════════════════════════════════════════════════════════════════
def train_arcface_954(cfg, manifest, device, ckpt_name="arcface_full954.pt", force=False):
    """Train ArcFace on all 954 species; val on a 5% per-species hold-out (R@1 model select)."""
    ckpt_path = cfg.checkpoint_path / ckpt_name

    if ckpt_path.exists() and not force:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        be = ckpt.get("metrics", {}).get("best_epoch", ckpt.get("epoch", "?"))
        r1 = ckpt.get("metrics", {}).get("R@1", "?")
        print(f"✅ {ckpt_name} EXISTS (best ep {be}/{cfg.epochs}, R@1={r1}) → skip")
        return ckpt_path

    all_items = manifest["meta-train"] + manifest["meta-val"] + manifest["meta-test"]
    print(f"\nFull SWI pool: {len(all_items)} images, "
          f"{len({it[1] for it in all_items})} species")

    by_sp = defaultdict(list)
    for it in all_items:
        by_sp[it[1]].append(it)
    rng_split = np.random.default_rng(cfg.seed)
    train_full_items, val_full_items = [], []
    for _sp, lst in by_sp.items():
        idxs = np.arange(len(lst)); rng_split.shuffle(idxs)
        n_val = max(1, int(0.05 * len(lst)))
        val_full_items.extend([lst[i] for i in idxs[:n_val]])
        train_full_items.extend([lst[i] for i in idxs[n_val:]])

    train_full_ds = ManifestDataset(train_full_items,
                                    transform=get_transforms(cfg.img_size, augment=True),
                                    scales=cfg.scales)
    val_full_ds = ManifestDataset(val_full_items,
                                  transform=get_transforms(cfg.img_size, augment=False),
                                  scales=cfg.scales)
    n_classes_full = len(train_full_ds.class_to_idx)

    sampler_full = PKSampler(train_full_ds.get_labels(), P=cfg.P, K=cfg.K)
    train_full_loader = DataLoader(train_full_ds, batch_sampler=sampler_full,
                                   num_workers=cfg.num_workers, persistent_workers=True,
                                   prefetch_factor=4, pin_memory=True)
    val_full_loader = DataLoader(val_full_ds, batch_size=cfg.batch_size_val,
                                 num_workers=cfg.num_workers, persistent_workers=True,
                                 prefetch_factor=4, pin_memory=True)

    print(f"🚀 Training ArcFace on all {n_classes_full} species for {cfg.epochs} epochs")
    set_seed(cfg.seed)
    model = _build_embedding_model(cfg.backbone, cfg.embedding_dim, cfg.pretrained, device)
    criterion = build_loss("arcface", n_classes_full, cfg.embedding_dim, cfg, device)
    bb = list(model.backbone.parameters())
    hd = list(model.head.parameters())
    opt_groups = [{"params": bb, "lr": cfg.backbone_lr},
                  {"params": hd, "lr": cfg.lr}]
    if list(criterion.parameters()):
        opt_groups.append({"params": list(criterion.parameters()), "lr": cfg.lr})
    opt = torch.optim.AdamW(opt_groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    best_r1, best_ep = 0.0, 0
    for ep_ in range(1, cfg.epochs + 1):
        for p in bb:
            p.requires_grad_(ep_ > cfg.warmup_epochs)
        model.train(); criterion.train()
        t_loss, n_b = 0.0, 0
        for imgs, lbls in train_full_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = criterion(model(imgs), lbls)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item(); n_b += 1
        sched.step()
        v_emb, v_lbl = extract_embeddings(model, val_full_loader, device)
        mets = compute_recall_at_k(v_emb, v_emb, v_lbl, v_lbl, ks=[1, 5])
        print(f"Ep {ep_:3d}/{cfg.epochs}  loss={t_loss/n_b:.4f}  "
              f"val R@1={mets['R@1']:.4f}  R@5={mets['R@5']:.4f}")
        if mets["R@1"] > best_r1:
            best_r1, best_ep = mets["R@1"], ep_
            save_checkpoint(model, opt, ep_, mets, ckpt_path,
                            config={"backbone": cfg.backbone, "loss": "arcface",
                                    "embedding_dim": cfg.embedding_dim,
                                    "n_classes": n_classes_full,
                                    "best_epoch": ep_, "total_epochs": cfg.epochs})
    if ckpt_path.exists():
        _stamp_complete(ckpt_path, cfg.epochs, best_ep)
    print(f"✅ ArcFace-954 ready (n_classes={n_classes_full}, best R@1={best_r1:.4f} @ ep {best_ep})")
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# Variants — backbone × loss ablation  (L1303-1340)
# ══════════════════════════════════════════════════════════════════════════
def train_variants(cfg, manifest, device, variants=None, force=False):
    """Train the backbone × loss ablation set (Var_ViTB_Arc, Var_RN50_Arc, Var_CvNxt_SupCon)."""
    from .config import VARIANTS
    variants = variants or VARIANTS
    paths = {}
    for bb, loss, tag in variants:
        vcfg = dataclasses.replace(cfg, backbone=bb, loss=loss)
        ckpt_name = f"metric_{bb}_{loss}.pt"
        paths[tag] = train_metric_model(vcfg, manifest, device, ckpt_name, force=force)
        print(f"✅ {tag} ({bb}/{loss}) ready → {paths[tag].name}")
    print(f"\n✅ {len(paths)} variants ready: {list(paths.keys())}")
    return paths

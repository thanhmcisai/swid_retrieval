# -*- coding: utf-8 -*-
"""Embedding extraction (timm/CE, CLIP, DINOv2).

Lifted from final_metric_learning_cea_2026.py:461-480, 1785-1814. Identical math
so re-extracted embeddings match the validated cache byte-for-byte (modulo GPU
nondeterminism, which cudnn.deterministic already pins).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..data import canonical_label


def extract_embeddings(model, loader, device, return_logits=False):
    """Extract embeddings (+ optional logits for CEClassifier)."""
    model.eval()
    all_emb, all_lbl, all_logits = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Embedding", leave=False):
            imgs = imgs.to(device)
            if return_logits and hasattr(model, "get_embedding"):
                emb = model.get_embedding(imgs, normalize=True).cpu()
                all_logits.append(model(imgs).cpu())
            else:
                emb = model(imgs).cpu()
            all_emb.append(emb)
            all_lbl.extend(lbls.tolist())
    embs = torch.cat(all_emb).numpy()
    labels = np.array(all_lbl)
    if return_logits:
        return embs, labels, torch.cat(all_logits).numpy()
    return embs, labels


def extract_embeddings_raw(model, loader, device):
    """CE penultimate embedding WITHOUT L2 normalization (embs_*_ce_*_raw)."""
    model.eval()
    out = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Embedding(raw)", leave=False):
            out.append(model.get_embedding(imgs.to(device), normalize=False).cpu())
    return torch.cat(out).numpy()


def extract_clip_embeddings(model_clip, items, transform, device, batch_size=64):
    all_emb, all_lbl = [], []
    for i in tqdm(range(0, len(items), batch_size), desc="CLIP", leave=False):
        batch = items[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
        with torch.no_grad():
            emb = F.normalize(model_clip.encode_image(imgs).float(), dim=1).cpu()
        all_emb.append(emb)
        all_lbl.extend([canonical_label(l) for _, l in batch])
    return torch.cat(all_emb).numpy(), np.array(all_lbl)


def extract_dinov2_embeddings(model, items, transform, device, batch_size=64):
    all_emb, all_lbl = [], []
    for i in tqdm(range(0, len(items), batch_size), desc="DINOv2", leave=False):
        batch = items[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
        with torch.no_grad():
            emb = F.normalize(model(imgs).float(), dim=1).cpu()
        all_emb.append(emb)
        all_lbl.extend([canonical_label(l) for _, l in batch])
    return torch.cat(all_emb).numpy(), np.array(all_lbl)

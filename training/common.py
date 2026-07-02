# -*- coding: utf-8 -*-
"""Shared training utilities (lifted from smartwoodid_experiments_full.py:147-216)."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, metrics, path, **extra):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "epoch": epoch, "metrics": metrics, **extra}, path)


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("metrics", {}), ckpt.get("epoch", 0)


def compute_recall_at_k(q_embs, g_embs, q_labels, g_labels, ks=(1, 5, 10)):
    q = F.normalize(torch.tensor(q_embs).float(), dim=1)
    g = F.normalize(torch.tensor(g_embs).float(), dim=1)
    ql, gl = np.asarray(q_labels), np.asarray(g_labels)
    sims = (q @ g.T).cpu().numpy()
    self_retrieval = (q_embs is g_embs or
                      (q_embs.shape == g_embs.shape and np.allclose(q_embs, g_embs, atol=1e-6)))
    if self_retrieval:
        np.fill_diagonal(sims, -1)
    out = {}
    for k in ks:
        top_k = np.argsort(-sims, axis=1)[:, :k]
        hits = sum(1 for i, row in enumerate(top_k) if ql[i] in gl[row])
        out[f"R@{k}"] = hits / len(ql)
    return out


def extract_embeddings(model, loader, device, return_logits=False):
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

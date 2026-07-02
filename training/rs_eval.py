# -*- coding: utf-8 -*-
"""Soft-retrieval scoring helpers (lifted verbatim from
smartwoodid_experiments_full.py:5116-5187, 5446-5496).

Only the subset the B1 density-gate trainer needs: L2-norm, softmax, top-m
soft-retrieval class scores, retrieval energy, macro accuracy, density features,
and the species support/query split. The eval package has its own scoring path
(experiments/registry.py); this copy keeps the training subpackage self-contained.
"""

import numpy as np


def rs_l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def rs_softmax(a, axis=1):
    a = np.asarray(a, dtype=np.float64)
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


def soft_retrieval_class_scores(query_embs, gallery_embs, gallery_labels,
                                top_m=50, tau=0.07, class_order=None):
    q = rs_l2(query_embs)
    g = rs_l2(gallery_embs)
    gl = np.asarray(gallery_labels)
    if class_order is None:
        class_order = np.array(sorted(np.unique(gl)))
    else:
        class_order = np.asarray(class_order)
    class_to_col = {c: i for i, c in enumerate(class_order)}
    sims = q @ g.T
    top_m_eff = min(int(top_m), sims.shape[1])
    top_idx = np.argpartition(-sims, top_m_eff - 1, axis=1)[:, :top_m_eff]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    weights = rs_softmax(top_sims / max(float(tau), 1e-6), axis=1)
    scores = np.zeros((len(q), len(class_order)), dtype=np.float64)
    for i in range(len(q)):
        for j, c in enumerate(gl[top_idx[i]]):
            if c in class_to_col:
                scores[i, class_to_col[c]] += weights[i, j]
    scores = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    return scores, class_order, top_idx, top_sims


def retrieval_energy_from_top_sims(top_sims, tau=0.07):
    tau = max(float(tau), 1e-6)
    m = np.max(top_sims, axis=1, keepdims=True)
    return -(m[:, 0] + tau * np.log(np.exp((top_sims - m) / tau).sum(axis=1)))


def predict_from_scores(scores, class_order):
    return np.asarray(class_order)[np.asarray(scores).argmax(axis=1)]


def macro_from_preds(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    per_sp = {c: float((preds[labels == c] == c).mean()) for c in sorted(np.unique(labels))}
    return (float(np.mean(list(per_sp.values()))) if per_sp else 0.0), per_sp


def soft_retrieval_eval(query_embs, query_labels, gallery_embs, gallery_labels,
                        top_m=50, tau=0.07, class_order=None):
    scores, cls, top_idx, top_sims = soft_retrieval_class_scores(
        query_embs, gallery_embs, gallery_labels, top_m=top_m, tau=tau, class_order=class_order)
    preds = predict_from_scores(scores, cls)
    macro, per_sp = macro_from_preds(preds, query_labels)
    energy = retrieval_energy_from_top_sims(top_sims, tau=tau)
    conf = scores.max(axis=1)
    entropy = -(scores * np.log(np.maximum(scores, 1e-12))).sum(axis=1)
    return {
        "mean": float(macro), "per_species": per_sp,
        "scores": scores, "class_order": cls,
        "preds": preds, "energy": energy,
        "confidence": conf, "entropy": entropy,
        "top_idx": top_idx, "top_sims": top_sims,
    }


def density_features(ev):
    scores = np.asarray(ev["scores"], dtype=np.float64)
    top_sims = np.asarray(ev["top_sims"], dtype=np.float64)
    top1_dist = 1.0 - top_sims[:, 0]
    mean5_dist = 1.0 - top_sims[:, :min(5, top_sims.shape[1])].mean(axis=1)
    mean10_dist = 1.0 - top_sims[:, :min(10, top_sims.shape[1])].mean(axis=1)
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    gap = sorted_scores[:, 0] - sorted_scores[:, 1] if scores.shape[1] > 1 else sorted_scores[:, 0]
    return np.stack([top1_dist, mean5_dist, mean10_dist, ev["energy"], ev["entropy"], gap],
                    axis=1).astype(np.float32)


def make_species_split(labels, support_per_species=5, query_per_species=10, seed=42):
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    support_idx, query_idx = [], []
    for sp in sorted(np.unique(labels)):
        idx = np.where(labels == sp)[0]
        if len(idx) < 2:
            continue
        rng.shuffle(idx)
        n_support = min(support_per_species, max(1, len(idx) // 2))
        n_query = min(query_per_species, max(1, len(idx) - n_support))
        support_idx.extend(idx[:n_support].tolist())
        query_idx.extend(idx[n_support:n_support + n_query].tolist())
    return np.array(support_idx, dtype=int), np.array(query_idx, dtype=int)

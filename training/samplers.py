# -*- coding: utf-8 -*-
"""PKSampler + episodic samplers (lifted from smartwoodid_experiments_full.py
L585-609, L5639-5693)."""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Sample P classes × K images per batch."""

    def __init__(self, labels, P=16, K=4):
        self.labels = np.array(labels)
        self.P, self.K = P, K
        self.class_to_idxs = defaultdict(list)
        for i, l in enumerate(self.labels):
            self.class_to_idxs[l].append(i)
        self.classes = [c for c, idxs in self.class_to_idxs.items() if len(idxs) >= 2]
        if len(self.classes) < P:
            self.classes = list(self.class_to_idxs.keys())

    def __iter__(self):
        classes = self.classes.copy()
        random.shuffle(classes)
        batch = []
        for cls in classes:
            idxs = self.class_to_idxs[cls]
            chosen = random.choices(idxs, k=self.K) if len(idxs) < self.K else random.sample(idxs, self.K)
            batch.extend(chosen)
            if len(batch) >= self.P * self.K:
                yield batch[:self.P * self.K]
                batch = []

    def __len__(self):
        return len(self.classes) // self.P


def urd_episode_indices(labels, n_way, k_support, q_query, rng):
    labels = np.asarray(labels)
    by_sp = {sp: np.where(labels == sp)[0] for sp in np.unique(labels)}
    valid = [sp for sp, idx in by_sp.items() if len(idx) >= k_support + q_query]
    if len(valid) < n_way:
        valid = [sp for sp, idx in by_sp.items() if len(idx) >= 2]
    if not valid:
        raise RuntimeError("URD episodic training needs a species with >=2 images.")
    chosen = rng.choice(valid, size=min(n_way, len(valid)), replace=False)
    support, query, st, qt = [], [], [], []
    for ci, sp in enumerate(chosen):
        idx = by_sp[sp].copy(); rng.shuffle(idx)
        n_s = min(k_support, max(1, len(idx) // 2))
        n_q = min(q_query, max(1, len(idx) - n_s))
        support.extend(idx[:n_s].tolist()); query.extend(idx[n_s:n_s + n_q].tolist())
        st.extend([ci] * n_s); qt.extend([ci] * n_q)
    return (np.array(support), np.array(query),
            torch.tensor(qt, dtype=torch.long), torch.tensor(st, dtype=torch.long))


def scurd_episode_indices(labels, n_way, k_support, q_query, q_ood, rng):
    labels = np.asarray(labels)
    by_sp = {sp: np.where(labels == sp)[0] for sp in np.unique(labels)}
    valid = [sp for sp, idx in by_sp.items() if len(idx) >= k_support + q_query]
    if len(valid) < n_way:
        valid = [sp for sp, idx in by_sp.items() if len(idx) >= 2]
    if not valid:
        raise RuntimeError("SC-URD episodic training needs a species with >=2 images.")
    chosen = rng.choice(valid, size=min(n_way, len(valid)), replace=False)
    support, query, st, qt = [], [], [], []
    for ci, sp in enumerate(chosen):
        idx = by_sp[sp].copy(); rng.shuffle(idx)
        n_s = min(k_support, max(1, len(idx) // 2))
        n_q = min(q_query, max(1, len(idx) - n_s))
        support.extend(idx[:n_s].tolist()); query.extend(idx[n_s:n_s + n_q].tolist())
        st.extend([ci] * n_s); qt.extend([ci] * n_q)
    ood_species = [sp for sp in by_sp if sp not in set(chosen)]
    ood_idx = []
    if ood_species and q_ood > 0:
        pool = np.concatenate([by_sp[sp] for sp in ood_species])
        ood_idx = rng.choice(pool, size=min(q_ood, len(pool)), replace=False).tolist()
    return (np.array(support), np.array(query), np.array(ood_idx),
            torch.tensor(qt, dtype=torch.long), torch.tensor(st, dtype=torch.long))


def urd_logits(query_z, support_z, support_targets, n_way, tau):
    sim = query_z @ support_z.T / max(float(tau), 1e-6)
    return torch.stack([torch.logsumexp(sim[:, support_targets.to(sim.device) == c], dim=1)
                        for c in range(n_way)], dim=1)

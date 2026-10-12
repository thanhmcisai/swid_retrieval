"""Scan-disjoint, scale-balanced episodes and top-rank surrogates."""

import math
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from . import gallery_experiment as study


def episode_catalog(items, budgets):
    by_label = defaultdict(lambda: defaultdict(list))
    scales = []
    for index, (path, label) in enumerate(items):
        scale = study.image_scale(path)
        if scale is None:
            raise ValueError(f"Missing crop scale: {path}")
        scales.append(scale)
        by_label[study.canonical(label)][study.source_scan_id(path)].append(index)
    eligible = {k: {} for k in budgets}
    for label, scans in by_label.items():
        for scan, rows in scans.items():
            support = [i for other, indices in scans.items() if other != scan
                       for i in indices]
            support_scales = {scales[i] for i in support}
            for k in budgets:
                if len(support) >= k and len(support_scales) >= 2:
                    eligible[k].setdefault(label, []).extend(
                        i for i in rows if support_scales - {scales[i]})
    return by_label, scales, eligible


def select_support(support, scales, scans, k, balanced, rng):
    if len(support) < k:
        raise ValueError("Not enough scan-disjoint support images")
    if not balanced:
        return rng.choice(support, size=k, replace=False).astype(int).tolist()
    remaining = rng.permutation(support).astype(int).tolist()
    scale_count, scan_count, pair_count = Counter(), Counter(), Counter()
    selected = []
    for _ in range(k):
        chosen = min(remaining, key=lambda i: (
            scale_count[scales[i]], scan_count[scans[i]],
            pair_count[(scans[i], scales[i])]))
        remaining.remove(chosen)
        selected.append(chosen)
        scale_count[scales[chosen]] += 1
        scan_count[scans[chosen]] += 1
        pair_count[(scans[chosen], scales[chosen])] += 1
    return selected


def plan_episodes(items, catalog, schedule, episodes, seed, epoch, balanced):
    by_label, scales, eligible = catalog
    scans = [study.source_scan_id(path) for path, _ in items]
    rng = np.random.default_rng(seed * 100003 + epoch)
    decks = defaultdict(list)
    batches, layouts = [], []
    for step in range(episodes):
        ways, k = schedule[step % len(schedule)]
        choices = sorted(eligible[k])
        if len(choices) < ways:
            raise ValueError(f"Only {len(choices)} eligible species for {ways}-way K={k}")
        classes = rng.choice(choices, size=ways, replace=False).tolist()
        refs, queries, labels, ref_scans = [], [], [], []
        for class_id, label in enumerate(classes):
            deck_key = (k, label)
            if not decks[deck_key]:
                decks[deck_key] = rng.permutation(eligible[k][label]).tolist()
            query = int(decks[deck_key].pop())
            support = [i for scan, rows in by_label[label].items()
                       if scan != scans[query] for i in rows]
            support_rng = np.random.default_rng(
                seed * 1000003 + epoch * 10007 + step * 101 + class_id)
            chosen = select_support(support, scales, scans, k, balanced, support_rng)
            if any(scans[i] == scans[query] for i in chosen):
                raise ValueError("Query and support share a source scan")
            if balanced and not ({scales[i] for i in chosen} - {scales[query]}):
                raise ValueError("Scale-balanced episode lacks a cross-scale positive")
            if balanced and len({scales[i] for i in support}) <= k and (
                    {scales[i] for i in chosen} != {scales[i] for i in support}):
                raise ValueError("Scale-balanced episode omitted an available scale")
            refs.extend(chosen)
            queries.append(query)
            labels.extend([class_id] * k)
            ref_scans.extend(scans[i] for i in chosen)
        batches.append(refs + queries)
        layouts.append((len(refs), labels, ref_scans, classes,
                        [scales[i] for i in refs], [scales[i] for i in queries]))
    return batches, layouts


def position_loss(scores, target, temperature=0.07, margin=0.0,
                  weight_by_rank=False):
    """Smooth pairwise inversion penalty, optionally upweighted by current rank.

    The detached rank only weights a differentiable loss; it is not treated as a
    gradient-bearing quantity. The weight is bounded to avoid letting a few
    ambiguous or mislabeled examples dominate optimization.
    """
    if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] < 2:
        raise ValueError("Expected one query and at least two candidate classes")
    if not 0 <= target < scores.shape[1] or temperature <= 0:
        raise ValueError("Invalid target or temperature")
    positive = scores[:, target:target + 1]
    negatives = torch.cat((scores[:, :target], scores[:, target + 1:]), dim=1)
    gaps = (negatives - positive + margin) / temperature
    loss = torch.log1p(F.softplus(gaps).sum(dim=1)).mean()
    rank = 1 + int((negatives.detach() >= positive.detach()).sum())
    if weight_by_rank:
        loss = loss * (1 + math.log(rank) / math.log(scores.shape[1]))
    return loss, rank

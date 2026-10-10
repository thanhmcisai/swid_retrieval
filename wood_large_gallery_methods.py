"""Class-balanced candidate search and bounded correspondence reranking."""

import torch
from torch.nn import functional as F


def class_scores(queries, references, labels, n_classes=None, chunk=32):
    """Return prototype and nearest-image scores without a dense query gallery matrix."""
    if queries.ndim != 2 or references.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected query/reference matrices and reference labels")
    if len(references) != len(labels) or queries.shape[1] != references.shape[1]:
        raise ValueError("Embedding or label count mismatch")
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("Invalid class labels")
    counts = torch.bincount(labels, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError("Every gallery class must have a reference")
    q = F.normalize(queries.float(), dim=-1)
    r = F.normalize(references.float(), dim=-1)
    sums = r.new_zeros((n_classes, r.shape[1]))
    sums.index_add_(0, labels, r)
    prototypes = F.normalize(sums / counts[:, None], dim=-1)
    prototype = q @ prototypes.T
    nearest = torch.empty_like(prototype)
    index = labels[None, :]
    for first in range(0, len(q), chunk):
        sim = q[first:first + chunk] @ r.T
        out = sim.new_full((len(sim), n_classes), -torch.inf)
        out.scatter_reduce_(1, index.expand(len(sim), -1), sim,
                            reduce="amax", include_self=True)
        nearest[first:first + chunk] = out
    return prototype, nearest


def candidate_union(score_matrices, width):
    """Union equal-width top lists, ordered by reciprocal-rank fusion."""
    if not score_matrices or width < 1:
        raise ValueError("Candidate scores and width are required")
    shape = score_matrices[0].shape
    if any(matrix.shape != shape or not torch.isfinite(matrix).all()
           for matrix in score_matrices):
        raise ValueError("Candidate score matrices must match and be finite")
    n = min(width, shape[1])
    fused = torch.zeros_like(score_matrices[0])
    for matrix in score_matrices:
        order = matrix.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = (torch.arange(shape[1], device=matrix.device)[None] + 1)
        ranks.scatter_(1, order, rank_values.expand_as(order))
        contribution = (1.0 / (60.0 + ranks.float())).masked_fill(ranks > n, 0)
        fused += contribution
    selected = fused.topk(n, dim=1).indices
    return selected, fused


def candidate_recall(candidates, targets):
    if candidates.ndim != 2 or targets.shape != (len(candidates),):
        raise ValueError("Candidate/target shape mismatch")
    return float((candidates == targets[:, None]).any(dim=1).float().mean())


def rerank_candidates(model, query, query_tokens, references, reference_tokens,
                      reference_labels, scans, candidates, base_scores,
                      *, mode="qkv", max_references_per_class=3):
    """Score candidate species using bounded reference evidence.

    Non-candidates are excluded. Candidate scores are mapped back to global
    class IDs before ranking.
    """
    if mode not in {"qkv", "global"} or max_references_per_class < 1:
        raise ValueError("Unsupported reranking configuration")
    if len(query) != len(query_tokens) or len(query) != len(candidates):
        raise ValueError("Query/candidate count mismatch")
    result = torch.full_like(base_scores, -torch.inf)
    for row in range(len(query)):
        chosen = candidates[row].unique(sorted=True)
        picked = []
        for class_id in chosen.tolist():
            indices = (reference_labels == class_id).nonzero(as_tuple=True)[0]
            seen_scans = set()
            for index in indices.tolist():
                scan = str(scans[index])
                if scan not in seen_scans:
                    picked.append(index)
                    seen_scans.add(scan)
                if len(seen_scans) >= max_references_per_class:
                    break
            if not seen_scans:
                raise ValueError("Candidate has no references")
        picked = torch.tensor(picked, device=query.device)
        local_labels = torch.searchsorted(chosen, reference_labels[picked])
        values, classes = model(query[row:row + 1], query_tokens[row:row + 1],
                                references[picked], reference_tokens[picked],
                                local_labels, [scans[i] for i in picked.tolist()],
                                mode=mode, query_chunk=1)
        if not torch.equal(classes, torch.arange(len(chosen), device=query.device)):
            raise ValueError("Reranker class order changed")
        result[row, chosen] = values[0]
    return result

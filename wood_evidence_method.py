"""Candidate-limited local evidence for scan-disjoint wood-image retrieval."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def prototype_scores(queries, references, reference_labels):
    """Cosine scores against L2-normalized per-species reference means."""
    classes, inverse = torch.unique(reference_labels.long(), sorted=True,
                                    return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(classes))
    sums = references.new_zeros((len(classes), references.shape[-1]))
    sums.index_add_(0, inverse, references)
    prototypes = F.normalize(sums / counts[:, None], dim=-1)
    return queries @ prototypes.T, classes


def local_pair_scores(query_tokens, reference_tokens):
    """Symmetric patch-set matching, without assuming spatial alignment."""
    if query_tokens.ndim != 2 or reference_tokens.ndim != 3:
        raise ValueError("Expected query [T,D] and references [R,T,D]")
    if query_tokens.shape != reference_tokens.shape[1:]:
        raise ValueError("Query and reference patch shapes differ")
    affinity = torch.einsum("td,rsd->rts", query_tokens.float(),
                            reference_tokens.float())
    return (affinity.max(dim=2).values.mean(dim=1) +
            affinity.max(dim=1).values.mean(dim=1)) / 2


def local_pair_scores_batch(query_tokens, reference_tokens):
    if query_tokens.ndim != 3 or reference_tokens.ndim != 3:
        raise ValueError("Expected query and reference patch batches")
    if query_tokens.shape[1:] != reference_tokens.shape[1:]:
        raise ValueError("Query and reference patch shapes differ")
    affinity = torch.einsum("qtd,rsd->qrts", query_tokens.float(),
                            reference_tokens.float())
    return (affinity.max(dim=3).values.mean(dim=2) +
            affinity.max(dim=2).values.mean(dim=2)) / 2


def scan_consensus_batch(pair_scores, labels, scan_ids, classes, enabled=True):
    if pair_scores.ndim != 2 or pair_scores.shape[1] != len(labels) or len(labels) != len(scan_ids):
        raise ValueError("Reference evidence, labels, and scan IDs must align")
    group_map, group_labels, group_ids = {}, [], []
    for label, scan in zip(labels.tolist(), scan_ids):
        key = (label, str(scan)) if enabled else (label, "all")
        if key not in group_map:
            group_map[key] = len(group_map)
            group_labels.append(label)
        group_ids.append(group_map[key])
    ids = torch.as_tensor(group_ids, device=pair_scores.device)
    grouped = pair_scores.new_full((len(pair_scores), len(group_map)), -torch.inf)
    grouped.scatter_reduce_(1, ids[None].expand(len(pair_scores), -1), pair_scores,
                            reduce="amax", include_self=True)
    result = []
    for label in classes.tolist():
        positions = [i for i, group_label in enumerate(group_labels) if group_label == label]
        if not positions:
            raise ValueError("Candidate class has no local references")
        values = grouped[:, positions]
        result.append(values.topk(min(2, values.shape[1]), dim=1).values.mean(dim=1))
    return torch.stack(result, dim=1)


def scan_consensus(pair_scores, labels, scan_ids, classes, enabled=True):
    """Reduce multiple views per scan before pooling evidence across scans."""
    if pair_scores.ndim != 1:
        raise ValueError("Expected one query's reference scores")
    return scan_consensus_batch(pair_scores[None], labels, scan_ids, classes,
                                enabled=enabled)[0]


class WoodEvidenceReranker(nn.Module):
    """Global shortlist plus a trainable, count-normalized local score residual."""

    def __init__(self, dimension=512, adapter_width=64, initial_weight=0.25):
        super().__init__()
        if dimension < 1 or adapter_width < 1 or initial_weight <= 0:
            raise ValueError("Invalid local scorer dimensions or initial weight")
        self.adapter = nn.Sequential(nn.Linear(dimension, adapter_width), nn.ReLU(),
                                     nn.Linear(adapter_width, dimension))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.weight_logit = nn.Parameter(torch.tensor(math.log(math.expm1(initial_weight))))

    def project_tokens(self, tokens):
        tokens = tokens.float()
        return F.normalize(tokens + self.adapter(tokens), dim=-1)

    def forward(self, queries, query_tokens, references, reference_tokens,
                reference_labels, scan_ids, *, top_classes=128, consensus=True,
                local=True, weight_override=None):
        if (queries.ndim != 2 or references.ndim != 2 or query_tokens.ndim != 3 or
                reference_tokens.ndim != 3 or len(queries) != len(query_tokens) or
                len(references) != len(reference_tokens) or
                queries.shape[-1] != query_tokens.shape[-1] or
                references.shape[-1] != reference_tokens.shape[-1] or
                len(references) != len(reference_labels) or len(scan_ids) != len(references)):
            raise ValueError("Global/local feature and reference metadata alignment failed")
        base, classes = prototype_scores(queries.float(), references.float(),
                                         reference_labels)
        if not local:
            return base, classes
        if top_classes < 1:
            raise ValueError("top_classes must be positive")
        query_tokens = self.project_tokens(query_tokens)
        reference_tokens = self.project_tokens(reference_tokens)
        weight = (F.softplus(self.weight_logit).clamp(max=4) if weight_override is None else
                  base.new_tensor(float(weight_override)))
        scores = base.clone()
        if top_classes >= len(classes):
            evidence = scan_consensus_batch(
                local_pair_scores_batch(query_tokens, reference_tokens),
                reference_labels, scan_ids, classes, enabled=consensus)
            centered = evidence - evidence.mean(dim=1, keepdim=True)
            scale = evidence.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
            global_scale = base.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
            scores = scores + weight * global_scale * centered / scale
            if not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite local reranking scores")
            return scores, classes
        for query_index in range(len(queries)):
            candidate_indices = base[query_index].topk(min(top_classes, len(classes))).indices
            candidate_classes = classes[candidate_indices]
            selected_mask = torch.isin(reference_labels, candidate_classes)
            selected_labels = reference_labels[selected_mask]
            selected_scans = [scan for scan, take in zip(scan_ids, selected_mask.tolist())
                              if take]
            pair = local_pair_scores(query_tokens[query_index],
                                     reference_tokens[selected_mask])
            evidence = scan_consensus(pair, selected_labels, selected_scans,
                                      candidate_classes, enabled=consensus)
            centered = evidence - evidence.mean()
            scale = evidence.std(unbiased=False).clamp_min(1e-4)
            global_scale = base[query_index, candidate_indices].std(
                unbiased=False).clamp_min(1e-4)
            scores[query_index, candidate_indices] += weight * global_scale * centered / scale
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite local reranking scores")
        return scores, classes

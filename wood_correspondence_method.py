"""All-class tissue-token retrieval with optional learned QKV correspondence."""

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F


def class_prototypes(references, labels):
    classes, inverse = torch.unique(labels.long(), sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(classes))
    sums = references.new_zeros((len(classes), references.shape[-1]))
    sums.index_add_(0, inverse, references)
    return F.normalize(sums / counts[:, None], dim=-1), classes, inverse


def scan_evidence(pair_scores, inverse, scans, n_classes, consensus):
    """Maximum evidence per scan, then at most two independent scans per class."""
    if not consensus:
        result = pair_scores.new_full((len(pair_scores), n_classes), -torch.inf)
        result.scatter_reduce_(1, inverse[None].expand(len(pair_scores), -1),
                               pair_scores, reduce="amax", include_self=True)
        return result
    slots, group_ids, slot_counts = {}, [], defaultdict(int)
    for label, scan in zip(inverse.tolist(), scans):
        key = (label, str(scan))
        if key not in slots:
            slots[key] = slot_counts[label]
            slot_counts[label] += 1
        group_ids.append((label, slots[key]))
    width = max(slot_counts.values())
    flat = torch.tensor([label * width + slot for label, slot in group_ids],
                        device=pair_scores.device)
    grouped = pair_scores.new_full((len(pair_scores), n_classes * width), -torch.inf)
    grouped.scatter_reduce_(1, flat[None].expand(len(pair_scores), -1),
                            pair_scores, reduce="amax", include_self=True)
    top = grouped.view(len(pair_scores), n_classes, width).topk(min(2, width), dim=2).values
    if width == 1:
        return top[:, :, 0]
    return torch.where(torch.isfinite(top[:, :, 1]), top.mean(dim=2), top[:, :, 0])


class WoodCorrespondence(nn.Module):
    """Shared global adapter and symmetric query/reference token interaction."""

    def __init__(self, dimension=512, token_dim=64, initial_weight=0.25):
        super().__init__()
        self.global_adapter = nn.Linear(dimension, dimension, bias=False)
        nn.init.zeros_(self.global_adapter.weight)
        self.query_key = nn.Linear(dimension, token_dim, bias=False)
        self.reference_key = nn.Linear(dimension, token_dim, bias=False)
        self.value = nn.Linear(dimension, token_dim, bias=False)
        nn.init.orthogonal_(self.query_key.weight)
        self.reference_key.weight.data.copy_(self.query_key.weight.data)
        self.value.weight.data.copy_(self.query_key.weight.data)
        self.salience = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.salience.weight)
        nn.init.zeros_(self.salience.bias)
        self.log_attention_scale = nn.Parameter(torch.tensor(1.6094379))
        self.log_local_weight = nn.Parameter(
            torch.tensor(float(torch.log(torch.expm1(torch.tensor(initial_weight))))))

    def _pair_block(self, qk, qv, qweights, rk, rv, rweights, mode):
        affinity = torch.einsum("btd,rsd->brts", qk, rk)
        if mode == "maxsim":
            forward = affinity.max(dim=3).values
            backward = affinity.max(dim=2).values
        else:
            scale = self.log_attention_scale.clamp(-2, 4).exp()
            attention = (affinity * scale).softmax(dim=3)
            aligned_ref = torch.einsum("brts,rsd->brtd", attention, rv)
            forward = (qv[:, None] * F.normalize(aligned_ref, dim=-1)).sum(dim=-1)
            reverse_attention = (affinity * scale).softmax(dim=2)
            aligned_query = torch.einsum("brts,btd->brsd", reverse_attention, qv)
            backward = (rv[None] * F.normalize(aligned_query, dim=-1)).sum(dim=-1)
        return ((forward * qweights[:, None]).sum(dim=2) +
                (backward * rweights[None]).sum(dim=2)) / 2

    def forward(self, queries, query_tokens, references, reference_tokens,
                reference_labels, scans, *, mode="qkv", top_classes=None,
                consensus=True, gate=True, query_chunk=8, reference_chunk=64):
        if (queries.ndim != 2 or references.ndim != 2 or query_tokens.ndim != 3 or
                reference_tokens.ndim != 3 or len(queries) != len(query_tokens) or
                len(references) != len(reference_tokens) or
                len(references) != len(reference_labels) or len(scans) != len(references) or
                queries.shape[-1] != references.shape[-1] or
                query_tokens.shape[1:] != reference_tokens.shape[1:] or
                query_tokens.shape[-1] != queries.shape[-1] or
                query_chunk < 1 or reference_chunk < 1 or
                mode not in {"global", "qkv", "maxsim"}):
            raise ValueError("Invalid correspondence inputs or score mode")
        q = F.normalize(queries.float() + self.global_adapter(queries.float()), dim=-1)
        r = F.normalize(references.float() + self.global_adapter(references.float()), dim=-1)
        prototypes, classes, inverse = class_prototypes(r, reference_labels)
        base = q @ prototypes.T
        if mode == "global":
            return base, classes
        qk = F.normalize(self.query_key(query_tokens.float()), dim=-1)
        rk = F.normalize(self.reference_key(reference_tokens.float()), dim=-1)
        qv = F.normalize(self.value(query_tokens.float()), dim=-1)
        rv = F.normalize(self.value(reference_tokens.float()), dim=-1)
        if gate:
            qw = self.salience(qv).squeeze(-1).softmax(dim=-1)
            rw = self.salience(rv).squeeze(-1).softmax(dim=-1)
        else:
            qw = qv.new_full(qv.shape[:2], 1 / qv.shape[1])
            rw = rv.new_full(rv.shape[:2], 1 / rv.shape[1])
        if top_classes is not None and top_classes < len(classes):
            if top_classes < 1:
                raise ValueError("top_classes must be positive")
            scores = base.clone()
            for index in range(len(q)):
                candidates = base[index].topk(top_classes).indices
                selected = torch.isin(inverse, candidates)
                selected_ids = selected.nonzero(as_tuple=True)[0]
                candidate_labels = torch.searchsorted(
                    candidates.sort().values, inverse[selected_ids])
                sorted_candidates = candidates.sort().values
                pair_parts = []
                for first in range(0, len(selected_ids), reference_chunk):
                    current = selected_ids[first:first + reference_chunk]
                    pair_parts.append(self._pair_block(
                        qk[index:index + 1], qv[index:index + 1], qw[index:index + 1],
                        rk[current], rv[current], rw[current], mode))
                pair = torch.cat(pair_parts, dim=1)
                selected_scans = [scans[j] for j in selected_ids.tolist()]
                evidence = scan_evidence(pair, candidate_labels, selected_scans,
                                         len(sorted_candidates), consensus)
                centered = evidence - evidence.mean(dim=1, keepdim=True)
                standardized = centered / evidence.std(
                    dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
                scale = base[index, sorted_candidates].std(unbiased=False).clamp_min(1e-4)
                residual = F.softplus(self.log_local_weight).clamp(max=4) * scale * standardized
                scores[index, sorted_candidates] = scores[index, sorted_candidates] + residual[0]
            if not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite shortlist correspondence scores")
            return scores, classes
        blocks = []
        for qstart in range(0, len(q), query_chunk):
            pieces = []
            for rstart in range(0, len(r), reference_chunk):
                stop = rstart + reference_chunk
                pieces.append(self._pair_block(qk[qstart:qstart + query_chunk],
                                               qv[qstart:qstart + query_chunk],
                                               qw[qstart:qstart + query_chunk],
                                               rk[rstart:stop], rv[rstart:stop],
                                               rw[rstart:stop], mode))
            blocks.append(torch.cat(pieces, dim=1))
        pair = torch.cat(blocks, dim=0)
        evidence = scan_evidence(pair, inverse, scans, len(classes), consensus)
        centered = evidence - evidence.mean(dim=1, keepdim=True)
        standardized = centered / evidence.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        scale = base.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        residual = F.softplus(self.log_local_weight).clamp(max=4) * scale * standardized
        scores = base + residual
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite all-class correspondence scores")
        return scores, classes

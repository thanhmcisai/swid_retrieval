"""Gallery-conditioned retrieval model used by the isolated research experiment.

The backbone is trainable by default. Its image-forward and gradient replay are
kept outside this module so the same model works for frozen and full fine-tuning.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GalleryEncoder(nn.Module):
    def __init__(self, backbone, feature_dim=768, embedding_dim=512):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )

    def project(self, features):
        return F.normalize(self.projection(features.float()), dim=-1)

    def forward(self, images):
        return self.project(self.backbone(images))


class GalleryScorer(nn.Module):
    """One score per class from its prototype and globally retrieved evidence.

    Unlike an uncorrected log-sum-exp, the evidence term is divided by the
    number of retrieved references for that class. Classes absent from the
    global candidate set retain their prototype score.
    """

    def __init__(self, top_m=64, temperature=0.07, mode="learned", normalize_evidence=True):
        super().__init__()
        if mode not in {"learned", "prototype", "fixed"}:
            raise ValueError(f"Unknown scoring mode: {mode}")
        self.top_m = int(top_m)
        self.temperature = float(temperature)
        self.mode = mode
        self.normalize_evidence = bool(normalize_evidence)
        if self.top_m < 1 or self.temperature <= 0:
            raise ValueError("top_m and temperature must be positive")
        self.evidence_gate = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.evidence_gate[-1].weight)
        nn.init.zeros_(self.evidence_gate[-1].bias)

    def forward(self, query, references, reference_labels):
        if query.ndim != 2 or references.ndim != 2 or len(references) != len(reference_labels):
            raise ValueError("Expected query [Q,D], references [G,D], labels [G]")
        if len(references) == 0:
            raise ValueError("The reference gallery is empty")
        labels = reference_labels.long()
        classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        n_classes = len(classes)
        counts = torch.bincount(inverse, minlength=n_classes)
        sums = torch.zeros((n_classes, references.shape[1]), dtype=references.dtype,
                           device=references.device).index_add(0, inverse, references)
        prototypes = F.normalize(sums / counts[:, None], dim=-1)
        proto = query @ prototypes.T / self.temperature
        if self.mode == "prototype":
            return proto, classes

        similarity = query @ references.T / self.temperature
        selected, indices = similarity.topk(min(self.top_m, len(references)), dim=1)
        selected_classes = inverse[indices]
        mask = selected_classes[:, :, None] == torch.arange(n_classes, device=query.device)[None, None, :]
        found = mask.any(dim=1)
        nearest = selected[:, :, None].masked_fill(~mask, -1e4).amax(dim=1)
        evidence = torch.logsumexp(selected[:, :, None].masked_fill(~mask, -1e4), dim=1)
        if self.normalize_evidence:
            evidence = evidence - mask.sum(dim=1).clamp(min=1).log()
        nearest = torch.where(found, nearest, proto)
        evidence = torch.where(found, evidence, proto)
        if self.mode == "fixed":
            return (proto + nearest + evidence) / 3, classes
        features = torch.stack((proto, nearest, evidence), dim=-1)
        return proto + self.evidence_gate(features).squeeze(-1), classes


def episode_objective(embeddings, scorer, n_way, n_support, n_query,
                      stability_weight=0.2, use_variable_gallery=True,
                      pseudo_ood_weight=0.1):
    """Balanced old/new episode with a smaller old-only reference gallery."""
    n_support_total = n_way * n_support
    support = embeddings[:n_support_total]
    query = embeddings[n_support_total:]
    expected = n_way * (n_support + n_query)
    if len(embeddings) != expected or n_way < 2:
        raise ValueError(f"Expected {expected} embeddings from a >=2-way episode")
    support_labels = torch.arange(n_way, device=embeddings.device).repeat_interleave(n_support)
    query_labels = torch.arange(n_way, device=embeddings.device).repeat_interleave(n_query)
    large_scores, _ = scorer(query, support, support_labels)
    loss = F.cross_entropy(large_scores, query_labels)
    diagnostics = {"large_ce": float(loss.detach())}
    if use_variable_gallery:
        n_old = n_way // 2
        old_support = support[:n_old * n_support].reshape(n_old, n_support, -1)[:, 0, :]
        old_query = query[:n_old * n_query]
        small_labels = torch.arange(n_old, device=embeddings.device)
        old_scores, _ = scorer(old_query, old_support, small_labels)
        old_targets = query_labels[:len(old_query)]
        old_ce = F.cross_entropy(old_scores, old_targets)
        # The expanded gallery introduces new negative species. A margin loss
        # explicitly penalizes loss of old-species identity after enrollment.
        old_large = large_scores[:len(old_query)]
        true_score = old_large.gather(1, old_targets[:, None]).squeeze(1)
        new_score = old_large[:, n_old:].amax(dim=1)
        expansion = F.relu(1.0 + new_score - true_score).mean()
        known_similarity = old_query @ old_support.T
        unknown_similarity = query[len(old_query):] @ old_support.T
        known_max = known_similarity.amax(dim=1)
        unknown_max = unknown_similarity.amax(dim=1)
        pseudo_ood = F.relu(0.1 + unknown_max[:, None] - known_max[None, :]).mean()
        loss = loss + old_ce + stability_weight * expansion + pseudo_ood_weight * pseudo_ood
        diagnostics.update(old_ce=float(old_ce.detach()), expansion=float(expansion.detach()),
                           pseudo_ood=float(pseudo_ood.detach()))
    return loss, diagnostics


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    similarities = embeddings @ embeddings.T / temperature
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = same & ~diagonal
    if not positive.any():
        raise ValueError("Supervised contrastive loss needs at least two images per class")
    log_prob = similarities - torch.logsumexp(similarities.masked_fill(diagonal, -1e4), dim=1)[:, None]
    per_query = -(log_prob * positive).sum(dim=1) / positive.sum(dim=1).clamp(min=1)
    return per_query[positive.any(dim=1)].mean()


class ArcFaceClassifier(nn.Module):
    """Training-only angular-margin baseline; inference uses its embeddings."""

    def __init__(self, embedding_dim, n_classes, margin=0.3, scale=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = float(margin)
        self.scale = float(scale)

    def forward(self, embeddings, labels):
        import math
        cosine = F.linear(embeddings.float(), F.normalize(self.weight.float(), dim=1))
        sine = torch.sqrt((1 - cosine.square()).clamp(min=1e-7))
        target = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        logits = torch.where(F.one_hot(labels.long(), self.weight.shape[0]).bool(), target, cosine)
        return F.cross_entropy(self.scale * logits, labels.long())

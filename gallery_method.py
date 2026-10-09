"""Gallery-conditioned retrieval model used by the isolated research experiment.

The backbone is trainable by default. Its image-forward and gradient replay are
kept outside this module so the same model works for frozen and full fine-tuning.
"""

from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F


class DensePatchBackbone(nn.Module):
    """Expose pooled DINOv2 patch tokens without changing its CLS embedding."""

    def __init__(self, model, grid_size=4):
        super().__init__()
        self.model = model
        self.grid_size = int(grid_size)
        self.num_features = model.num_features
        if self.grid_size < 1:
            raise ValueError("Patch evidence grid must be positive")

    def forward(self, images):
        return self.model(images)

    def forward_with_tokens(self, images):
        output = self.model.forward_features(images)
        patches = output["x_norm_patchtokens"]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("DINOv2 patch tokens do not form a square grid")
        patch_map = patches.transpose(1, 2).reshape(len(images), patches.shape[-1], side, side)
        local = F.adaptive_avg_pool2d(patch_map, (self.grid_size, self.grid_size))
        local = local.flatten(2).transpose(1, 2)
        return output["x_norm_clstoken"], local


class GalleryEncoder(nn.Module):
    def __init__(self, backbone, feature_dim=768, embedding_dim=512, local_dim=0,
                 local_feature_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.local_projection = (nn.Sequential(nn.LayerNorm(local_feature_dim),
                                               nn.Linear(local_feature_dim, local_dim))
                                 if local_dim else None)

    def project(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            projected = self.projection(features.float())
            return F.normalize(projected, dim=-1, eps=1e-6)

    def forward(self, images):
        return self.project(self.backbone(images))

    def project_tokens(self, tokens):
        if self.local_projection is None:
            raise ValueError("Local evidence is not enabled for this encoder")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return F.normalize(self.local_projection(tokens.float()), dim=-1, eps=1e-6)

    def forward_with_tokens(self, images):
        features, tokens = self.backbone.forward_with_tokens(images)
        return self.project(features), self.project_tokens(tokens)


def local_pair_similarity(query_tokens, reference_tokens):
    """Symmetric soft spatially unordered match of two small tissue-token sets."""
    if (query_tokens.ndim != 3 or reference_tokens.ndim != 3 or
            query_tokens.shape[-1] != reference_tokens.shape[-1]):
        raise ValueError("Expected query [Q,T,D] and reference [R,T,D] tokens")
    affinity = torch.einsum("qtd,rsd->qrts", query_tokens.float(), reference_tokens.float())
    return (affinity.max(dim=3).values.mean(dim=2) +
            affinity.max(dim=2).values.mean(dim=2)) / 2


def local_species_scores(query_tokens, reference_tokens, reference_labels, temperature):
    classes, inverse = torch.unique(reference_labels.long(), sorted=True, return_inverse=True)
    pair = local_pair_similarity(query_tokens, reference_tokens) / temperature
    scores = pair.new_full((len(query_tokens), len(classes)), -torch.inf)
    scores.scatter_reduce_(1, inverse[None].expand(len(query_tokens), -1),
                           pair, reduce="amax", include_self=True)
    return scores, classes


class GalleryMemory:
    """FIFO of detached meta-train species prototypes used as distractors.

    Current support and query images retain full gradients. Memory vectors are
    stale negatives and must never contain validation or public images.
    """

    def __init__(self, capacity=256, min_classes=80):
        self.capacity = int(capacity)
        self.min_classes = int(min_classes)
        if self.capacity < 0 or self.min_classes < 0 or self.min_classes > self.capacity:
            raise ValueError("Invalid gallery memory capacity or warmup")
        self.entries = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def add(self, embeddings, labels):
        if self.capacity == 0:
            return
        labels = labels.detach().cpu().long()
        embeddings = embeddings.detach().float().cpu()
        for label in labels.unique(sorted=True):
            key = int(label)
            prototype = F.normalize(embeddings[labels == label].mean(0), dim=0)
            self.entries.pop(key, None)
            self.entries[key] = prototype
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

    def distractors(self, active_labels, n_way, device):
        if len(self.entries) < self.min_classes:
            return None
        active = {int(label) for label in active_labels.detach().cpu().tolist()}
        selected = [vector for label, vector in self.entries.items() if label not in active]
        if not selected:
            return None
        vectors = torch.stack(selected).to(device)
        labels = torch.arange(n_way, n_way + len(selected), device=device)
        return vectors, labels

    def state_dict(self):
        return {"capacity": self.capacity, "min_classes": self.min_classes,
                "labels": list(self.entries), "vectors": [v.clone() for v in self.entries.values()]}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["min_classes"] != self.min_classes:
            raise ValueError("Gallery memory recipe changed")
        if len(state["labels"]) != len(state["vectors"]) or len(state["labels"]) > self.capacity:
            raise ValueError("Invalid gallery memory checkpoint")
        if len(set(state["labels"])) != len(state["labels"]):
            raise ValueError("Repeated class in gallery memory checkpoint")
        self.entries = OrderedDict((int(label), vector.clone().cpu()) for label, vector in
                                   zip(state["labels"], state["vectors"]))


class GalleryScorer(nn.Module):
    """One score per class from exact nearest-image or aggregated evidence.

    In learned/fixed modes, evidence is count-normalized; classes absent from
    global top-M image candidates retain their prototype score.
    """

    def __init__(self, top_m=64, temperature=0.07, mode="learned", normalize_evidence=True):
        super().__init__()
        if mode not in {"learned", "prototype", "fixed", "nearest"}:
            raise ValueError(f"Unknown scoring mode: {mode}")
        self.top_m = int(top_m)
        self.temperature = float(temperature)
        self.mode = mode
        self.normalize_evidence = bool(normalize_evidence)
        if self.top_m < 1 or self.temperature <= 0:
            raise ValueError("top_m and temperature must be positive")
        self.evidence_gate = None
        if mode == "learned":
            self.evidence_gate = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.evidence_gate[-1].weight)
            nn.init.zeros_(self.evidence_gate[-1].bias)

    def forward(self, query, references, reference_labels, similarity=None):
        if query.ndim != 2 or references.ndim != 2 or len(references) != len(reference_labels):
            raise ValueError("Expected query [Q,D], references [G,D], labels [G]")
        if len(references) == 0:
            raise ValueError("The reference gallery is empty")
        if similarity is not None and similarity.shape != (len(query), len(references)):
            raise ValueError("Precomputed similarity must have shape [Q,G]")
        labels = reference_labels.long()
        classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        n_classes = len(classes)
        if self.mode == "nearest":
            if similarity is None:
                similarity = query @ references.T
            scaled = similarity / self.temperature
            scores = scaled.new_full((len(query), n_classes), -torch.inf)
            scores.scatter_reduce_(1, inverse[None].expand(len(query), -1),
                                   scaled, reduce="amax", include_self=True)
            return scores, classes
        counts = torch.bincount(inverse, minlength=n_classes)
        sums = torch.zeros((n_classes, references.shape[1]), dtype=references.dtype,
                           device=references.device).index_add(0, inverse, references)
        prototypes = F.normalize(sums / counts[:, None], dim=-1)
        proto = query @ prototypes.T / self.temperature
        if self.mode == "prototype":
            return proto, classes

        if similarity is None:
            similarity = query @ references.T
        similarity = similarity / self.temperature
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
                      pseudo_ood_weight=0.1, background=None):
    """Balanced old/new episode with a smaller old-only reference gallery."""
    n_support_total = n_way * n_support
    support = embeddings[:n_support_total]
    query = embeddings[n_support_total:]
    expected = n_way * (n_support + n_query)
    if len(embeddings) != expected or n_way < 2:
        raise ValueError(f"Expected {expected} embeddings from a >=2-way episode")
    support_labels = torch.arange(n_way, device=embeddings.device).repeat_interleave(n_support)
    query_labels = torch.arange(n_way, device=embeddings.device).repeat_interleave(n_query)
    if background is not None:
        negative_refs, negative_labels = background
        if negative_refs.requires_grad or bool((negative_labels < n_way).any()):
            raise ValueError("Background must contain detached, disjoint distractors")
        large_gallery = torch.cat((support, negative_refs), dim=0)
        large_labels = torch.cat((support_labels, negative_labels), dim=0)
    else:
        large_gallery, large_labels = support, support_labels
    large_scores, _ = scorer(query, large_gallery, large_labels)
    loss = F.cross_entropy(large_scores, query_labels)
    diagnostics = {"large_ce": float(loss.detach()),
                   "train_episode_r1": float((large_scores.argmax(dim=1) == query_labels)
                                             .float().mean().detach()),
                   "gallery_images": len(large_gallery),
                   "gallery_species": int(large_labels.unique().numel()),
                   "top_m_active": scorer.mode in {"learned", "fixed"} and
                   len(large_gallery) > scorer.top_m}
    if use_variable_gallery:
        n_old = n_way // 2
        old_support = support[:n_old * n_support].reshape(n_old, n_support, -1)[:, 0, :]
        old_query = query[:n_old * n_query]
        small_labels = torch.arange(n_old, device=embeddings.device)
        if background is not None:
            old_gallery = torch.cat((old_support, negative_refs), dim=0)
            old_labels = torch.cat((small_labels, negative_labels), dim=0)
        else:
            old_gallery, old_labels = old_support, small_labels
        old_scores, _ = scorer(old_query, old_gallery, old_labels)
        old_targets = query_labels[:len(old_query)]
        old_ce = F.cross_entropy(old_scores, old_targets)
        # The expanded gallery introduces new episode species, while detached
        # meta-train prototypes supply additional competitors in both views.
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


def supervised_contrastive_loss(embeddings, labels, temperature=0.07, background=None):
    similarities = embeddings @ embeddings.T / temperature
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = same & ~diagonal
    if not positive.any():
        raise ValueError("Supervised contrastive loss needs at least two images per class")
    denominator_scores = similarities.masked_fill(diagonal, -1e4)
    if background is not None:
        if background.requires_grad:
            raise ValueError("SupCon background must be detached")
        denominator_scores = torch.cat((denominator_scores,
                                        embeddings @ background.T / temperature), dim=1)
    log_prob = similarities - torch.logsumexp(denominator_scores, dim=1)[:, None]
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

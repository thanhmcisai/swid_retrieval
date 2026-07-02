# -*- coding: utf-8 -*-
"""Metric-learning losses (lifted verbatim from smartwoodid_experiments_full.py:514-579)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, n_classes, embedding_dim=512, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.W = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels):
        W = F.normalize(self.W, dim=1)
        cos_theta = (embeddings @ W.T).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1)
        return F.cross_entropy(torch.cos(theta + self.m * one_hot) * self.s, labels)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        sim = (embeddings @ embeddings.T) / self.temperature
        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float() - torch.eye(B, device=sim.device)
        sim_masked = sim - 1e9 * torch.eye(B, device=sim.device)
        log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)
        return -(pos_mask * log_prob).sum(1).div(pos_mask.sum(1).clamp(min=1)).mean()


class ProtoNetLoss(nn.Module):
    """Episodic K-shot prototypes (v2: prototypes re-normalized → pure cosine)."""

    def __init__(self, k_shot=5):
        super().__init__()
        self.k_shot = k_shot

    def forward(self, embeddings, labels):
        classes = labels.unique()
        prototypes, queries, targets = [], [], []
        for ep_lbl, cls in enumerate(classes):
            mask = (labels == cls).nonzero(as_tuple=True)[0]
            k = min(self.k_shot, len(mask) - 1)
            if k < 1:
                continue
            prototypes.append(F.normalize(embeddings[mask[:k]].mean(0), dim=0))
            for idx in mask[k:]:
                queries.append(embeddings[idx]); targets.append(ep_lbl)
        if not queries:
            return embeddings.sum() * 0
        sims = torch.stack(queries) @ torch.stack(prototypes).T
        return F.cross_entropy(sims, torch.tensor(targets, device=sims.device))

    def parameters(self):
        return iter([])  # no learnable params


def build_loss(loss_name, n_classes, embedding_dim, cfg, device):
    if loss_name == "arcface":
        return ArcFaceLoss(n_classes, embedding_dim, cfg.arcface_s, cfg.arcface_m).to(device)
    if loss_name == "supcon":
        return SupConLoss(cfg.supcon_temp).to(device)
    if loss_name == "protonet":
        return ProtoNetLoss(cfg.protonet_k_shot).to(device)
    raise ValueError(f"Unknown loss: {loss_name}")

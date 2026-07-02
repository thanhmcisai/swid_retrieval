# -*- coding: utf-8 -*-
"""Research-suite heads (lifted from smartwoodid_experiments_full.py:5294-5361).

SCURDResidualHead is reused from ..scurd (byte-identical). Here we add the URD
projection head, the anchored-DINO head, and the B1 density gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scurd import SCURDResidualHead  # re-export (canonical)

__all__ = ["URDProjectionHead", "SCURDAnchoredDINOHead", "B1DensityGate", "SCURDResidualHead",
           "make_scurd_model"]


class B1DensityGate(nn.Module):
    def __init__(self, in_dim, n_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, n_experts),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class URDProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SCURDAnchoredDINOHead(nn.Module):
    def __init__(self, in_dim, beta=0.05):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, in_dim),
        )
        self.register_buffer("fixed_beta", torch.tensor(float(beta), dtype=torch.float32))

    def beta_value(self):
        return self.fixed_beta

    def forward(self, x):
        base = F.normalize(x, dim=1)
        delta = F.normalize(self.adapter(x), dim=1)
        return F.normalize(base + self.beta_value().to(x.device) * delta, dim=1)


def make_scurd_model(in_dim, head_type="residual", beta=0.1, learnable_beta=False, device="cpu"):
    """Build a SC-URD head (residual or anchored_dino) — matches monolith _scurd_make_model."""
    if head_type == "anchored_dino":
        b = 0.05 if not isinstance(beta, (int, float)) else float(beta)
        return SCURDAnchoredDINOHead(in_dim, beta=b).to(device)
    b = 0.1 if (learnable_beta or not isinstance(beta, (int, float))) else float(beta)
    return SCURDResidualHead(in_dim, out_dim=512, beta=b, learnable_beta=bool(learnable_beta)).to(device)

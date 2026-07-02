# -*- coding: utf-8 -*-
"""SC-URD residual adapter: model class, checkpoint loader, batched projection.

Lifted verbatim from variance_retrieval_evidence_colab.py:982-1036 so the
adapter and its forward pass match the validated SC-URD evaluation. Used here to
project the FULL 954-species SWI DINOv2 gallery through the adapter and store it
as `swi_pool`, keeping SC-URD aligned with the rest of the full gallery.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCURDResidualHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, beta=0.1, learnable_beta=False):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, out_dim),
        )
        if learnable_beta:
            beta = float(beta if isinstance(beta, (int, float)) else 0.1)
            beta = min(max(beta, 1e-4), 0.999)
            self.logit_beta = nn.Parameter(torch.tensor(np.log(beta / (1.0 - beta)), dtype=torch.float32))
        else:
            self.register_buffer("fixed_beta", torch.tensor(float(beta), dtype=torch.float32))
            self.logit_beta = None

    def beta_value(self):
        return torch.sigmoid(self.logit_beta) if self.logit_beta is not None else self.fixed_beta

    def forward(self, x):
        base = F.normalize(self.base(x), dim=1)
        delta = F.normalize(self.adapter(x), dim=1)
        beta = self.beta_value().to(x.device)
        return F.normalize(base + beta * delta, dim=1)


def load_scurd_model(ckpt_path, in_dim=768, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = int(ckpt.get("in_dim", in_dim))
    out_dim = int(ckpt.get("out_dim", 512))
    beta = ckpt.get("beta", 0.1)
    learnable = bool(ckpt.get("learnable_beta", False))
    model = SCURDResidualHead(in_dim, out_dim=out_dim,
                              beta=0.1 if learnable else float(beta),
                              learnable_beta=learnable).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def project_np(model, arr, device, batch_size=8192):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(arr), batch_size):
            x = torch.tensor(arr[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs, axis=0)

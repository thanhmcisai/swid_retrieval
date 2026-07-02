# -*- coding: utf-8 -*-
"""Model construction + checkpoint loading (NO training).

The package only EVALUATES, so every backbone is loaded from a pre-trained
checkpoint on Drive (or torch.hub / timm for the zero-shot ones). This mirrors the
"§7D — Load ALL models" block of final_metric_learning_cea_2026.py:1146-1595 but
strips out all the training short-circuits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms

from . import config


class EmbeddingModel(nn.Module):
    """Backbone + projection head → L2-normalized embedding (ArcFace/SupCon/Proto)."""

    def __init__(self, backbone_name="convnext_base", embedding_dim=512, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                           num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Linear(feat_dim, embedding_dim), nn.BatchNorm1d(embedding_dim))
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=1)


class CEClassifier(nn.Module):
    """Dual-mode: logits for classification, embedding for retrieval."""

    def __init__(self, backbone_name="convnext_base", n_classes=954, embedding_dim=512, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                           num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Linear(feat_dim, embedding_dim),
                                  nn.BatchNorm1d(embedding_dim), nn.ReLU())
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.classifier(self.head(self.backbone(x)))

    def get_embedding(self, x, normalize=True):
        feat = self.head(self.backbone(x))
        return F.normalize(feat, dim=1) if normalize else feat


def _load_metric(ckpt_path, backbone="convnext_base", embedding_dim=512, device="cpu"):
    model = EmbeddingModel(backbone, embedding_dim, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def _load_ce(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    n_classes = ckpt["metrics"]["n_classes"]
    model = CEClassifier("convnext_base", n_classes=n_classes, embedding_dim=512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def build_models(device=None, need_clip=True, need_dinov2=True):
    """Return dict of all eval models keyed by the embedding-cache tag stem.

    Keys: imagenet, arc (ArcFace-557), arc954, proto, ce (CE-Full), ce_narrow,
          clip, dinov2, and one per VARIANT tag (Var_ViTB_Arc, ...).
    Also returns clip_transform / dinov2_transform when requested.
    """
    device = device or config.resolve_device()
    models, transforms_out = {}, {}

    # Zero-shot ImageNet ConvNeXt-B (penultimate features)
    models["imagenet"] = timm.create_model("convnext_base", pretrained=True,
                                            num_classes=0, global_pool="avg").eval().to(device)

    # Trained metric models (loaded from Drive checkpoints)
    models["arc"], _ = _load_metric(config.CKPT_ARC_557, "convnext_base", 512, device)
    models["arc954"], _ = _load_metric(config.CKPT_ARC_954, "convnext_base", 512, device)
    models["proto"], _ = _load_metric(config.CKPT_PROTO, "convnext_base", 512, device)

    # CE classifiers (norm embedding + logits + raw)
    models["ce"], _ = _load_ce(config.CKPT_CE_FULL, device)
    models["ce_narrow"], _ = _load_ce(config.CKPT_CE_NARROW, device)

    # Backbone × loss variants
    for backbone, loss, tag in config.VARIANTS:
        ckpt_path = config.CKPT_DIR / f"metric_{backbone}_{loss}.pt"
        models[tag], _ = _load_metric(ckpt_path, backbone, 512, device)

    # Zero-shot foundation models
    if need_clip:
        import clip
        model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        model_clip.eval()
        models["clip"] = model_clip
        transforms_out["clip"] = preprocess_clip
    if need_dinov2:
        models["dinov2"] = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
        transforms_out["dinov2"] = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    print(f"✅ Loaded models: {sorted(models.keys())}")
    return models, transforms_out

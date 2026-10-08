"""Compact, from-scratch multi-scale texture encoder for the gallery study.

This is a research candidate, not a validated wood-anatomy detector. Its
multi-scale token pooling tests whether repeated local texture helps retrieval.
"""

import torch
from torch import nn
from torch.nn import functional as F


class TextureBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3,
                                   groups=channels, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.reduce = nn.Conv2d(2 * channels, channels, 1)
        self.scale = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, x):
        residual = self.reduce(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        return x + self.scale[None, :, None, None] * residual


class WoodPatternNet(nn.Module):
    """Learn local patterns at three resolutions and pool their distribution.

    GroupNorm and zero-dropout attention keep the two-pass gradient replay
    deterministic without relying on running batch statistics.
    """

    num_features = 384

    def __init__(self, use_attention=True, use_multiscale=True):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.use_multiscale = bool(use_multiscale)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48), nn.GELU(),
            TextureBlock(48), TextureBlock(48),
        )
        self.stage2 = self._stage(48, 96, 2)
        self.stage3 = self._stage(96, 192, 3)
        self.stage4 = self._stage(192, 256, 2)
        self.token_projections = nn.ModuleList(
            nn.Conv2d(channels, 128, 1) for channels in (96, 192, 256))
        self.scale_embedding = nn.Parameter(torch.zeros(3, 128))
        self.grid = nn.AdaptiveAvgPool2d((4, 4))
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0,
                                       activation="gelu", batch_first=True),
            num_layers=2, enable_nested_tensor=False)
        self.token_gate = nn.Linear(128, 1)
        self.output = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384))

    @staticmethod
    def _stage(in_channels, out_channels, blocks):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU(),
            *(TextureBlock(out_channels) for _ in range(blocks)),
        )

    def forward(self, x):
        x = self.stem(x)
        second = self.stage2(x)
        third = self.stage3(second)
        fourth = self.stage4(third)
        maps = (second, third, fourth)
        scales = range(3) if self.use_multiscale else (2,)
        tokens = []
        for index in scales:
            projected = self.grid(self.token_projections[index](maps[index]))
            token = projected.flatten(2).transpose(1, 2)
            tokens.append(token + self.scale_embedding[index])
        tokens = torch.cat(tokens, dim=1)
        if self.use_attention:
            tokens = self.context(tokens)
            weights = self.token_gate(tokens).softmax(dim=1)
            pooled = (weights * tokens).sum(dim=1)
        else:
            pooled = tokens.mean(dim=1)
        mean = tokens.mean(dim=1)
        dispersion = tokens.var(dim=1, unbiased=False).add(1e-6).sqrt()
        return self.output(torch.cat((pooled, mean, dispersion), dim=1))

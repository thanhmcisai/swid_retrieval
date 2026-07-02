# -*- coding: utf-8 -*-
"""Training Config (lifted from smartwoodid_experiments_full.py:96-139) + the full
§I.5 research grid + staged flags. Paths point at the eval package's CKPT_DIR /
RESEARCH_DIR so trained checkpoints match the names the eval loads."""

import os
from dataclasses import dataclass, field

from .. import config as C


@dataclass
class Config:
    backbone: str = "convnext_base"
    embedding_dim: int = 512
    pretrained: bool = True
    loss: str = "arcface"          # arcface | supcon | protonet
    arcface_s: float = 64.0
    arcface_m: float = 0.5
    supcon_temp: float = 0.07
    protonet_k_shot: int = 5
    epochs: int = 40
    lr: float = 1e-3
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    P: int = 32
    K: int = 4
    batch_size_ce: int = 128
    batch_size_val: int = 128
    img_size: int = 224
    scales: str = "all"
    seed: int = 42
    num_workers: int = field(default_factory=C.num_workers)

    @property
    def checkpoint_path(self):
        return C.CKPT_DIR

    @property
    def run_name(self):
        return f"metric_{self.backbone}_{self.loss}"


# Backbone × loss ablation variants (matches eval config.VARIANTS).
VARIANTS = [(bb, loss, tag) for (bb, loss, tag) in C.VARIANTS]

# ── §I.5 research grid ───────────────────────────────────────────────────────
RESEARCH_CACHE_VERSION = os.environ.get("RESEARCH_CACHE_VERSION", "v2")
SC_URD_CACHE_VERSION = os.environ.get("SC_URD_CACHE_VERSION", "v2")
URD_EPOCH_ABLATIONS = [10, 20, 40]
SC_URD_BETAS = [0.1, 0.3, "learnable"]
SC_URD_TRAIN_EPOCH_ABLATIONS = [10, 15, 20, 40]
SC_URD_DINO_ANCHOR_BETAS = [0.05, 0.1]
SC_URD_DINO_TEACHER_ETAS = [0.1, 0.3]
SC_URD_DEFAULT_EPOCHS = 10
URD_DEFAULT_EPOCHS = 10
# episodic sampler config (monolith)
N_WAY = 16
K_SUPPORT = 5
Q_QUERY = 4
Q_OOD = 32
EPISODES_PER_EPOCH = 500
RS_TAU = 0.07
RS_TOP_M = 50
SCURD_TRAIN_LR = 1e-3
SCURD_WEIGHT_DECAY = 1e-4
# OOD-consistency (entropy-rejection) hyperparameters — monolith L5027-5028.
SC_URD_GAMMA_OOD = 0.2
SC_URD_ENTROPY_MARGIN_FRAC = 0.8

# ── staged flags / FAST mode ─────────────────────────────────────────────────
FAST = os.environ.get("FAST", "0") == "1"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
RUN_TRAIN_MAIN = os.environ.get("RUN_TRAIN_MAIN", "1") == "1"
RUN_TRAIN_VARIANTS = os.environ.get("RUN_TRAIN_VARIANTS", "1") == "1"
RUN_TRAIN_RESEARCH = os.environ.get("RUN_TRAIN_RESEARCH", "1") == "1"
RUN_TRAIN_PHASE2 = os.environ.get("RUN_TRAIN_PHASE2", "1") == "1"
RUN_TRAIN_SCURD_SEEDS = os.environ.get("RUN_TRAIN_SCURD_SEEDS", "1") == "1"

if FAST:  # tiny smoke-test profile
    SC_URD_DEFAULT_EPOCHS = URD_DEFAULT_EPOCHS = 2
    EPISODES_PER_EPOCH = 30
    URD_EPOCH_ABLATIONS = [2]
    SC_URD_TRAIN_EPOCH_ABLATIONS = [2]


def base_cfg():
    c = Config()
    if FAST:
        c.epochs = 2
    return c

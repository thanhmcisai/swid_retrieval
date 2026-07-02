# -*- coding: utf-8 -*-
"""Central paths, env vars, gallery scope, and SC-URD hyperparameters.

Single source of truth for everything the flat scripts each re-derived. All values
are env-overridable so the same package runs on Colab (Drive paths) or locally.
The env names and defaults match variance_retrieval_evidence_colab.py:47-96 so the
validated engine, when exec()'d by the orchestrator, sees an identical contract.
"""

import os
from pathlib import Path

# ── Roots ────────────────────────────────────────────────────────────────
ROOT_PATH = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS"))
DATASETS_DIR = ROOT_PATH / "datasets"
PSI_DIR = DATASETS_DIR / "PSI_smartwoodid"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT_PATH / "results" / "paper_reframe"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT_PATH / "checkpoints"))
RESEARCH_DIR = RESULTS_DIR / "research_directions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = ROOT_PATH / "swi_manifest.json"
_split_default = Path(os.environ.get("SPLIT_CSV", ROOT_PATH / "smartwoodid_split.csv"))
SPLIT_CSV = _split_default if _split_default.exists() else PSI_DIR / "smartwoodid_split.csv"
ID_SPECIES_CSV = ROOT_PATH / "ID_species_public.csv"
OOD_SPECIES_CSV = ROOT_PATH / "OOD_species_public.csv"
ID_IMAGES_CSV = ROOT_PATH / "ID_images_expanded.csv"
OOD_IMAGES_CSV = ROOT_PATH / "OOD_images_expanded.csv"

# ── Embedding caches ─────────────────────────────────────────────────────
# Old (meta-test only, ~317 species). Source of the unchanged ID/OOD query side.
SOURCE_EMB_CACHE_NAME = os.environ.get("SOURCE_EMB_CACHE_NAME", "embedding_cache_v3.npz")
SOURCE_EMB_CACHE_PATH = ROOT_PATH / SOURCE_EMB_CACHE_NAME
# New (full 954-species SWI gallery). What downstream eval must read.
FULL954_CACHE_NAME = os.environ.get("FULL954_CACHE_NAME", "embedding_cache_full954_v3.npz")
FULL954_CACHE_PATH = ROOT_PATH / FULL954_CACHE_NAME
FULL954_META_PATH = ROOT_PATH / (FULL954_CACHE_NAME.replace(".npz", "_meta.json"))

EXP4_CACHE_NAME = os.environ.get("EXP4_CACHE_NAME", "exp4_embedding_cache_v3.npz")  # VN26 — must NOT be the 954 cache

# ── Gallery scope ────────────────────────────────────────────────────────
GALLERY_SCOPE = os.environ.get("GALLERY_SCOPE", "full_swi").strip().lower()
MIN_FULL_GALLERY_SPECIES = int(os.environ.get("MIN_FULL_GALLERY_SPECIES", "900"))
FULL_SCOPES = {"full", "full_swi", "all_swi", "954", "954_swi"}
ID_SCOPES = {"id", "id_only", "public_id", "24", "24_id"}
# ce_train: full 954-species gallery restricted to CE-Full's exact train images
# (fairness control: retrieval gets no reference image CE-Full did not learn from).
CE_TRAIN_SCOPES = {"ce_train", "ce_train_only", "cetrain"}
# Scopes that span all 954 cache rows and apply a boolean mask (vs the id-filter).
FULL_ROW_SCOPES = FULL_SCOPES | CE_TRAIN_SCOPES

# ── Checkpoints (loaded, never trained, by this package) ─────────────────
CKPT_CE_FULL = CKPT_DIR / os.environ.get("CKPT_CE_FULL", "ce_954sp_convnext_base.pt")
CKPT_CE_NARROW = CKPT_DIR / os.environ.get("CKPT_CE_NARROW", "ce_narrow_557sp_convnext_base.pt")
CKPT_ARC_557 = CKPT_DIR / os.environ.get("CKPT_ARC_557", "metric_convnext_base_arcface.pt")
CKPT_ARC_954 = CKPT_DIR / os.environ.get("CKPT_ARC_954", "arcface_full954.pt")
CKPT_PROTO = CKPT_DIR / os.environ.get("CKPT_PROTO", "metric_convnext_base_protonet_v2.pt")

# Backbone × loss variants (tag → (backbone_name, checkpoint filename))
VARIANTS = [
    ("vit_base_patch16_224", "arcface", "Var_ViTB_Arc"),
    ("resnet50", "arcface", "Var_RN50_Arc"),
    ("convnext_base", "supcon", "Var_CvNxt_SupCon"),
]

# ── SC-URD (residual adapter on frozen DINOv2) ───────────────────────────
SC_URD_CACHE_VERSION = os.environ.get("SC_URD_CACHE_VERSION", "v2")
RESEARCH_CACHE_VERSION = os.environ.get("RESEARCH_CACHE_VERSION", "v2")
SCURD_PROJ_CACHE = RESEARCH_DIR / os.environ.get(
    "SCURD_PROJ_CACHE_NAME", "sc_urd_eval_embeddings_scurd_r01_e20_recomputed_v2.npz")
SCURD_MAIN_CKPT = RESEARCH_DIR / os.environ.get(
    "SCURD_MAIN_CKPT_NAME", f"sc_urd_checkpoint_scurd_r01_e20_{SC_URD_CACHE_VERSION}.pt")
SCURD_MODE = os.environ.get("SCURD_MAIN_MODE", "centered").strip() or "centered"
SCURD_TAU = float(os.environ.get("SCURD_TAU", "0.07"))
SCURD_TOP_M = int(os.environ.get("SCURD_TOP_M", "50"))
SCURD_BETA = float(os.environ.get("SCURD_BETA", "0.1"))
SCURD_TRAIN_EPOCHS = int(os.environ.get("SCURD_TRAIN_EPOCHS", "20"))
SCURD_TRAIN_EPISODES = int(os.environ.get("SCURD_TRAIN_EPISODES", "500"))
SCURD_TRAIN_LAMBDA_CONS = float(os.environ.get("SCURD_TRAIN_LAMBDA_CONS", "0.5"))
SCURD_TRAIN_LR = float(os.environ.get("SCURD_TRAIN_LR", "1e-3"))
SCURD_WEIGHT_DECAY = float(os.environ.get("SCURD_WEIGHT_DECAY", "1e-4"))
SCURD_N_WAY = int(os.environ.get("SCURD_N_WAY", "16"))
SCURD_K_SUPPORT = int(os.environ.get("SCURD_K_SUPPORT", "5"))
SCURD_Q_QUERY = int(os.environ.get("SCURD_Q_QUERY", "4"))

# ── Build / run flags ────────────────────────────────────────────────────
RUN_BUILD_FULL954 = os.environ.get("RUN_BUILD_FULL954", "1") == "1"
FORCE_REBUILD_FULL954 = os.environ.get("FORCE_REBUILD_FULL954", "0") == "1"
SAVE_PARTIAL = os.environ.get("SAVE_PARTIAL", "1") == "1"
RUN_REVIEW_TAXONOMY_FULL_GALLERY = os.environ.get("RUN_REVIEW_TAXONOMY_FULL_GALLERY", "1") == "1"
RUN_REVIEW_TAXONOMY_DEPLOYMENT = os.environ.get("RUN_REVIEW_TAXONOMY_DEPLOYMENT", "1") == "1"
RUN_EDGE_PROXY = os.environ.get("RUN_EDGE_PROXY", "1") == "1"

DEVICE = os.environ.get("DEVICE", "cuda")
PARTIAL_CACHE_PATH = ROOT_PATH / os.environ.get("PARTIAL_CACHE_NAME", "embedding_cache_full954_v3.partial.npz")
SEED = int(os.environ.get("SEED", "42"))


def is_full_scope(scope=None):
    return (scope or GALLERY_SCOPE) in FULL_SCOPES


def resolve_device():
    if DEVICE == "cuda":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return DEVICE


def num_workers():
    import multiprocessing
    return int(os.environ.get("NUM_WORKERS", str(min(8, multiprocessing.cpu_count()))))


def configure_runtime(seed=None, verbose=True):
    """Set deterministic runtime knobs used by the original Colab notebook."""
    seed = SEED if seed is None else int(seed)
    try:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if verbose:
            print(
                f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} "
                f"| Device: {resolve_device()} | Workers: {num_workers()}"
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"⚠️  runtime deterministic setup skipped: {exc}")

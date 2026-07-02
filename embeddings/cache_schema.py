# -*- coding: utf-8 -*-
"""Embedding-cache key schema.

Mirrors _EMB_VAR_NAMES (final_metric_learning_cea_2026.py:1841-1859). The
full-954 builder re-extracts only the SWI-gallery side; everything on the
ID/OOD query side is copied verbatim from the source meta-test cache.
"""

from .. import config

VARIANT_TAGS = [tag for _, _, tag in config.VARIANTS]

# SWI-gallery keys that MUST be re-extracted over all 954 species.
SWI_BASE_KEYS = [
    "embs_swi_imagenet", "embs_swi_arc", "embs_swi_arc954", "embs_swi_proto",
    "embs_swi_ce_full_norm", "logits_swi_ce_full", "embs_swi_ce_full_raw",
    "embs_swi_ce_narrow_norm", "logits_swi_ce_narrow", "embs_swi_ce_narrow_raw",
    "embs_swi_clip", "labels_swi_clip",
    "embs_swi_dinov2", "labels_swi_dinov2",
]
SWI_VARIANT_KEYS = [f"embs_swi_{t}" for t in VARIANT_TAGS]
SWI_KEYS = SWI_BASE_KEYS + SWI_VARIANT_KEYS

# Query-side prefixes copied unchanged from the source cache (independent of
# the SWI gallery scope: public-ID and public-OOD are external images).
COPY_PREFIXES = (
    "embs_id_", "labels_id_", "logits_id_",
    "embs_ood_", "labels_ood_", "logits_ood_",
)


def is_copy_key(key):
    return key.startswith(COPY_PREFIXES)


def build_meta(swi_images, swi_species, copied_keys, swi_keys, source_cache):
    """Provenance sidecar for the full-954 cache."""
    return {
        "artifact": "embedding_cache_full954",
        "gallery_scope": "full_swi",
        "swi_source_splits": ["meta-train", "meta-val", "meta-test"],
        "swi_full_images": int(swi_images),
        "swi_full_species": int(swi_species),
        "swi_keys_recomputed": sorted(swi_keys),
        "id_ood_keys_copied": sorted(copied_keys),
        "id_ood_copied_from": str(source_cache),
        "transforms": {
            "eval_size": 224,
            "dinov2_size": 518,
            "clip_model": "ViT-B/32",
            "dinov2_model": "facebookresearch/dinov2:dinov2_vitb14",
        },
    }

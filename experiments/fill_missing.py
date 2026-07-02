# -*- coding: utf-8 -*-
"""§K/§L — complete ablation matrix (fill-missing).

The monolith filled Exp1A/1B/2/3-Part1/4A for every remaining method
(smartwoodid_experiments_full.py §K L7534, §L L7928-8415) because its experiment
blocks were written incrementally. In this package `registry.build_M` already
constructs the COMPLETE method registry (ImageNet, CLIP, DINOv2, CE-Full/Narrow,
ArcFace-557/954, ProtoNet, ArcFace-ViT/RN50, SupCon, ArcFace+PDA/CORAL/αQE, Fusion,
SC-URD), so this module is a completeness pass: it computes the cross-method
ablation matrix (Exp1A prototype + R@1, OOD Dist AUROC/FPR95, RQ4 SWI→VN26) for
EVERY registry method and writes it once.

Cache-only (no GPU/training): operates on the embeddings already in M. Opt-in via
RUN_FILL_MISSING=1 in the orchestrator. Idempotent — skips when the output already
covers all current methods (mirrors the monolith `_l_skip`).

Output → fill_missing_ablation.json.
"""

import json
import os

import numpy as np

from . import registry as R
from .ood_baselines import _cos_dist, _auroc, _fpr95


def _exp1a(md, labels_id):
    """Cross-domain public-ID: prototype-macro + R@1 against the active gallery."""
    gal_e, gal_l = md["gal"]
    proto, _ = R.prototype_classify_per_species(md["id"], labels_id, gal_e, gal_l)
    r1, _ = R.recall_at_1_per_species(md["id"], labels_id, gal_e, gal_l)
    return float(proto), float(r1)


def _ood(md, test_mask):
    if md.get("ood") is None:
        return None, None
    gal_e, _ = md["gal"]
    sid = _cos_dist(md["id"], gal_e)
    sod = _cos_dist(md["ood"], gal_e)[test_mask]
    return float(_auroc(sid, sod)), float(_fpr95(sid, sod))


def _rq4_swi_to_vn26(md):
    """SWI_pool gallery → VN26_all query R@1 (Exp4A), when both pools are cached."""
    swi_pool = (md.get("swi_scales") or {}).get("pool")
    vn26_all = (md.get("vn26_mags") or {}).get("all")
    if swi_pool is None or vn26_all is None:
        return None
    q_e, q_l = vn26_all
    g_e, g_l = swi_pool
    macro, _ = R.recall_at_1_per_species(q_e, q_l, g_e, g_l)
    return float(macro)


def run(M, ctx, out_dir):
    """M, ctx from registry.build_M. Writes fill_missing_ablation.json."""
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fill_missing_ablation.json")

    # Skip-if-complete: if every current method is already present, do nothing.
    if os.path.exists(path):
        try:
            prev = json.load(open(path)).get("matrix", {})
            if set(prev) >= set(M):
                print(f"  fill_missing: all {len(M)} methods already present in "
                      f"{os.path.basename(path)} → skip")
                return prev
        except Exception:  # noqa: BLE001
            pass

    labels_id = ctx["labels_id"]
    labels_ood = ctx.get("labels_ood")
    test_mask = R.ood_test_mask(labels_ood) if labels_ood is not None else None

    matrix = {}
    for name, md in M.items():
        try:
            proto, r1 = _exp1a(md, labels_id)
            au, fp = (_ood(md, test_mask) if test_mask is not None else (None, None))
            matrix[name] = {
                "exp1a_prototype": proto,
                "exp1a_R1": r1,
                "ood_auroc": au,
                "ood_fpr95": fp,
                "rq4_swi_to_vn26_R1": _rq4_swi_to_vn26(md),
                "has_ood": md.get("ood") is not None,
            }
        except Exception as e:  # noqa: BLE001
            print(f"  fill_missing: {name} failed ({e}); skipping that method.")

    out = {"gallery_scope": ctx.get("scope"), "n_methods": len(matrix), "matrix": matrix}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    print(f"  ✅ fill_missing ablation matrix: {len(matrix)} methods → {os.path.basename(path)}")
    return matrix

# -*- coding: utf-8 -*-
"""RQ5 extra — SC-URD inference/calibration sweep + (optional) per-epoch projection.

Complements the engine's 6 memory-mode rows with a prototype-mix-alpha sweep and a
score-level hybrid with DINOv2/Fusion (monolith _scurd_inference_calibration_suite
L4708). Per-epoch URD/SC-URD projection ablation (L4537) runs only if the e10/e20/e40
checkpoints exist under research_directions. Cache-only (operates on the deployment
gallery embeddings in M). Output → rq5_inference_calibration.json.
"""

import json
import os

import numpy as np

from .. import config
from . import registry as R


def _e1a(md, gal_e, gal_l, mode, alpha=None):
    if md.get("scurd_mode") is None:
        return R.method_retrieval_eval(md, md["id"], md["_ql"], gal_e, gal_l)["mean"]
    return R.scurd_retrieval_eval(md["id"], md["_ql"], gal_e, gal_l, mode=mode, prototype_mix_alpha=alpha)["mean"]


def run(M, ctx, out_dir):
    """Writes rq5_inference_calibration.json. Cache-only."""
    if "SC-URD" not in M:
        print("  rq5_extra: SC-URD not in registry; skipping.")
        return None
    labels_id = ctx["labels_id"]
    sc = dict(M["SC-URD"]); sc["_ql"] = labels_id
    gal_e, gal_l = sc["gal"]
    rows = []

    # 1) memory modes (E1A retrieval) — mirrors engine, kept for the ablation table.
    for mode in sorted(R.SCURD_VALID_MODES):
        rows.append({"ablation": "memory_mode", "memory_mode": mode,
                     "exp1a_mean": _e1a(sc, gal_e, gal_l, mode)})

    # 2) prototype-mix alpha sweep.
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rows.append({"ablation": "prototype_mix_alpha", "memory_mode": "prototype_mix",
                     "alpha": a, "exp1a_mean": _e1a(sc, gal_e, gal_l, "prototype_mix", alpha=a)})

    # 3) score-level hybrid with DINOv2 / Fusion (blend nearest-centroid sims).
    def _proto_sims(md):
        q, g = R._norm(md["id"]), R._norm(md["gal"][0])
        gl = np.asarray(md["gal"][1]); classes = np.array(sorted(set(gl.tolist())))
        cent = R._norm(np.stack([g[gl == c].mean(0) for c in classes]))
        return (q @ cent.T), classes
    sc_scores, sc_classes = R.scurd_class_scores(sc["id"], gal_e, gal_l, mode=config.SCURD_MODE)
    for base in ["DINOv2", "Fusion"]:
        if base not in M:
            continue
        bs, bc = _proto_sims(M[base])
        if list(bc) != list(sc_classes):
            continue
        for lam in [0.25, 0.5, 0.75]:
            blend = lam * sc_scores + (1 - lam) * bs
            preds = sc_classes[blend.argmax(1)]
            macro, _ = R._macro_from_preds(preds, labels_id)
            rows.append({"ablation": "score_hybrid", "hybrid_base": base, "lambda": lam, "exp1a_mean": macro})

    # 4) per-epoch projection ablation (only if checkpoints exist).
    proj_rows = []
    try:
        from ..scurd import load_scurd_model, project_np
        dev = ctx["device"]
        dino_id = ctx["emb"]["embs_id_dinov2"]
        dino_gal = ctx["emb"]["embs_swi_dinov2"][ctx["gallery_mask"]]
        for ckpt in sorted(config.RESEARCH_DIR.glob("sc_urd_checkpoint_*_e*_v2.pt")):
            try:
                model, _ = load_scurd_model(ckpt, in_dim=dino_id.shape[1], device=dev)
                md = {"id": project_np(model, dino_id, dev), "gal": (project_np(model, dino_gal, dev), gal_l),
                      "scurd_mode": config.SCURD_MODE, "_ql": labels_id}
                proj_rows.append({"checkpoint": ckpt.name,
                                  "exp1a_mean": _e1a(md, md["gal"][0], gal_l, config.SCURD_MODE)})
            except Exception as e:  # noqa: BLE001
                print(f"  projection ablation: {ckpt.name} failed ({e})")
    except Exception:
        pass

    out = {"gallery_scope": ctx["scope"], "inference_calibration": rows, "projection_ablation": proj_rows}
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "rq5_inference_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path} ({len(rows)} calibration rows, {len(proj_rows)} projection rows)")
    return out

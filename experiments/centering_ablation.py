# -*- coding: utf-8 -*-
"""Backbone-agnostic centering ablation (the named SC-URD novelty).

Shows the gallery-conditioned scoring layer (centered / logmeanexp / prototype-mix
modes) helps ACROSS backbones, not just on DINOv2 — i.e. it is a training-free,
backbone-agnostic contribution. For each base embedding and each scoring mode,
reports Public-ID R@1; OOD AUROC is reported once per base (mode-independent
nearest-gallery distance).

MEMORY-SAFE: does NOT call build_M (which would materialize every method's full
954-gallery at once). Instead it loads ONE base at a time directly from the cache
and CHUNKS the query×gallery matmul, so the matched-954 scope (≈176k gallery
images) does not blow up RAM. Each gallery scope is wrapped independently so the
cheap deployment-24 result is always written even if the heavy 954 scope is
skipped. Output: centering_ablation.{csv,json}.

    python -m swid_retrieval.experiments.centering_ablation [run_dir]
"""

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

from .. import config
from ..data import canonical_label
from ..gallery import build_gallery_mask
from . import registry as R
from .ood_baselines import _auroc

# id/swi/ood emb keys per base; Fusion = L2(concat(L2 arc, L2 dinov2)).
BASES = {
    "DINOv2": {"id": "embs_id_dinov2", "swi": "embs_swi_dinov2", "ood": "embs_ood_dinov2"},
    "ArcFace-557": {"id": "embs_id_arc", "swi": "embs_swi_arc", "ood": "embs_ood_arc"},
    "CE-Full": {"id": "embs_id_ce_full_norm", "swi": "embs_swi_ce_full_norm", "ood": "embs_ood_ce_full_norm"},
    "Fusion": {"fuse": True},
}
SCOPES = ["id_only", "full_swi"]
CHUNK = int(os.environ.get("CENTERING_CHUNK", "256"))


def _run_dir(argv):
    if len(argv) > 1 and argv[1] and not str(argv[1]).endswith(".tex"):
        return Path(argv[1])
    stamp = os.environ.get("FULL954_RUN_STAMP", "overnight")
    return Path(os.environ.get(
        "FULL954_RESULTS_DIR", config.ROOT_PATH / "results" / f"paper_reframe_full954_{stamp}"))


def _emb(cache, spec, which):
    if spec.get("fuse"):
        a = {"id": "embs_id_arc", "swi": "embs_swi_arc", "ood": "embs_ood_arc"}[which]
        d = {"id": "embs_id_dinov2", "swi": "embs_swi_dinov2", "ood": "embs_ood_dinov2"}[which]
        return R._fuse(np.asarray(cache[a]), np.asarray(cache[d]))
    return np.asarray(cache[spec[which]])


def _r1_chunked(q, ql, g, gl, mode):
    """Public-ID R@1 with a given scoring mode, chunked over query rows (memory-safe)."""
    preds = []
    for i in range(0, len(q), CHUNK):
        scores, classes = R.scurd_class_scores(q[i:i + CHUNK], g, gl, mode=mode)
        preds.append(np.asarray(classes)[scores.argmax(axis=1)])
    preds = np.concatenate(preds) if preds else np.array([])
    macro, _ = R._macro_from_preds(preds, ql)
    return float(macro)


def _cosdist_chunked(q, gn):
    """1 - max cosine sim to gallery, chunked. gn must be pre-L2-normalized gallery."""
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), CHUNK):
        out[i:i + CHUNK] = 1.0 - (R._norm(q[i:i + CHUNK]) @ gn.T).max(axis=1)
    return out


def _run_scope(scope, cache, labels_id, labels_swi, ce_mask, test_mask, modes, rows, deltas):
    mask, gal_labels, _ = build_gallery_mask(
        labels_swi, set(labels_id.tolist()), scope=scope, ce_train_mask=ce_mask)
    for base, spec in BASES.items():
        try:
            g = _emb(cache, spec, "swi")[mask]
            qid = _emb(cache, spec, "id")
            qood = _emb(cache, spec, "ood")
            gn = R._norm(g)
            au = float(_auroc(_cosdist_chunked(qid, gn), _cosdist_chunked(qood, gn)[test_mask]))
            r1 = {}
            for mode in modes:
                r1[mode] = _r1_chunked(qid, labels_id, g, gal_labels, mode)
                rows.append({"scope": scope, "base": base, "memory_mode": mode,
                             "r1": r1[mode], "ood_auroc": au})
            if "raw" in r1 and "centered" in r1:
                deltas.append({"scope": scope, "base": base, "r1_raw": r1["raw"],
                               "r1_centered": r1["centered"],
                               "delta_centered_minus_raw": r1["centered"] - r1["raw"],
                               "ood_auroc": au})
            print(f"  [{scope}] {base}: raw={r1.get('raw'):.3f} centered={r1.get('centered'):.3f} "
                  f"OOD_AUROC={au:.3f}")
            del g, qid, qood, gn
            gc.collect()
        except Exception as e:  # noqa: BLE001
            print(f"  [{scope}] {base} failed ({type(e).__name__}: {e}); skipping.")


def run(out_dir):
    cache = np.load(config.FULL954_CACHE_PATH, allow_pickle=False)
    labels_id = np.array([canonical_label(x) for x in cache["labels_id_dinov2"]])
    labels_swi = np.array([canonical_label(x) for x in cache["labels_swi_dinov2"]])
    labels_ood = np.array([canonical_label(x) for x in cache["labels_ood_dinov2"]])
    ce_mask = cache["swi_in_ce_train"] if "swi_in_ce_train" in cache.files else None
    test_mask = R.ood_test_mask(labels_ood)
    modes = sorted(R.SCURD_VALID_MODES)

    rows, deltas = [], []
    for scope in SCOPES:
        try:
            _run_scope(scope, cache, labels_id, labels_swi, ce_mask, test_mask, modes, rows, deltas)
        except Exception as e:  # noqa: BLE001
            print(f"  centering_ablation: scope {scope} failed ({type(e).__name__}: {e}); continuing.")
    if not rows:
        print("  centering_ablation: nothing computed.")
        return None

    out = {"bases": list(BASES), "modes": modes, "grid": rows, "centering_delta": deltas}
    os.makedirs(str(out_dir), exist_ok=True)
    with open(os.path.join(str(out_dir), "centering_ablation.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(str(out_dir), "centering_ablation.csv"), index=False)
    except Exception as e:  # noqa: BLE001
        print(f"  centering_ablation: CSV skipped ({e})")
    pos = sum(1 for d in deltas if d["delta_centered_minus_raw"] > 0)
    print(f"  ✅ centering_ablation: {len(rows)} rows; centering helped {pos}/{len(deltas)} base×scope cells")
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    return run(_run_dir(argv) / "native_experiments")


if __name__ == "__main__":
    main(sys.argv)

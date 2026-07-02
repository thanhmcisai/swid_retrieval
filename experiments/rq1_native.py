# -*- coding: utf-8 -*-
"""RQ1 fair-comparison helpers: prototype (anchor-matched) + native CE.

The fair closed-set-vs-retrieval comparison needs, per method, at the full-954
gallery:
  - PROTOTYPE macro-top-1: one L2-normalized centroid per species (954 anchors),
    the anchor-matched counterpart of CE's one weight vector per class. This is
    the fair paradigm headline.
  - NATIVE CE macro accuracy: 954-way softmax argmax. Gallery-INDEPENDENT, so it
    is identical to the previously published value (0.128); recomputed here from
    the cached logits for self-containedness, with a carry-forward cross-check.

These are standalone (no torch needed for prototype/carry-forward) so the
validated variance script can import them, or they can be called directly.
"""

import json
from pathlib import Path

import numpy as np


def _canon(s):
    return str(s).replace(" ", "_").lower().strip()


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _macro(preds, labels):
    preds, labels = np.asarray(preds), np.asarray(labels)
    per_sp = {sp: float((preds[labels == sp] == sp).mean()) for sp in sorted(np.unique(labels))}
    return float(np.mean(list(per_sp.values()))), per_sp


def prototype_macro_top1(q_emb, q_labels, g_emb, g_labels, centered=False):
    """Nearest class-centroid macro-top-1 over the gallery species (954).

    One centroid per gallery species = the anchor-matched analogue of CE's
    one-weight-per-class. `centered` mirrors SC-URD gallery-mean centering so the
    prototype metric is computed in the same space the method is scored in.
    """
    q = np.asarray(q_emb, dtype=np.float32)
    g = np.asarray(g_emb, dtype=np.float32)
    if centered:
        mu = g.mean(axis=0, keepdims=True)
        q, g = q - mu, g - mu
    q, g = _norm(q), _norm(g)
    gl = np.asarray(g_labels)
    classes = np.array(sorted(set(gl.tolist())))
    centroids = _norm(np.stack([g[gl == c].mean(axis=0) for c in classes]))
    preds = classes[(q @ centroids.T).argmax(axis=1)]
    macro, per_sp = _macro(preds, q_labels)
    return {"mean": macro, "per_species": per_sp, "n_centroids": int(len(classes))}


def native_ce_macro(logits_id, ce_species_list, q_labels):
    """954-way native CE accuracy, macro per query species (gallery-independent)."""
    species = [_canon(s) for s in ce_species_list]
    preds = np.array([species[i] for i in np.asarray(logits_id).argmax(axis=1)])
    macro, per_sp = _macro(preds, [_canon(x) for x in q_labels])
    return {"mean": macro, "per_species": per_sp}


def load_ce_species_list(ce_ckpt_path):
    """Read the ordered 954-class species list from the CE-Full checkpoint."""
    import torch
    ckpt = torch.load(ce_ckpt_path, map_location="cpu")
    return ckpt.get("ce_species_list")


def carry_forward_native(source_results_dir):
    """Carry the gallery-independent native accuracy forward from the original
    rq1_paradigm.json (a list of {method, native, ...}). Exact, since native CE
    does not depend on the retrieval gallery scope."""
    path = Path(source_results_dir) / "rq1_paradigm.json"
    out = {}
    if not path.exists():
        return out
    try:
        obj = json.load(open(path))
    except Exception:
        return out
    block = obj.get("paradigm_comparison", obj)
    if isinstance(block, list):
        for d in block:
            if isinstance(d, dict) and isinstance(d.get("native"), (int, float)):
                out[d.get("method")] = float(d["native"])
    return out

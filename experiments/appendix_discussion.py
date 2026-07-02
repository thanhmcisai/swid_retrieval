# -*- coding: utf-8 -*-
"""Appendix + discussion evidence (cache-only parts).

Lifts the cache-only pieces of run_appendix_evidence (L5072) and run_discussion_analysis
(L6681): all-pairs RQ1 Wilcoxon, retrieval-quality diagnostics (Hit@k / Purity@5 /
mean-rank / similarity-gap), and RQ1 per-species hard cases / largest gains. The
density-gate B1 path (needs meta-val embeddings, GPU) is intentionally omitted here.
Output → appendix_evidence.json, discussion_evidence.json.
"""

import json
import os

import numpy as np

from . import registry as R


def _retrieval_quality(md, q_labels):
    """Hit@1/5/10, Purity@5, mean first-correct rank, similarity gap (per method)."""
    q = R._norm(md["id"]); g = R._norm(md["gal"][0])
    ql = np.asarray(q_labels); gl = np.asarray(md["gal"][1])
    sims = q @ g.T
    order = np.argsort(-sims, axis=1)
    ranked = gl[order]
    correct = ranked == ql[:, None]
    out = {}
    for k in (1, 5, 10):
        kk = min(k, ranked.shape[1])
        out[f"Hit@{k}"] = float(correct[:, :kk].any(1).mean())
    out["Purity@5"] = float(correct[:, :min(5, ranked.shape[1])].mean())
    first = np.argmax(correct, axis=1)
    first[~correct.any(1)] = ranked.shape[1]
    out["mean_rank"] = float(first.mean() + 1)
    top1_sim = np.take_along_axis(sims, order[:, :1], axis=1).ravel()
    same = correct[:, 0]
    out["sim_gap"] = float(top1_sim[same].mean() - top1_sim[~same].mean()) if same.any() and (~same).any() else 0.0
    return out


def run(M, ctx, out_dir, para_dir=None, dep_dir=None):
    """Writes appendix_evidence.json + discussion_evidence.json. Cache-only."""
    labels_id = ctx["labels_id"]
    # Per-species RQ1 for every method (deployment gallery).
    per_sp = {}
    for name, md in M.items():
        ev = R.method_retrieval_eval(md, md["id"], labels_id, md["gal"][0], md["gal"][1])
        per_sp[name] = ev["per_species"]

    # All-pairs Wilcoxon (vs SC-URD if present, else vs DINOv2).
    anchor = "SC-URD" if "SC-URD" in per_sp else ("DINOv2" if "DINOv2" in per_sp else None)
    pairwise = []
    if anchor:
        for base in [m for m in per_sp if m != anchor]:
            w = R.paired_wilcoxon(per_sp[anchor], per_sp[base])
            pairwise.append({"anchor": anchor, "baseline": base,
                             "mean_anchor": float(np.mean(list(per_sp[anchor].values()))),
                             "mean_baseline": float(np.mean(list(per_sp[base].values()))),
                             "delta": w["delta"], "p_value": w["p_value"], "sig": w["sig"], "n": w["n"]})

    retrieval_quality = {name: _retrieval_quality(md, labels_id) for name, md in M.items()
                         if name in {"DINOv2", "ArcFace-557", "CE-Full", "Fusion", "SupCon", "SC-URD"}}

    appendix = {"gallery_scope": ctx["scope"], "pairwise_rq1": pairwise,
                "retrieval_quality": retrieval_quality}
    os.makedirs(str(out_dir), exist_ok=True)
    with open(os.path.join(str(out_dir), "appendix_evidence.json"), "w") as f:
        json.dump(appendix, f, indent=2, sort_keys=True)

    # Discussion: RQ1 hard cases / largest SC-URD gains over DINOv2.
    discussion = {}
    if anchor and "DINOv2" in per_sp:
        species = sorted(per_sp[anchor])
        hard = sorted(species, key=lambda s: per_sp[anchor][s])[:10]
        gains = sorted(species, key=lambda s: per_sp[anchor].get(s, 0) - per_sp["DINOv2"].get(s, 0), reverse=True)[:10]
        discussion["rq1_hard_species"] = [{"species": s, "acc": per_sp[anchor][s]} for s in hard]
        discussion["rq1_largest_gains_vs_dinov2"] = [
            {"species": s, "acc": per_sp[anchor].get(s, 0),
             "dinov2": per_sp["DINOv2"].get(s, 0),
             "delta": per_sp[anchor].get(s, 0) - per_sp["DINOv2"].get(s, 0)} for s in gains]
    with open(os.path.join(str(out_dir), "discussion_evidence.json"), "w") as f:
        json.dump(discussion, f, indent=2, sort_keys=True)
    print(f"saved appendix_evidence.json ({len(pairwise)} pairwise) + discussion_evidence.json")
    return {"appendix": appendix, "discussion": discussion}

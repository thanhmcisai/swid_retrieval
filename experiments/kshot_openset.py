# -*- coding: utf-8 -*-
"""RQ2.2 — K-shot open-set retrieval (OOD-only gallery).

Lifted from final_metric_learning_cea_2026.py:5388-5466. Query = OOD query images
of the top-50 OOD species; gallery = K random OOD shots per species (no SWI mixed
in), K∈{0,1,3,5,10}, 50 repeats (seed 42+rep), + Wilcoxon at K=10. Cache-only
(reads `embs_ood_*` from the cache via the registry). Output → rq2_kshot.json,
tab:appendix_ood_kshot, fig3b.
"""

import json
import os

import numpy as np

from .. import config
from . import registry as R

K_VALS = [0, 1, 3, 5, 10]
N_REPS = 50
_WILCOXON_PAIRS = [
    ("SC-URD", "ArcFace-557", "SC-URD vs ArcFace-557"),
    ("SC-URD", "DINOv2", "SC-URD vs DINOv2 frozen"),
    ("Fusion", "ArcFace-557", "Fusion vs ArcFace-557"),
    ("ArcFace-557", "CE-Full", "ArcFace vs CE-Full NN"),
]


def run(M, out_dir):
    """M from registry.build_M(...). Writes rq2_kshot.json into out_dir."""
    splits, top50 = R.ood_species_splits()
    rq2_kshot = {}
    for name, md in M.items():
        if md.get("ood") is None:
            continue
        gal_e, gal_l = md["gal"]
        ood_e = md["ood"]
        rq2_kshot[name] = {}
        for K in K_VALS:
            if K == 0:  # OOD queries vs the SWI gallery (≈0 — OOD not in SWI)
                q = np.concatenate([ood_e[splits[sp]["query_indices"]] for sp in top50])
                ql = np.array([sp for sp in top50 for _ in splits[sp]["query_indices"]])
                macro = R.method_retrieval_eval(md, q, ql, gal_e, gal_l)["mean"]
                rq2_kshot[name][K] = {"mean": float(macro), "std": 0.0, "ci_lo": float(macro), "ci_hi": float(macro)}
                continue
            accs = []
            for rep in range(N_REPS):
                rng = np.random.RandomState(42 + rep)
                q_e, q_l, g_e, g_l = [], [], [], []
                for sp in top50:
                    qi = splits[sp]["query_indices"]
                    q_e.append(ood_e[qi]); q_l.extend([sp] * len(qi))
                    pool = splits[sp]["pool_indices"]
                    k_eff = min(K, len(pool))
                    sel = rng.choice(len(pool), k_eff, replace=False)
                    g_e.append(ood_e[[pool[i] for i in sel]]); g_l.extend([sp] * k_eff)
                macro = R.method_retrieval_eval(md, np.concatenate(q_e), np.array(q_l),
                                                np.concatenate(g_e), np.array(g_l))["mean"]
                accs.append(macro)
            m, lo, hi = R.bootstrap(accs)
            rq2_kshot[name][K] = {"mean": float(m), "std": float(np.std(accs)),
                                  "ci_lo": float(lo), "ci_hi": float(hi), "per_repeat": accs}
        print(f"  kshot {name:25s} K=1:{rq2_kshot[name][1]['mean']:.3f} "
              f"K=5:{rq2_kshot[name][5]['mean']:.3f} K=10:{rq2_kshot[name][10]['mean']:.3f}")

    kshot_wilcoxon = []
    for m1, m2, label in _WILCOXON_PAIRS:
        if m1 not in rq2_kshot or m2 not in rq2_kshot:
            continue
        a = rq2_kshot[m1][10].get("per_repeat", [])
        b = rq2_kshot[m2][10].get("per_repeat", [])
        if len(a) < 2 or len(b) < 2:
            continue
        n = min(len(a), len(b))
        _, p, sig = R.wilcoxon_test(np.array(a[:n]), np.array(b[:n]))
        kshot_wilcoxon.append({"pair": label, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                               "delta": float(np.mean(a) - np.mean(b)), "p": p, "sig": sig})

    save = {name: {K: {k: v for k, v in vd.items() if k != "per_repeat"} for K, vd in Kd.items()}
            for name, Kd in rq2_kshot.items()}
    out = {"scurd_main_mode": config.SCURD_MODE, "kshot_retrieval": save, "kshot_wilcoxon": kshot_wilcoxon}
    path = os.path.join(str(out_dir), "rq2_kshot.json")
    os.makedirs(str(out_dir), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path}")
    return out

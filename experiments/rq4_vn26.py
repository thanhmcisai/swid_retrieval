# -*- coding: utf-8 -*-
"""RQ4 — SWI→VN26 cross-domain + intra-VN26 cross-magnification.

Lifted verbatim from final_metric_learning_cea_2026.py:5620-5713. Uses each
method's own `swi_scales`/`vn26_mags` galleries from the exp4 cache (independent
of the 24/954 SWI mask). 4A: per gallery scale × VN26 query mag, 19 common species.
4B: cross-magnification. + paired Wilcoxon on SWI_pool→VN26_all. Cache-only when
exp4 cache present (RQ4 is skipped if it is missing). Output → rq4_generalization.json.
"""

import json
import os

import numpy as np

from .. import config
from . import registry as R


def run(M, out_dir):
    """M from registry.build_M(with_exp4=True). Writes rq4_generalization.json."""
    rq4a, rq4b = {}, {}
    have_exp4 = any(md.get("swi_scales") for md in M.values())
    if not have_exp4:
        print("  RQ4 skipped: no exp4 (VN26) embeddings loaded.")
        return None

    for name, md in M.items():
        if not md["swi_scales"]:
            continue
        rq4a[name], rq4b[name] = {}, {}
        for scale, (swi_e, swi_l) in md["swi_scales"].items():
            swi_key = f"SWI_{scale}"
            rq4a[name][swi_key] = {}
            for qcfg in R.QUERY_CONFIGS:
                mags = R.VN26_MAGS if qcfg == "all" else [qcfg]
                ve = [md["vn26_mags"][mg][0] for mg in mags if mg in md["vn26_mags"]]
                vl = [md["vn26_mags"][mg][1] for mg in mags if mg in md["vn26_mags"]]
                if not ve:
                    continue
                vn26_e, vn26_l = np.concatenate(ve), np.concatenate(vl)
                common = set(vn26_l) & set(swi_l)
                if not common:
                    continue
                gm = np.array([s in common for s in swi_l])
                qm = np.array([s in common for s in vn26_l])
                ev = R.method_retrieval_eval(md, vn26_e[qm], vn26_l[qm], swi_e[gm], swi_l[gm])
                m4, lo4, hi4 = R.bootstrap(list(ev["per_species"].values()))
                rq4a[name][swi_key][f"VN26_{qcfg}"] = {
                    "mean": m4, "ci_lo": lo4, "ci_hi": hi4,
                    "n_common": len(common), "per_species": ev["per_species"]}
        for g_mags, q_mags in R.VN26_CONFIGS:
            g_key, q_key = "+".join(g_mags), "+".join(q_mags)
            ge = [md["vn26_mags"][mg][0] for mg in g_mags if mg in md["vn26_mags"]]
            qe = [md["vn26_mags"][mg][0] for mg in q_mags if mg in md["vn26_mags"]]
            gl = [md["vn26_mags"][mg][1] for mg in g_mags if mg in md["vn26_mags"]]
            ql = [md["vn26_mags"][mg][1] for mg in q_mags if mg in md["vn26_mags"]]
            if not ge or not qe:
                continue
            ev = R.method_retrieval_eval(md, np.concatenate(qe), np.concatenate(ql),
                                         np.concatenate(ge), np.concatenate(gl))
            m4b, lo4b, hi4b = R.bootstrap(list(ev["per_species"].values()))
            rq4b[name].setdefault(g_key, {})[q_key] = {
                "mean": m4b, "ci_lo": lo4b, "ci_hi": hi4b, "per_species": ev["per_species"]}
        pool_all = rq4a[name].get("SWI_pool", {}).get("VN26_all", {}).get("mean")
        b4b = [rq4b[name][g][q]["mean"] for g in rq4b[name] for q in rq4b[name][g]]
        print(f"  RQ4 {name:25s} E4A_pool={f'{pool_all:.3f}' if pool_all else '—':>6} "
              f"E4B_best={f'{max(b4b):.3f}' if b4b else '—':>6}")

    e4a_wilcoxon = []
    for m1, m2, label in [("SC-URD", "ArcFace-557", "SC-URD vs ArcFace-557"),
                          ("SC-URD", "DINOv2", "SC-URD vs DINOv2 frozen"),
                          ("Fusion", "ArcFace-557", "Fusion vs ArcFace-557"),
                          ("DINOv2", "ArcFace-557", "DINOv2 vs ArcFace-557")]:
        ps1 = rq4a.get(m1, {}).get("SWI_pool", {}).get("VN26_all", {}).get("per_species", {})
        ps2 = rq4a.get(m2, {}).get("SWI_pool", {}).get("VN26_all", {}).get("per_species", {})
        common = sorted(set(ps1) & set(ps2))
        if len(common) < 2:
            continue
        a = np.array([ps1[s] for s in common]); b = np.array([ps2[s] for s in common])
        _, p, sig = R.wilcoxon_test(a, b)
        e4a_wilcoxon.append({"pair": label, "n_sp": len(common), "mean_a": float(a.mean()),
                             "mean_b": float(b.mean()), "delta": float(a.mean() - b.mean()), "p": p, "sig": sig})

    out = {"scurd_main_mode": config.SCURD_MODE, "cross_domain": rq4a,
           "cross_magnification": rq4b, "e4a_wilcoxon": e4a_wilcoxon}
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "rq4_generalization.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path}")
    return out

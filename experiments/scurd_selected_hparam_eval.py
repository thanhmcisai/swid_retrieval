# -*- coding: utf-8 -*-
"""Evaluate the SC-URD configuration selected on SWI meta-val.

This is a post-hoc audit, not the main paper configuration. It reads
hyperparameters/scurd_hyperparameter_selection.json, applies the selected
mode/tau/top_m to the existing SC-URD projected embeddings, and writes a compact
comparison across the main deployment axes without overwriting centered-default
results.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import registry as R


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _distance_scores(q, g):
    q, g = _norm(q), _norm(g)
    return 1.0 - (q @ g.T).max(axis=1)


def _auroc(scores_id, scores_ood):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    s = np.concatenate([scores_id, scores_ood])
    return float(roc_auc_score(y, s))


def _fpr95(scores_id, scores_ood):
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    s = np.concatenate([scores_id, scores_ood])
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, 0.95)
    return float(fpr[min(idx, len(fpr) - 1)])


def _selected_config(run_root):
    path = Path(run_root) / "hyperparameters" / "scurd_hyperparameter_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected hyperparameter JSON: {path}")
    obj = json.load(open(path))
    selected = obj.get("selected") or {}
    required = {"checkpoint", "mode", "tau", "top_m"}
    missing = required - set(selected)
    if missing:
        raise KeyError(f"Selected hyperparameter row missing keys: {sorted(missing)}")
    return selected, path


def _scurd_eval(md, query_embs, query_labels, gallery_embs, gallery_labels, selected):
    return R.scurd_retrieval_eval(
        query_embs, query_labels, gallery_embs, gallery_labels,
        mode=str(selected["mode"]),
        tau=float(selected["tau"]),
        top_m=int(selected["top_m"]),
    )


def _incremental_eval(md, ctx, selected, K=10):
    labels_id = ctx["labels_id"]
    labels_ood = ctx["labels_ood"]
    splits, top50 = R.ood_species_splits()
    pool_idx, query_idx, query_labels = [], [], []
    for sp in top50:
        pool = splits[sp]["pool_indices"]
        pool_idx.extend(pool[:min(K, len(pool))])
        qi = splits[sp]["query_indices"]
        query_idx.extend(qi)
        query_labels.extend([sp] * len(qi))
    gal_e = np.concatenate([md["gal"][0], md["ood"][pool_idx]])
    gal_l = np.concatenate([md["gal"][1], labels_ood[pool_idx]])
    old_ev = _scurd_eval(md, md["id"], labels_id, gal_e, gal_l, selected)
    new_ev = _scurd_eval(md, md["ood"][query_idx], np.asarray(query_labels), gal_e, gal_l, selected)
    return old_ev, new_ev


def _ood_kshot(md, selected, K=10, n_reps=50):
    splits, top50 = R.ood_species_splits()
    accs = []
    for rep in range(int(n_reps)):
        rng = np.random.RandomState(42 + rep)
        q_e, q_l, g_e, g_l = [], [], [], []
        for sp in top50:
            qi = splits[sp]["query_indices"]
            q_e.append(md["ood"][qi])
            q_l.extend([sp] * len(qi))
            pool = splits[sp]["pool_indices"]
            k_eff = min(int(K), len(pool))
            sel = rng.choice(len(pool), k_eff, replace=False)
            g_e.append(md["ood"][[pool[i] for i in sel]])
            g_l.extend([sp] * k_eff)
        ev = _scurd_eval(md, np.concatenate(q_e), np.asarray(q_l),
                         np.concatenate(g_e), np.asarray(g_l), selected)
        accs.append(ev["mean"])
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "min": float(np.min(accs)),
        "max": float(np.max(accs)),
        "n_repeats": int(n_reps),
    }


def _vn26_pool(md, selected):
    if "pool" not in md.get("swi_scales", {}) or "all" not in md.get("vn26_mags", {}):
        return None
    swi_e, swi_l = md["swi_scales"]["pool"]
    vn_e, vn_l = md["vn26_mags"]["all"]
    common = set(swi_l) & set(vn_l)
    if not common:
        return None
    gm = np.asarray([x in common for x in swi_l])
    qm = np.asarray([x in common for x in vn_l])
    ev = _scurd_eval(md, vn_e[qm], vn_l[qm], swi_e[gm], swi_l[gm], selected)
    m, lo, hi = R.bootstrap(list(ev["per_species"].values()))
    return {"mean": float(m), "ci_lo": float(lo), "ci_hi": float(hi), "n_common": int(len(common))}


def _row_for_scope(scope, selected, with_exp4):
    M, ctx = R.build_M(scope, with_exp4=with_exp4, with_scurd=True)
    md = M["SC-URD"]
    gal_e, gal_l = md["gal"]
    r1 = _scurd_eval(md, md["id"], ctx["labels_id"], gal_e, gal_l, selected)
    sid = _distance_scores(md["id"], gal_e)
    sod = _distance_scores(md["ood"], gal_e)[R.ood_test_mask(ctx["labels_ood"])]
    old_ev, new_ev = _incremental_eval(md, ctx, selected, K=10)
    row = {
        "scope": scope,
        "gallery_scope": ctx["scope"],
        "gallery_images": int(len(gal_l)),
        "gallery_species": int(len(set(gal_l.tolist()))),
        "selected_checkpoint": selected["checkpoint"],
        "selected_mode": selected["mode"],
        "selected_tau": float(selected["tau"]),
        "selected_top_m": int(selected["top_m"]),
        "public_id_macro_R1": float(r1["mean"]),
        "ood_auroc": _auroc(sid, sod),
        "ood_fpr95": _fpr95(sid, sod),
        "old_after_plus50_K10": float(old_ev["mean"]),
        "new50_after_K10": float(new_ev["mean"]),
    }
    vn = _vn26_pool(md, selected) if with_exp4 else None
    if vn is not None:
        row.update({
            "vn26_swi_pool_all": vn["mean"],
            "vn26_ci_lo": vn["ci_lo"],
            "vn26_ci_hi": vn["ci_hi"],
            "vn26_n_common": vn["n_common"],
        })
    return row, M, ctx


def run(run_root, force=False):
    run_root = Path(run_root)
    out_dir = run_root / "native_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scurd_selected_hparam_eval.csv"
    json_path = out_dir / "scurd_selected_hparam_eval.json"
    force = force or os.environ.get("FORCE_SCURD_SELECTED_HPARAM_EVAL", "0") == "1"
    if csv_path.exists() and json_path.exists() and not force:
        print(f"  ✅ selected-hparam eval exists: {csv_path}")
        return json.load(open(json_path))

    selected, selected_path = _selected_config(run_root)
    main_ckpt_name = Path(getattr(config, "SCURD_MAIN_CKPT", "")).name
    if main_ckpt_name and str(selected["checkpoint"]) != main_ckpt_name:
        print(
            "  ⚠️ selected checkpoint differs from configured SC-URD checkpoint: "
            f"selected={selected['checkpoint']} configured={main_ckpt_name}. "
            "This audit reuses the configured projected embeddings."
        )
    rows = []
    scopes = [
        s.strip()
        for s in os.environ.get("SCURD_SELECTED_EVAL_SCOPES", "id_only").split(",")
        if s.strip()
    ]
    if not scopes:
        scopes = ["id_only"]
    M_dep = None
    for scope in scopes:
        if scope == "id_only":
            row, M_scope, _ = _row_for_scope("id_only", selected, with_exp4=True)
            M_dep = M_scope
        elif scope == "full_swi":
            row, _, _ = _row_for_scope("full_swi", selected, with_exp4=False)
            row["note"] = "Full-SWI selected-hparam evaluation is memory-intensive."
        else:
            raise ValueError(
                f"Unknown SCURD_SELECTED_EVAL_SCOPES entry {scope!r}; use id_only and optionally full_swi."
            )
        rows.append(row)
    if M_dep is None:
        _, M_dep, _ = _row_for_scope("id_only", selected, with_exp4=True)
    kshot = _ood_kshot(M_dep["SC-URD"], selected, K=10, n_reps=int(os.environ.get("SCURD_SELECTED_KSHOT_REPS", "50")))

    out = {
        "selected_source": str(selected_path),
        "selected": selected,
        "eval_scopes": scopes,
        "rows": rows,
        "ood_only_kshot_K10": kshot,
        "note": (
            "Post-hoc audit of the meta-val selected scoring configuration. "
            "Main paper results are not overwritten."
        ),
    }
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {csv_path}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"  selected-config OOD-only K=10: {kshot['mean']:.3f} ± {kshot['std']:.3f}")
    return out

# -*- coding: utf-8 -*-
"""RQ2.1b — extended OOD baseline suite (tab:appendix_ood_extended).

Lifted from final_metric_learning_cea_2026.py:3576-4392. Cache-only for
MSP/Energy/Narrow + per-method distance + Fusion-Dist (from cached logits/embeddings).
ODIN and OpenMax load their published score caches (`exp2_{odin,openmax}_scores_
recomputed_v2.npz` in the source results dir) when present; otherwise they are
recomputed on GPU (CE model + input gradients / Weibull) only if RUN_HEAVY=1.

Output → rq2_ood_extended.json.
"""

import json
import os

import numpy as np

from .. import config
from ..data import canonical_label
from . import registry as R

OOD_SCORE_CACHE_VERSION = os.environ.get("OOD_SCORE_CACHE_VERSION", "v2")


# ── OOD scorers (monolith L506-537) ──────────────────────────────────────────
def _msp(logits, T=1.0):
    import torch.nn.functional as F
    import torch
    t = torch.tensor(np.asarray(logits)).float()
    return (1 - F.softmax(t / T, dim=1).max(dim=1).values).numpy()


def _energy(logits):
    import torch
    t = torch.tensor(np.asarray(logits)).float()
    return (-torch.logsumexp(t, dim=1)).numpy()


def _cos_dist(query, gallery):
    return 1.0 - (R._norm(query) @ R._norm(gallery).T).max(axis=1)


# ── kNN baselines (monolith L2514-2545) ──────────────────────────────────────
# Cache-only numpy (no GPU / RUN_HEAVY): OOD score = cosine distance to the k-th
# nearest gallery neighbour; classification = k-NN majority vote.
def knn_ood_score(query, gallery, k=5):
    sims = R._norm(query) @ R._norm(gallery).T
    k_eff = min(int(k), sims.shape[1])
    part = np.partition(-sims, k_eff - 1, axis=1)[:, :k_eff]
    kth = (-part).min(axis=1)  # k-th largest similarity per query row
    return 1.0 - kth


def knn_classify_per_species(query, query_labels, gallery, gallery_labels, k=5):
    from collections import Counter
    sims = R._norm(query) @ R._norm(gallery).T
    g_lbl, q_lbl = np.asarray(gallery_labels), np.asarray(query_labels)
    k_eff = min(int(k), sims.shape[1])
    topk_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
    preds = np.array([Counter(g_lbl[topk_idx[i]]).most_common(1)[0][0] for i in range(len(q_lbl))])
    per_sp = {c: float((preds[q_lbl == c] == c).mean()) for c in sorted(set(q_lbl))}
    return (float(np.mean(list(per_sp.values()))) if per_sp else 0.0), per_sp


# Map kNN baseline name → registry method key (monolith _knn_backbones).
_KNN_BACKBONES = {"kNN-CE-Full": "CE-Full", "kNN-ArcFace": "ArcFace-557",
                  "kNN-DINOv2": "DINOv2", "kNN-ViT": "ArcFace-ViT", "kNN-ProtoNet": "ProtoNet"}
_KNN_K_GRID = [1, 5, 10, 20, 50]


def _auroc(sid, sod):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sod))])
    return float(roc_auc_score(y, np.concatenate([sid, sod])))


def _fpr95(sid, sod, tpr_target=0.95):
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sod))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([sid, sod]))
    idx = np.searchsorted(tpr, tpr_target)
    return float(fpr[min(idx, len(fpr) - 1)])


def _aupr(sid, sod):
    from sklearn.metrics import average_precision_score
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sod))])
    return float(average_precision_score(y, np.concatenate([sid, sod])))


def _boot_ood_scores(sid, sod, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    sid, sod = np.asarray(sid), np.asarray(sod)

    def _b(fn):
        vals = [fn(sid[rng.choice(len(sid), len(sid), True)], sod[rng.choice(len(sod), len(sod), True)])
                for _ in range(n)]
        return {"mean": float(np.mean(vals)), "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5))}
    return {"AUROC": _b(_auroc), "FPR@95TPR": _b(_fpr95), "AUPR": _b(_aupr)}


def _odin_openmax_from_cache(kind, test_mask):
    """Load published ODIN/OpenMax score cache (cache-only). None if absent."""
    path = config.RESULTS_DIR / f"exp2_{kind}_scores_recomputed_{OOD_SCORE_CACHE_VERSION}.npz"
    if not path.exists():
        return None
    c = np.load(path, allow_pickle=False)
    if "scores_id" not in c.files or "scores_ood" not in c.files:
        return None
    m = _boot_ood_scores(c["scores_id"], c["scores_ood"][test_mask])
    if kind == "odin":
        m["best_params"] = {"auroc": float(c.get("best_val_auroc", np.nan)),
                            "T": int(c["best_T"]) if "best_T" in c.files else None,
                            "epsilon": float(c["best_epsilon"]) if "best_epsilon" in c.files else None}
    else:
        m["best_params"] = {"auroc": float(c.get("best_val_auroc", np.nan)),
                            "alpha": int(c["best_alpha"]) if "best_alpha" in c.files else None}
    print(f"  loaded {kind} score cache: {path.name}")
    return m


def run(M, ctx, out_dir):
    """M, ctx from registry.build_M('id_only'). Writes rq2_ood_extended.json."""
    emb = ctx["emb"]
    labels_ood = ctx["labels_ood"]
    val_mask, test_mask = R.ood_val_test_masks(labels_ood)
    baselines = {}
    separation = {}  # per-method mean(OOD dist) − mean(ID dist), for fig_discussion_ood

    # CE logit baselines (cache-only).
    for name, lid_key, lood_key in [("CE-Full", "logits_id_ce_full", "logits_ood_ce_full"),
                                    ("CE-Narrow", "logits_id_ce_narrow", "logits_ood_ce_narrow")]:
        if lid_key in emb.files and lood_key in emb.files:
            lid, lood = emb[lid_key], emb[lood_key]
            baselines[f"{name}-MSP"] = _boot_ood_scores(_msp(lid), _msp(lood)[test_mask])
            baselines[f"{name}-Energy"] = _boot_ood_scores(_energy(lid), _energy(lood)[test_mask])

    # Per-method distance (cache-only).
    for name, md in M.items():
        if md.get("ood") is None:
            continue
        gal_e, _ = md["gal"]
        sid = _cos_dist(md["id"], gal_e)
        sod = _cos_dist(md["ood"], gal_e)[test_mask]
        baselines[f"{name}-Dist"] = _boot_ood_scores(sid, sod)
        separation[name] = float(np.mean(sod) - np.mean(sid))

    if "ArcFace-557" in M and "DINOv2" in M:
        gal_f = R._fuse(M["ArcFace-557"]["gal"][0], M["DINOv2"]["gal"][0])
        baselines["Fusion-ArcFace-DINOv2-Dist"] = _boot_ood_scores(
            _cos_dist(R._fuse(M["ArcFace-557"]["id"], M["DINOv2"]["id"]), gal_f),
            _cos_dist(R._fuse(M["ArcFace-557"]["ood"], M["DINOv2"]["ood"]), gal_f)[test_mask])

    # kNN-OOD (k-th-NN cosine distance): tune k on the OOD val split per backbone,
    # report on the test split. Tests whether backbone quality > training paradigm.
    knn_best = {}
    for label, key in _KNN_BACKBONES.items():
        md = M.get(key)
        if md is None or md.get("ood") is None:
            continue
        gal_e, _ = md["gal"]
        best_k, best_au = _KNN_K_GRID[0], -1.0
        for k in _KNN_K_GRID:
            s_id = knn_ood_score(md["id"], gal_e, k=k)
            s_od = knn_ood_score(md["ood"], gal_e, k=k)[val_mask]
            au = _auroc(s_id, s_od)
            if au > best_au:
                best_au, best_k = au, k
        knn_best[label] = best_k
        m = _boot_ood_scores(knn_ood_score(md["id"], gal_e, k=best_k),
                             knn_ood_score(md["ood"], gal_e, k=best_k)[test_mask])
        m["best_params"] = {"k": int(best_k), "val_auroc": float(best_au)}
        baselines[label] = m
    if knn_best:
        print(f"  kNN best k (val-tuned): {knn_best}")

    # ODIN / OpenMax: prefer published score caches; recompute only under RUN_HEAVY.
    heavy = os.environ.get("RUN_HEAVY", "0") == "1"
    for kind, label in [("odin", "CE-Full-ODIN"), ("openmax", "CE-Full-OpenMax")]:
        cached = _odin_openmax_from_cache(kind, test_mask)
        if cached is not None:
            baselines[label] = cached
        elif heavy:
            try:
                baselines[label] = _recompute(kind, ctx)
            except Exception as e:  # noqa: BLE001
                print(f"  {kind} recompute failed: {e}")
        else:
            print(f"  {kind}: no score cache and RUN_HEAVY=0 → skipping (set RUN_HEAVY=1 to recompute).")

    out = {"gallery_scope": ctx["scope"], "ood_baseline_suite": baselines,
           "ood_score_separation": separation}
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "rq2_ood_extended.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path} ({len(baselines)} baselines)")
    return out


# ── GPU recompute (RUN_HEAVY): ODIN input-gradients / OpenMax Weibull ────────
def _recompute(kind, ctx):
    """Recompute ODIN/OpenMax from the CE-Full model + public images. Heavy."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import pandas as pd
    from ..models import _load_ce
    from ..data import CSVImageDataset, get_transforms, ManifestDataset, load_swi_manifest, full_swi_items, ce_train_image_paths

    emb, labels_ood = ctx["emb"], ctx["labels_ood"]
    dev = ctx["device"]
    val_mask, test_mask = R.ood_val_test_masks(labels_ood)
    eval_tf = get_transforms(224, augment=False)
    model_ce, ce_ckpt = _load_ce(config.CKPT_CE_FULL, dev)
    ce_species = [canonical_label(s) for s in ce_ckpt.get("ce_species_list")]
    logits_id, logits_ood = emb["logits_id_ce_full"], emb["logits_ood_ce_full"]

    id_df = pd.read_csv(config.ID_IMAGES_CSV)
    ood_df = pd.read_csv(config.OOD_IMAGES_CSV)
    nw = config.num_workers()

    if kind == "odin":
        T_grid, eps_grid = [1, 10, 100, 1000], [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]

        def grid(df):
            ds = CSVImageDataset(df, transform=eval_tf)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=nw, pin_memory=True)
            res = {(T, e): [] for T in T_grid for e in eps_grid if e > 0}
            nz = [e for e in eps_grid if e > 0]
            for T in T_grid:
                for imgs, _ in loader:
                    model_ce.zero_grad(set_to_none=True)
                    imgs = imgs.to(dev).requires_grad_(True)
                    F.log_softmax(model_ce(imgs) / T, dim=1).max(dim=1).values.sum().backward()
                    gs = imgs.grad.sign().detach(); det = imgs.detach()
                    with torch.no_grad():
                        for e in nz:
                            p = torch.clamp(det - float(e) * gs, -3.0, 3.0)
                            res[(T, e)].append((1.0 - F.softmax(model_ce(p) / T, dim=1).max(1).values).cpu())
            return {k: torch.cat(v).numpy() for k, v in res.items()}

        idg, oog = grid(id_df), grid(ood_df)
        for T in T_grid:
            idg[(T, 0.0)] = (1.0 - F.softmax(torch.tensor(logits_id) / T, dim=1).max(1).values.numpy())
            oog[(T, 0.0)] = (1.0 - F.softmax(torch.tensor(logits_ood) / T, dim=1).max(1).values.numpy())
        best = {"auroc": -1, "T": None, "eps": None}
        for T in T_grid:
            for e in eps_grid:
                au = _auroc(idg[(T, e)], oog[(T, e)][val_mask])
                if au > best["auroc"]:
                    best = {"auroc": au, "T": T, "eps": e}
        sid, sod = idg[(best["T"], best["eps"])], oog[(best["T"], best["eps"])]
        np.savez_compressed(config.RESULTS_DIR / f"exp2_odin_scores_recomputed_{OOD_SCORE_CACHE_VERSION}.npz",
                            scores_id=sid, scores_ood=sod, best_T=np.array(best["T"]),
                            best_epsilon=np.array(best["eps"]), best_val_auroc=np.array(best["auroc"]))
        m = _boot_ood_scores(sid, sod[test_mask]); m["best_params"] = best
        return m

    # OpenMax
    from scipy.stats import weibull_min
    manifest = load_swi_manifest()
    train_set = ce_train_image_paths(manifest)
    items = [(p, canonical_label(s)) for p, s in full_swi_items(manifest) if p in train_set]
    cls_to_idx = {sp: i for i, sp in enumerate(ce_species)}
    items = [(p, s) for p, s in items if s in cls_to_idx]
    tdf = pd.DataFrame([{"file_path": p, "label": s} for p, s in items])
    tds = CSVImageDataset(tdf, transform=eval_tf)
    tloader = DataLoader(tds, batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    tl, tlab = [], []
    model_ce.eval()
    with torch.no_grad():
        for imgs, _ in tloader:
            tl.append(model_ce(imgs.to(dev)).cpu().numpy())
    train_logits = np.concatenate(tl)
    train_labels = np.array([cls_to_idx[s] for _, s in items])

    preds = train_logits.argmax(1)
    mavs, wb = {}, {}
    for cls in sorted(np.unique(train_labels)):
        mask = (train_labels == cls) & (preds == cls)
        if mask.sum() == 0:
            mask = train_labels == cls
        vecs = train_logits[mask]
        if len(vecs) == 0:
            continue
        mav = vecs.mean(0); d = np.linalg.norm(vecs - mav, axis=1)
        ts = max(1, min(20, len(d)))
        try:
            sh, lo, sc = weibull_min.fit(np.sort(d)[-ts:], floc=0)
        except Exception:
            sh, lo, sc = 1.0, 0.0, float(d.mean()) + 1e-8
        mavs[int(cls)] = mav; wb[int(cls)] = (sh, lo, sc)

    def omscore(logits, alpha):
        out = np.zeros(len(logits), np.float32)
        for i, av in enumerate(np.asarray(logits)):
            top = np.argsort(av)[::-1][:int(alpha)]; rev = av.copy(); omega = 0.0
            for ci in top:
                ci = int(ci)
                if ci not in mavs:
                    continue
                w = float(weibull_min.cdf(float(np.linalg.norm(av - mavs[ci])), wb[ci][0], loc=wb[ci][1], scale=wb[ci][2]))
                rev[ci] = av[ci] * (1 - w); omega += av[ci] * w
            aug = np.concatenate([[omega], rev]); ex = np.exp(aug - aug.max())
            out[i] = float(ex[0] / ex.sum())
        return out

    best = {"auroc": -1, "alpha": None}
    for alpha in [5, 10, 15, 20]:
        au = _auroc(omscore(logits_id, alpha), omscore(logits_ood[val_mask], alpha))
        if au > best["auroc"]:
            best = {"auroc": au, "alpha": alpha}
    sid, sod = omscore(logits_id, best["alpha"]), omscore(logits_ood, best["alpha"])
    np.savez_compressed(config.RESULTS_DIR / f"exp2_openmax_scores_recomputed_{OOD_SCORE_CACHE_VERSION}.npz",
                        scores_id=sid, scores_ood=sod, best_alpha=np.array(best["alpha"]),
                        best_val_auroc=np.array(best["auroc"]), fit_classes=np.array(len(mavs)))
    m = _boot_ood_scores(sid, sod[test_mask]); m["best_params"] = best; m["fit_classes"] = len(mavs)
    return m

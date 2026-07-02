# -*- coding: utf-8 -*-
"""Shared method registry + scorers/stats for the native (ported) experiments.

Rebuilds the monolith's `M[name]` registry (final_metric_learning_cea_2026.py
L5144) by reading the npz caches instead of module globals, at a chosen gallery
scope (id_only / full_swi / ce_train). All ported experiments (RQ4, OOD baselines,
K-shot, RQ5-extra, appendix/discussion, CE-finetune, costs) consume `build_M(...)`.

Lifted verbatim (only re-parameterized): prototype/recall (L1687/L1709), PDA
shift/CORAL/aQE (L2140-2185), _fuse/_norm/_softmax_np (L2761-2774), SC-URD
scoring (L2786-2871), _bootstrap/_wilcoxon_test (L2929-2950), exp4 cache loader
(L2345-2505). Stats use the monolith's exact seeds (bootstrap seed=42, 1000 reps).
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import config
from ..data import canonical_label

# ── exp4 (VN26) constants — match monolith L2321-2340 ────────────────────────
SWI_SCALES = [256, 512, 768]
GALLERY_CONFIGS = [256, 512, 768, "pool"]
VN26_MAGS = ["x10", "x20", "x50"]
QUERY_CONFIGS = ["x10", "x20", "x50", "all"]
VN26_CONFIGS = [
    (["x10"], ["x20", "x50"]), (["x20"], ["x10", "x50"]), (["x50"], ["x10", "x20"]),
    (["x10", "x20"], ["x50"]), (["x10", "x50"], ["x20"]), (["x20", "x50"], ["x10"]),
]
METHODS = ["ArcFace-557", "ArcFace-954", "CE-Full", "CE-Narrow", "DINOv2"]
VARIANT_METHODS = ["ArcFace-ViT", "ArcFace-RN50", "SupCon-ConvNeXt"]
METHODS_ALL = METHODS + VARIANT_METHODS


def _e4key(prefix, method, split):
    return f"{prefix}__{method.replace('-', '_').replace('/', '_')}__{str(split)}"


# ── basic vector ops (monolith L2761-2774) ──────────────────────────────────
def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _fuse(e1, e2):
    return _norm(np.concatenate([_norm(e1), _norm(e2)], axis=1))


def _softmax_np(x, axis=1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-12)


# ── prototype / recall (monolith L1687/L1709) ────────────────────────────────
def prototype_classify_per_species(query_embs, query_labels, gallery_embs, gallery_labels):
    q = F.normalize(torch.tensor(query_embs).float(), dim=1)
    g = F.normalize(torch.tensor(gallery_embs).float(), dim=1)
    ql, gl = np.asarray(query_labels), np.asarray(gallery_labels)
    classes = np.unique(gl)
    proto = torch.cat([F.normalize(g[gl == c].mean(0, keepdim=True), dim=1) for c in classes], dim=0)
    preds = classes[(q @ proto.T).argmax(dim=1).numpy()]
    per_species = {c: float((preds[ql == c] == c).mean()) for c in np.unique(ql)}
    return float(np.mean(list(per_species.values()))), per_species


def recall_at_1_per_species(query_embs, query_labels, gallery_embs, gallery_labels, self_retrieval=False):
    q = F.normalize(torch.tensor(query_embs).float(), dim=1)
    g = F.normalize(torch.tensor(gallery_embs).float(), dim=1)
    ql, gl = np.asarray(query_labels), np.asarray(gallery_labels)
    sims = (q @ g.T).cpu().numpy()
    if self_retrieval:
        np.fill_diagonal(sims, -np.inf)
    correct = (gl[sims.argmax(axis=1)] == ql).astype(float)
    per_species = {c: float(correct[ql == c].mean()) for c in np.unique(ql)}
    return float(np.mean(list(per_species.values()))), per_species


# ── stats (monolith L2929-2950) ──────────────────────────────────────────────
def bootstrap(values, n_boot=1000, ci=0.95, seed=42):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot bootstrap an empty value list.")
    rng = np.random.RandomState(seed)
    boots = [np.mean(arr[rng.choice(len(arr), len(arr), replace=True)]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(np.mean(arr)), float(lo), float(hi)


def wilcoxon_test(a, b):
    from scipy.stats import wilcoxon as _wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a - b, 0):
        return 0.0, 1.0, "n.s."
    try:
        stat, p = _wilcoxon(a, b, zero_method="wilcox")
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."
        return float(stat), float(p), sig
    except Exception:
        return None, None, "n.a."


def paired_wilcoxon(per_species_a, per_species_b):
    common = sorted(set(per_species_a) & set(per_species_b))
    a = np.array([per_species_a[c] for c in common])
    b = np.array([per_species_b[c] for c in common])
    stat, p, sig = wilcoxon_test(a, b)
    return {"delta": float(a.mean() - b.mean()), "p_value": p, "sig": sig, "n": len(common)}


# ── SC-URD memory-mode scoring (monolith L2786-2871) ─────────────────────────
SCURD_VALID_MODES = {"raw", "centered", "logmeanexp", "prototype_mix",
                     "centered_logmeanexp", "centered_prototype_mix"}


def _scurd_mode_config(mode, prototype_mix_alpha=None):
    if mode not in SCURD_VALID_MODES:
        raise ValueError(f"Invalid SC-URD memory mode: {mode!r}")
    alpha = 0.5 if mode in {"prototype_mix", "centered_prototype_mix"} else None
    if prototype_mix_alpha is not None and mode in {"prototype_mix", "centered_prototype_mix"}:
        alpha = float(prototype_mix_alpha)
    return {"center": mode in {"centered", "centered_logmeanexp", "centered_prototype_mix"},
            "class_balance": mode in {"logmeanexp", "centered_logmeanexp"},
            "prototype_mix_alpha": alpha}


def _macro_from_preds(preds, labels):
    labels, preds = np.asarray(labels), np.asarray(preds)
    per_sp = {sp: float((preds[labels == sp] == sp).mean()) for sp in sorted(np.unique(labels))}
    return float(np.mean(list(per_sp.values()))), per_sp


def scurd_class_scores(query_embs, gallery_embs, gallery_labels, top_m=50, tau=0.07,
                       class_order=None, mode="centered", prototype_mix_alpha=None):
    cfg = _scurd_mode_config(mode, prototype_mix_alpha=prototype_mix_alpha)
    q = np.asarray(query_embs, dtype=np.float32)
    g = np.asarray(gallery_embs, dtype=np.float32)
    gl = np.asarray(gallery_labels)
    if cfg["center"]:
        mu = g.mean(axis=0, keepdims=True)
        q, g = q - mu, g - mu
    q, g = _norm(q), _norm(g)
    class_order = np.asarray(sorted(np.unique(gl)) if class_order is None else class_order)
    class_to_col = {c: i for i, c in enumerate(class_order)}
    sims = q @ g.T
    top_m_eff = min(int(top_m), sims.shape[1])
    top_idx = np.argpartition(-sims, top_m_eff - 1, axis=1)[:, :top_m_eff]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    sample_scores = np.full((len(q), len(class_order)), -1e9, dtype=np.float64)
    tau = max(float(tau), 1e-6)
    for i in range(len(q)):
        top_labels = gl[top_idx[i]]
        for c in np.unique(top_labels):
            if c not in class_to_col:
                continue
            vals = top_sims[i, top_labels == c] / tau
            m = vals.max()
            score = m + np.log(np.exp(vals - m).sum())
            if cfg["class_balance"]:
                score -= np.log(max(1, np.sum(gl == c)))
            sample_scores[i, class_to_col[c]] = score
    if cfg["prototype_mix_alpha"] is None:
        logits = sample_scores
    else:
        proto = []
        for c in class_order:
            mask = gl == c
            proto.append(g[mask].mean(axis=0) if mask.any() else np.zeros(g.shape[1], dtype=np.float32))
        proto = _norm(np.asarray(proto, dtype=np.float32))
        proto_scores = (q @ proto.T) / tau
        finite_sample = np.where(sample_scores < -1e8, proto_scores, sample_scores)
        alpha = float(cfg["prototype_mix_alpha"])
        logits = alpha * finite_sample + (1.0 - alpha) * proto_scores
    return _softmax_np(logits, axis=1), class_order


def scurd_retrieval_eval(query_embs, query_labels, gallery_embs, gallery_labels,
                         top_m=50, tau=0.07, mode="centered", prototype_mix_alpha=None):
    scores, classes = scurd_class_scores(query_embs, gallery_embs, gallery_labels,
                                         top_m=top_m, tau=tau, mode=mode,
                                         prototype_mix_alpha=prototype_mix_alpha)
    preds = classes[np.argmax(scores, axis=1)]
    macro, per_sp = _macro_from_preds(preds, query_labels)
    return {"mean": macro, "per_species": per_sp}


def method_retrieval_eval(md, query_embs, query_labels, gallery_embs, gallery_labels):
    if md.get("scurd_mode"):
        return scurd_retrieval_eval(query_embs, query_labels, gallery_embs, gallery_labels,
                                    mode=md["scurd_mode"], prototype_mix_alpha=md.get("scurd_alpha"))
    macro, per_sp = recall_at_1_per_species(query_embs, query_labels, gallery_embs, gallery_labels)
    return {"mean": float(macro), "per_species": per_sp}


# ── exp4 cache loader (cache-only; monolith L2457-2505) ──────────────────────
def load_exp4_cache():
    """Load VN26/SWI-scale embeddings from exp4_embedding_cache_v3.npz into
    swi_embs[(method,scale)] / vn26_embs[(method,mag)] (+ pooled). Returns
    (swi_embs, vn26_embs); empty if the cache is absent (RQ4 then skipped)."""
    swi_embs, vn26_embs = {}, {}
    path = config.ROOT_PATH / config.EXP4_CACHE_NAME
    if not path.exists():
        print(f"⚠️  exp4 cache missing: {path} — RQ4/VN26 will be skipped (run with RUN_HEAVY to build).")
        return swi_embs, vn26_embs
    c4 = np.load(path, allow_pickle=False)
    files = set(c4.files)
    for method in METHODS_ALL:
        for scale in SWI_SCALES:
            k = _e4key("swi", method, scale)
            if k in files and f"{k}_lbl" in files:
                swi_embs[(method, scale)] = (c4[k], np.array([canonical_label(x) for x in c4[f"{k}_lbl"]]))
        for mag in VN26_MAGS:
            k = _e4key("vn26", method, mag)
            if k in files and f"{k}_lbl" in files:
                vn26_embs[(method, mag)] = (c4[k], np.array([canonical_label(x) for x in c4[f"{k}_lbl"]]))
    for method in METHODS_ALL:
        if all((method, s) in swi_embs for s in SWI_SCALES):
            swi_embs[(method, "pool")] = (
                np.concatenate([swi_embs[(method, s)][0] for s in SWI_SCALES]),
                np.concatenate([swi_embs[(method, s)][1] for s in SWI_SCALES]))
        if all((method, m) in vn26_embs for m in VN26_MAGS):
            vn26_embs[(method, "all")] = (
                np.concatenate([vn26_embs[(method, m)][0] for m in VN26_MAGS]),
                np.concatenate([vn26_embs[(method, m)][1] for m in VN26_MAGS]))
    print(f"Loaded exp4 cache: {len(swi_embs)} SWI-scale, {len(vn26_embs)} VN26-mag arrays")
    return swi_embs, vn26_embs


# ── OOD species splits (monolith/engine — single source) ────────────────────
def ood_test_mask(labels_ood):
    """Held-out OOD test mask (drops the 20% used for ODIN val tuning), seed=42."""
    species = sorted(set(labels_ood))
    rng = np.random.RandomState(42)
    rng.shuffle(species)
    n_val = max(1, int(0.2 * len(species)))
    test_species = set(species[n_val:])
    return np.asarray([s in test_species for s in labels_ood])


def ood_val_test_masks(labels_ood):
    """(val_mask, test_mask) over OOD images: 20% val species (ODIN/OpenMax tuning),
    80% test species (reported), seed=42 — matches monolith ood_val/test split."""
    species = sorted(set(labels_ood))
    rng = np.random.RandomState(42)
    rng.shuffle(species)
    n_val = max(1, int(0.2 * len(species)))
    val_species, test_species = set(species[:n_val]), set(species[n_val:])
    labels_ood = np.asarray(labels_ood)
    return (np.array([s in val_species for s in labels_ood]),
            np.array([s in test_species for s in labels_ood]))


def id_species_splits():
    """Per public-ID species → {query:[paths], pool:[paths]} from ID_images_expanded.
    Uses one RandomState(42) iterated over sorted species (matches monolith L2188)."""
    df = pd.read_csv(config.ID_IMAGES_CSV)
    canon = df["label"].apply(canonical_label)
    rng = np.random.RandomState(42)
    splits = {}
    for sp in sorted(set(canon)):
        sp_df = df[canon == sp].reset_index(drop=True)
        idx = np.arange(len(sp_df)); rng.shuffle(idx)
        n_query = min(10, len(sp_df) // 2)
        splits[sp] = {"query": sp_df.iloc[idx[:n_query]]["file_path"].tolist(),
                      "pool": sp_df.iloc[idx[n_query:]]["file_path"].tolist()}
    return splits


def ood_species_splits():
    """Top-50 OOD species → {query_indices, pool_indices} into OOD_images_expanded
    row order (== embs_ood_* order). seed=42, n_query=min(10, n//2)."""
    df = pd.read_csv(config.OOD_IMAGES_CSV)
    df["canon"] = df["label"].apply(canonical_label)
    counts = df.groupby("canon").size().sort_values(ascending=False)
    top50 = counts.head(50).index.tolist()
    rng = np.random.RandomState(42)
    splits = {}
    for sp in top50:
        idx = np.where(df["canon"].values == sp)[0]
        rng.shuffle(idx)
        n_query = min(10, len(idx) // 2)
        splits[sp] = {"query_indices": idx[:n_query].tolist(), "pool_indices": idx[n_query:].tolist()}
    return splits, top50


# ── the registry builder ─────────────────────────────────────────────────────
_VARIANT_E4 = {"ArcFace-ViT": "Var_ViTB_Arc", "ArcFace-RN50": "Var_RN50_Arc",
               "SupCon-ConvNeXt": "Var_CvNxt_SupCon"}


def build_M(scope="id_only", with_exp4=True, with_scurd=True, emb=None, device=None):
    """Rebuild the monolith method registry from the full-954 cache at `scope`.

    Returns (M, ctx) where ctx has labels_id/labels_swi/labels_ood, gallery_mask,
    gal_labels, swi_embs/vn26_embs (exp4), and id/ood split helpers.
    """
    from ..gallery import build_gallery_mask
    device = device or config.resolve_device()
    if emb is None:
        emb = np.load(config.FULL954_CACHE_PATH, allow_pickle=False)
    labels_id = np.array([canonical_label(x) for x in emb["labels_id_dinov2"]])
    labels_swi = np.array([canonical_label(x) for x in emb["labels_swi_dinov2"]])
    labels_ood = np.array([canonical_label(x) for x in emb["labels_ood_dinov2"]])
    ce_mask = emb["swi_in_ce_train"] if "swi_in_ce_train" in emb.files else None
    mask, gal_labels, scope_label = build_gallery_mask(labels_swi, set(labels_id), scope=scope,
                                                       ce_train_mask=ce_mask)

    def G(key):  # masked SWI gallery for an embedding key
        return emb[key][mask]

    # PDA domain stats (monolith L2145-2177), computed at the active gallery.
    gal_arc = G("embs_swi_arc")
    ood_arc = emb["embs_ood_arc"]
    mu_swi, mu_pub = gal_arc.mean(0), ood_arc.mean(0)
    shift_vec = mu_swi - mu_pub
    lam = 1e-4
    xo, xs = ood_arc - mu_pub, gal_arc - mu_swi
    co = (xo.T @ xo) / len(ood_arc) + lam * np.eye(ood_arc.shape[1])
    cs = (xs.T @ xs) / len(gal_arc) + lam * np.eye(gal_arc.shape[1])

    def _isqrt(A):
        v, V = np.linalg.eigh(A); v = np.maximum(v, 1e-8); return V @ np.diag(1/np.sqrt(v)) @ V.T

    def _sqrt(A):
        v, V = np.linalg.eigh(A); v = np.maximum(v, 0.0); return V @ np.diag(np.sqrt(v)) @ V.T

    coral_t = _isqrt(co) @ _sqrt(cs)
    pda_shift = lambda x: _norm(x + shift_vec)
    pda_coral = lambda x: _norm((x - mu_pub) @ coral_t + mu_swi)
    pda_aqe = lambda x, g: _norm(0.5 * x + 0.5 * g[np.argsort(-(x @ g.T), axis=1)[:, :10]].mean(axis=1))

    swi_embs, vn26_embs = (load_exp4_cache() if with_exp4 else ({}, {}))

    def _pool(d):
        if all(s in d for s in SWI_SCALES):
            return {**d, "pool": (np.concatenate([d[s][0] for s in SWI_SCALES]),
                                  np.concatenate([d[s][1] for s in SWI_SCALES]))}
        return d

    def _entry(id_e, gal_e, ood_e=None, swi_sc=None, vn26_mg=None, gal_lbl=None, **extra):
        return {"id": id_e, "gal": (gal_e, gal_lbl if gal_lbl is not None else gal_labels),
                "ood": ood_e, "swi_scales": swi_sc or {}, "vn26_mags": vn26_mg or {}, **extra}

    M = {}
    M["ImageNet"] = _entry(emb["embs_id_imagenet"], G("embs_swi_imagenet"))
    M["CLIP"] = _entry(emb["embs_id_clip"], G("embs_swi_clip"),
                       gal_lbl=np.array([canonical_label(x) for x in emb["labels_swi_clip"]])[mask])
    gal_dinov2 = G("embs_swi_dinov2")
    M["DINOv2"] = _entry(emb["embs_id_dinov2"], gal_dinov2, ood_e=emb["embs_ood_dinov2"],
                         swi_sc=_pool({s: swi_embs[("DINOv2", s)] for s in SWI_SCALES if ("DINOv2", s) in swi_embs}),
                         vn26_mg={m: vn26_embs[("DINOv2", m)] for m in VN26_MAGS if ("DINOv2", m) in vn26_embs})
    M["CE-Full"] = _entry(emb["embs_id_ce_full_norm"], G("embs_swi_ce_full_norm"), ood_e=emb["embs_ood_ce_full_norm"],
                          swi_sc=_pool({s: swi_embs[("CE-Full", s)] for s in SWI_SCALES if ("CE-Full", s) in swi_embs}),
                          vn26_mg={m: vn26_embs[("CE-Full", m)] for m in VN26_MAGS if ("CE-Full", m) in vn26_embs})
    M["CE-Narrow"] = _entry(emb["embs_id_ce_narrow_norm"], G("embs_swi_ce_narrow_norm"), ood_e=emb["embs_ood_ce_narrow_norm"],
                            swi_sc=_pool({s: swi_embs[("CE-Narrow", s)] for s in SWI_SCALES if ("CE-Narrow", s) in swi_embs}),
                            vn26_mg={m: vn26_embs[("CE-Narrow", m)] for m in VN26_MAGS if ("CE-Narrow", m) in vn26_embs})
    for name, id_key, swi_key, ood_key, e4 in [
        ("ArcFace-557", "embs_id_arc", "embs_swi_arc", "embs_ood_arc", "ArcFace-557"),
        ("ArcFace-954", "embs_id_arc954", "embs_swi_arc954", "embs_ood_arc954", "ArcFace-954"),
        ("ProtoNet", "embs_id_proto", "embs_swi_proto", "embs_ood_proto", None),
        ("ArcFace-ViT", "embs_id_Var_ViTB_Arc", "embs_swi_Var_ViTB_Arc", "embs_ood_Var_ViTB_Arc", "ArcFace-ViT"),
        ("ArcFace-RN50", "embs_id_Var_RN50_Arc", "embs_swi_Var_RN50_Arc", "embs_ood_Var_RN50_Arc", "ArcFace-RN50"),
        ("SupCon", "embs_id_Var_CvNxt_SupCon", "embs_swi_Var_CvNxt_SupCon", "embs_ood_Var_CvNxt_SupCon", "SupCon-ConvNeXt"),
    ]:
        if id_key not in emb.files:
            continue
        swi_sc = _pool({s: swi_embs[(e4, s)] for s in SWI_SCALES if e4 and (e4, s) in swi_embs})
        vn26_mg = {m: vn26_embs[(e4, m)] for m in VN26_MAGS if e4 and (e4, m) in vn26_embs}
        M[name] = _entry(emb[id_key], G(swi_key), ood_e=emb[ood_key], swi_sc=swi_sc, vn26_mg=vn26_mg)

    # PDA variants (transforms of ArcFace).
    M["ArcFace+PDA"] = _entry(pda_shift(emb["embs_id_arc"]), gal_arc, ood_e=pda_shift(ood_arc))
    M["ArcFace+CORAL"] = _entry(pda_coral(emb["embs_id_arc"]), gal_arc, ood_e=pda_coral(ood_arc))
    M["ArcFace+αQE"] = _entry(pda_aqe(emb["embs_id_arc"], gal_arc), gal_arc, ood_e=pda_aqe(ood_arc, gal_arc))

    # Fusion (ArcFace ⊕ DINOv2).
    fus_sc = _pool({s: (_fuse(swi_embs[("ArcFace-557", s)][0], swi_embs[("DINOv2", s)][0]), swi_embs[("ArcFace-557", s)][1])
                    for s in SWI_SCALES if ("ArcFace-557", s) in swi_embs and ("DINOv2", s) in swi_embs})
    M["Fusion"] = _entry(_fuse(emb["embs_id_arc"], emb["embs_id_dinov2"]), _fuse(gal_arc, gal_dinov2),
                         ood_e=_fuse(ood_arc, emb["embs_ood_dinov2"]), swi_sc=fus_sc,
                         vn26_mg={m: (_fuse(vn26_embs[("ArcFace-557", m)][0], vn26_embs[("DINOv2", m)][0]), vn26_embs[("ArcFace-557", m)][1])
                                  for m in VN26_MAGS if ("ArcFace-557", m) in vn26_embs and ("DINOv2", m) in vn26_embs})

    # SC-URD (projected DINOv2) at the active gallery.
    if with_scurd and config.SCURD_MAIN_CKPT.exists():
        from ..scurd import load_scurd_model, project_np
        model, _ = load_scurd_model(config.SCURD_MAIN_CKPT, in_dim=emb["embs_id_dinov2"].shape[1], device=device)
        pj = lambda a: project_np(model, a, device)
        M["SC-URD"] = _entry(pj(emb["embs_id_dinov2"]), pj(gal_dinov2), ood_e=pj(emb["embs_ood_dinov2"]),
                             swi_sc=_pool({s: (pj(swi_embs[("DINOv2", s)][0]), swi_embs[("DINOv2", s)][1]) for s in SWI_SCALES if ("DINOv2", s) in swi_embs}),
                             vn26_mg={m: (pj(vn26_embs[("DINOv2", m)][0]), vn26_embs[("DINOv2", m)][1]) for m in VN26_MAGS if ("DINOv2", m) in vn26_embs},
                             scurd_mode=config.SCURD_MODE, scurd_alpha=None)

    ctx = {"emb": emb, "labels_id": labels_id, "labels_swi": labels_swi, "labels_ood": labels_ood,
           "gallery_mask": mask, "gal_labels": gal_labels, "scope": scope_label,
           "swi_embs": swi_embs, "vn26_embs": vn26_embs, "device": device}
    print(f"Method registry built (scope={scope_label}): {list(M.keys())}")
    return M, ctx

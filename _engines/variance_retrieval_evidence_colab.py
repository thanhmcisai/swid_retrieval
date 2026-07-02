# -*- coding: utf-8 -*-
"""
variance_retrieval_evidence_colab.py

Reviewer-facing variance/retrieval evidence:
  1) Retrieval mAP/MRR/Recall@K from existing embedding caches.
  2) Gallery-reference sampling variance for K-shot deployment.
  3) SC-URD training-seed sensitivity. Missing seed checkpoints can be
     trained from cached DINOv2 meta-train embeddings, then evaluated.

The first two parts do not load images or train models. The seed-sensitivity
part also does not load images; if enabled, it trains only the small SC-URD
residual adapter from cached DINOv2 embeddings. It does not retrain DINOv2,
ArcFace, SupCon, or any image backbone.

Seed checkpoints use explicit seed suffixes, e.g.:

    paper_reframe/research_directions/sc_urd_checkpoint_scurd_r01_e20_seed42_v2.pt

Colab one-cell usage:

    import os
    os.environ["ROOT_PATH"] = "/content/drive/MyDrive/NCS"
    os.environ["RUN_MAP"] = "1"
    os.environ["RUN_HEADLINE_RECOMPUTE"] = "1"
    os.environ["RUN_GALLERY_RESAMPLING"] = "1"
    os.environ["RUN_SCURD_SEED_SENSITIVITY"] = "1"
    os.environ["RUN_TRAIN_SCURD_SEEDS"] = "1"
    os.environ["GALLERY_SCOPE"] = "full_swi"
    os.environ["SCURD_TRAIN_SEEDS"] = "42,43,44"
    os.environ["N_GALLERY_REPEATS"] = "100"
    exec(open("/content/drive/MyDrive/NCS/variance_retrieval_evidence_colab.py").read())

Outputs are written under:
    /content/drive/MyDrive/NCS/results/paper_reframe/variance_evidence
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_PATH = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT_PATH / "results/paper_reframe"))
OUT_DIR = Path(os.environ.get("OUT_DIR", RESULTS_DIR / "variance_evidence"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_CACHE_PATH = ROOT_PATH / os.environ.get("EMB_CACHE_NAME", "embedding_cache_v3.npz")
EXP4_CACHE_PATH = ROOT_PATH / os.environ.get("EXP4_CACHE_NAME", "exp4_embedding_cache_v3.npz")
SCURD_PROJ_CACHE = (
    RESULTS_DIR
    / "research_directions"
    / os.environ.get("SCURD_PROJ_CACHE_NAME", "sc_urd_eval_embeddings_scurd_r01_e20_recomputed_v2.npz")
)
SCURD_CKPT_DIR = RESULTS_DIR / "research_directions"
SCURD_CACHE_VERSION = os.environ.get("SC_URD_CACHE_VERSION", "v2")
RESEARCH_CACHE_VERSION = os.environ.get("RESEARCH_CACHE_VERSION", "v2")
SCURD_META_CACHE = SCURD_CKPT_DIR / os.environ.get("SCURD_META_CACHE_NAME", f"urd_v2_meta_dinov2_embeddings_{RESEARCH_CACHE_VERSION}.npz")
SCURD_MAIN_CKPT = SCURD_CKPT_DIR / os.environ.get("SCURD_MAIN_CKPT_NAME", f"sc_urd_checkpoint_scurd_r01_e20_{SCURD_CACHE_VERSION}.pt")
SCURD_SEED_CKPT_GLOB = os.environ.get("SCURD_SEED_CKPT_GLOB", f"sc_urd_checkpoint_scurd_r01_e20_seed*_{SCURD_CACHE_VERSION}.pt")

RUN_MAP = os.environ.get("RUN_MAP", "1") == "1"
RUN_HEADLINE_RECOMPUTE = os.environ.get("RUN_HEADLINE_RECOMPUTE", "1") == "1"
RUN_GALLERY_RESAMPLING = os.environ.get("RUN_GALLERY_RESAMPLING", "1") == "1"
RUN_SCURD_SEED_SENSITIVITY = os.environ.get("RUN_SCURD_SEED_SENSITIVITY", "1") == "1"
RUN_TRAIN_SCURD_SEEDS = os.environ.get("RUN_TRAIN_SCURD_SEEDS", "1") == "1"
RUN_REVIEWER_GAP_FULL_GALLERY = os.environ.get("RUN_REVIEWER_GAP_FULL_GALLERY", "1") == "1"
RUN_RQ5_FULL_GALLERY = os.environ.get("RUN_RQ5_FULL_GALLERY", "1") == "1"
N_GALLERY_REPEATS = int(os.environ.get("N_GALLERY_REPEATS", "100"))
N_BOOT = int(os.environ.get("N_BOOT", "2000"))
SEED = int(os.environ.get("SEED", "42"))
GALLERY_SCOPE = os.environ.get("GALLERY_SCOPE", "full_swi").strip().lower()
MIN_FULL_GALLERY_SPECIES = int(os.environ.get("MIN_FULL_GALLERY_SPECIES", "900"))

SCURD_MODE = os.environ.get("SCURD_MAIN_MODE", "centered").strip() or "centered"
SCURD_TAU = float(os.environ.get("SCURD_TAU", "0.07"))
SCURD_TOP_M = int(os.environ.get("SCURD_TOP_M", "50"))
SCURD_BETA = float(os.environ.get("SCURD_BETA", "0.1"))
SCURD_TRAIN_EPOCHS = int(os.environ.get("SCURD_TRAIN_EPOCHS", "20"))
SCURD_TRAIN_EPISODES = int(os.environ.get("SCURD_TRAIN_EPISODES", "500"))
SCURD_TRAIN_LAMBDA_CONS = float(os.environ.get("SCURD_TRAIN_LAMBDA_CONS", "0.5"))
SCURD_TRAIN_LR = float(os.environ.get("SCURD_TRAIN_LR", "1e-3"))
SCURD_WEIGHT_DECAY = float(os.environ.get("SCURD_WEIGHT_DECAY", "1e-4"))
SCURD_N_WAY = int(os.environ.get("SCURD_N_WAY", "16"))
SCURD_K_SUPPORT = int(os.environ.get("SCURD_K_SUPPORT", "5"))
SCURD_Q_QUERY = int(os.environ.get("SCURD_Q_QUERY", "4"))
SCURD_FORCE_RETRAIN_SEEDS = os.environ.get("SCURD_FORCE_RETRAIN_SEEDS", "0") == "1"
SCURD_TRAIN_SEEDS = [
    int(x.strip()) for x in os.environ.get("SCURD_TRAIN_SEEDS", "42,43,44").split(",")
    if x.strip()
]
DEVICE = os.environ.get("DEVICE", "cuda")


def canonical_label(s):
    return str(s).replace(" ", "_").lower().strip()


def save_csv(rows, name):
    path = OUT_DIR / name
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")
    return df


def save_json(obj, name):
    path = OUT_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"saved {path}")


def save_json_to_results(obj, name):
    path = OUT_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"saved {path}")
    return path


def protocol_audit_rows(methods, labels_id, labels_ood):
    rows = []
    for name, md in methods.items():
        rows.append({
            "experiment_group": "main_gallery",
            "method": name,
            "query": "public_id",
            "gallery": GALLERY_SCOPE,
            "gallery_images": int(len(md["gal_labels"])),
            "gallery_species": int(len(set(md["gal_labels"]))),
            "public_id_species": int(len(set(labels_id))),
            "ood_species": int(len(set(labels_ood))),
            "is_full_swi_gallery": bool(GALLERY_SCOPE == "full_swi" and len(set(md["gal_labels"])) >= MIN_FULL_GALLERY_SPECIES),
            "notes": "Used by RQ1 retrieval, distance OOD, RQ3 old-gallery extension and RQ5 E1/OOD/E3 axes.",
        })
    rows.append({
        "experiment_group": "gallery_strategy_B",
        "method": "all",
        "query": "public_id",
        "gallery": "public_only_kshot",
        "gallery_images": None,
        "gallery_species": int(len(set(labels_id))),
        "public_id_species": int(len(set(labels_id))),
        "ood_species": int(len(set(labels_ood))),
        "is_full_swi_gallery": False,
        "notes": "Intentional public-only enrollment scenario; not a full-SWI gallery experiment.",
    })
    rows.append({
        "experiment_group": "vn26_transfer",
        "method": "all",
        "query": "vn26",
        "gallery": "swi_pool_or_vn26_magnification_specific",
        "gallery_images": None,
        "gallery_species": None,
        "public_id_species": int(len(set(labels_id))),
        "ood_species": int(len(set(labels_ood))),
        "is_full_swi_gallery": None,
        "notes": "VN26 experiments use their own SWI_pool/VN26 galleries and are not corrected by the public-ID gallery mask.",
    })
    return rows


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fuse(a, b):
    return norm(np.concatenate([norm(a), norm(b)], axis=1))


def macro_from_preds(preds, labels):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_sp = {}
    for sp in sorted(np.unique(labels)):
        m = labels == sp
        per_sp[sp] = float((preds[m] == labels[m]).mean())
    return float(np.mean(list(per_sp.values()))), per_sp


def bootstrap_values(values, n_boot=1000, seed=42):
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    rng = np.random.RandomState(seed)
    boots = [float(np.mean(vals[rng.choice(len(vals), len(vals), replace=True)])) for _ in range(n_boot)]
    return {
        "mean": float(np.mean(vals)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n": int(len(vals)),
    }


def nn_macro_top1(q_emb, q_labels, g_emb, g_labels):
    q = norm(q_emb)
    g = norm(g_emb)
    gl = np.asarray(g_labels)
    preds = gl[(q @ g.T).argmax(axis=1)]
    return macro_from_preds(preds, q_labels)[0]


def _prototype_macro_top1(q_emb, q_labels, g_emb, g_labels, centered=False):
    """Nearest class-centroid macro-top-1 (one centroid per gallery species).

    Anchor-matched analogue of CE's one-weight-per-class — the fair RQ1 headline.
    Self-contained fallback; mirrors swid_retrieval.experiments.rq1_native.
    """
    q = np.asarray(q_emb, dtype=np.float32)
    g = np.asarray(g_emb, dtype=np.float32)
    if centered:
        mu = g.mean(axis=0, keepdims=True)
        q, g = q - mu, g - mu
    q, g = norm(q), norm(g)
    gl = np.asarray(g_labels)
    classes = np.array(sorted(set(gl.tolist())))
    centroids = norm(np.stack([g[gl == c].mean(axis=0) for c in classes]))
    preds = classes[(q @ centroids.T).argmax(axis=1)]
    return macro_from_preds(preds, q_labels)[0]


def _carry_forward_native(source_results_dir):
    """Gallery-independent native CE accuracy from the original rq1_paradigm.json."""
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


def nn_eval(q_emb, q_labels, g_emb, g_labels):
    q = norm(q_emb)
    g = norm(g_emb)
    gl = np.asarray(g_labels)
    preds = gl[(q @ g.T).argmax(axis=1)]
    mean, per_sp = macro_from_preds(preds, q_labels)
    return {"mean": mean, "per_species": per_sp, "preds": preds}


def compute_auroc(scores_id, scores_ood):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    s = np.concatenate([scores_id, scores_ood])
    return float(roc_auc_score(y, s))


def fpr_at_tpr(scores_id, scores_ood, tpr_target=0.95):
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    s = np.concatenate([scores_id, scores_ood])
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, tpr_target)
    return float(fpr[min(idx, len(fpr) - 1)])


def bootstrap_fpr95_difference(scores_id, scores_ood, base_id, base_ood, n_boot=1000, seed=42):
    sid = np.asarray(scores_id, dtype=float)
    sod = np.asarray(scores_ood, dtype=float)
    bid = np.asarray(base_id, dtype=float)
    bod = np.asarray(base_ood, dtype=float)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        ii = rng.choice(len(sid), len(sid), replace=True)
        oo = rng.choice(len(sod), len(sod), replace=True)
        vals.append(fpr_at_tpr(bid[ii], bod[oo]) - fpr_at_tpr(sid[ii], sod[oo]))
    return {
        "mean_reduction": float(np.mean(vals)),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
    }


def distance_scores(query_embs, gallery_embs):
    q = norm(query_embs)
    g = norm(gallery_embs)
    return 1.0 - (q @ g.T).max(axis=1)


def _scurd_mode_config(mode):
    return {
        "center": mode in {"centered", "centered_logmeanexp", "centered_prototype_mix"},
        "class_balance": mode in {"logmeanexp", "centered_logmeanexp"},
        "prototype_mix_alpha": 0.5 if mode in {"prototype_mix", "centered_prototype_mix"} else None,
    }


def scurd_class_scores(query_embs, gallery_embs, gallery_labels, mode=SCURD_MODE, tau=SCURD_TAU, top_m=SCURD_TOP_M):
    cfg = _scurd_mode_config(mode)
    q = np.asarray(query_embs, dtype=np.float32)
    g = np.asarray(gallery_embs, dtype=np.float32)
    gl = np.asarray(gallery_labels)
    if cfg["center"]:
        mu = g.mean(axis=0, keepdims=True)
        q = q - mu
        g = g - mu
    q = norm(q)
    g = norm(g)
    classes = np.asarray(sorted(np.unique(gl)))
    class_to_col = {c: i for i, c in enumerate(classes)}
    sims = q @ g.T
    top_m_eff = min(int(top_m), sims.shape[1])
    top_idx = np.argpartition(-sims, top_m_eff - 1, axis=1)[:, :top_m_eff]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)

    logits = np.full((len(q), len(classes)), -1e9, dtype=np.float64)
    tau = max(float(tau), 1e-6)
    for i in range(len(q)):
        labs = gl[top_idx[i]]
        for c in np.unique(labs):
            vals = top_sims[i, labs == c] / tau
            m = vals.max()
            score = m + np.log(np.exp(vals - m).sum())
            if cfg["class_balance"]:
                score -= np.log(max(1, np.sum(gl == c)))
            logits[i, class_to_col[c]] = score

    if cfg["prototype_mix_alpha"] is not None:
        proto = norm(np.asarray([g[gl == c].mean(axis=0) for c in classes], dtype=np.float32))
        proto_scores = (q @ proto.T) / tau
        sample_scores = np.where(logits < -1e8, proto_scores, logits)
        alpha = float(cfg["prototype_mix_alpha"])
        logits = alpha * sample_scores + (1.0 - alpha) * proto_scores
    evidence_sim = np.max(top_sims, axis=1)
    return logits, classes, evidence_sim


def scurd_macro_top1(q_emb, q_labels, g_emb, g_labels, mode=SCURD_MODE):
    logits, classes, _ = scurd_class_scores(q_emb, g_emb, g_labels, mode=mode)
    preds = classes[logits.argmax(axis=1)]
    return macro_from_preds(preds, q_labels)[0]


def scurd_eval(q_emb, q_labels, g_emb, g_labels, mode=SCURD_MODE):
    logits, classes, evidence_sim = scurd_class_scores(q_emb, g_emb, g_labels, mode=mode)
    preds = classes[logits.argmax(axis=1)]
    mean, per_sp = macro_from_preds(preds, q_labels)
    return {"mean": mean, "per_species": per_sp, "preds": preds, "logits": logits, "classes": classes,
            "evidence_sim": evidence_sim}


def centered_sample_space(q_emb, g_emb):
    q = np.asarray(q_emb, dtype=np.float32)
    g = np.asarray(g_emb, dtype=np.float32)
    if SCURD_MODE in {"centered", "centered_logmeanexp", "centered_prototype_mix"}:
        mu = g.mean(axis=0, keepdims=True)
        q = q - mu
        g = g - mu
    return norm(q), norm(g)


def retrieval_quality(q_emb, q_labels, g_emb, g_labels, centered=False):
    ql = np.asarray(q_labels)
    gl = np.asarray(g_labels)
    if centered:
        q, g = centered_sample_space(q_emb, g_emb)
    else:
        q, g = norm(q_emb), norm(g_emb)
    sims = q @ g.T
    order = np.argsort(-sims, axis=1)
    ap_values, rr_values = [], []
    hit_at = {1: [], 5: [], 10: []}
    purity5 = []
    for i in range(len(q)):
        relevant = gl == ql[i]
        n_rel = int(relevant.sum())
        if n_rel == 0:
            continue
        ranked_rel = relevant[order[i]]
        rel_pos = np.where(ranked_rel)[0]
        first = int(rel_pos[0]) + 1
        rr_values.append(1.0 / first)
        precisions = []
        hits = 0
        for rank0, is_rel in enumerate(ranked_rel):
            if is_rel:
                hits += 1
                precisions.append(hits / float(rank0 + 1))
        ap_values.append(float(np.sum(precisions) / max(1, n_rel)))
        for k in hit_at:
            hit_at[k].append(float(ranked_rel[:k].any()))
        purity5.append(float(ranked_rel[:5].mean()))

    return {
        "mAP_query": float(np.mean(ap_values)),
        "MRR_query": float(np.mean(rr_values)),
        "Hit@1_query": float(np.mean(hit_at[1])),
        "Hit@5_query": float(np.mean(hit_at[5])),
        "Hit@10_query": float(np.mean(hit_at[10])),
        "Purity@5_query": float(np.mean(purity5)),
        "n_queries": int(len(ap_values)),
    }


def load_embedding_artifacts():
    if not EMB_CACHE_PATH.exists():
        raise FileNotFoundError(EMB_CACHE_PATH)
    emb = np.load(EMB_CACHE_PATH, allow_pickle=False)
    labels_id = np.asarray([canonical_label(x) for x in emb["labels_id_dinov2"]])
    labels_swi = np.asarray([canonical_label(x) for x in emb["labels_swi_dinov2"]])
    labels_ood = np.asarray([canonical_label(x) for x in emb["labels_ood_dinov2"]])
    id_species = set(labels_id)
    if GALLERY_SCOPE in {"full", "full_swi", "all_swi", "954", "954_swi"}:
        gallery_mask = np.ones(len(labels_swi), dtype=bool)
        gallery_scope_label = "full_swi"
    elif GALLERY_SCOPE in {"ce_train", "ce_train_only", "cetrain"}:
        # Fairness robustness control: full 954-species gallery restricted to the
        # exact images CE-Full was trained on (so retrieval gets no reference
        # image the classifier did not learn from).
        if "swi_in_ce_train" not in emb.files:
            raise KeyError(
                f"{EMB_CACHE_PATH} has no 'swi_in_ce_train' mask; rebuild the "
                f"full-954 cache (swid_retrieval.embeddings.build_full954) which "
                f"writes it, then set GALLERY_SCOPE=ce_train."
            )
        gallery_mask = np.asarray(emb["swi_in_ce_train"], dtype=bool)
        gallery_scope_label = "ce_train"
    elif GALLERY_SCOPE in {"id", "id_only", "public_id", "24", "24_id"}:
        gallery_mask = np.asarray([x in id_species for x in labels_swi])
        gallery_scope_label = "id_only"
    else:
        raise ValueError(
            f"Unknown GALLERY_SCOPE={GALLERY_SCOPE!r}; use full_swi, ce_train or id_only."
        )
    uses_full_rows = gallery_scope_label in {"full_swi", "ce_train"}
    gal_labels = labels_swi[gallery_mask]
    print(
        f"Gallery scope={gallery_scope_label}: "
        f"{int(gallery_mask.sum())} SWI images, {len(set(gal_labels))} species"
    )
    # full_swi and ce_train both span all ~954 species (CE-train keeps >=1 image
    # per species), so the species-count floor applies to both.
    if uses_full_rows and len(set(gal_labels)) < MIN_FULL_GALLERY_SPECIES:
        raise RuntimeError(
            f"Full-gallery protocol violation ({gallery_scope_label}): got only "
            f"{len(set(gal_labels))} gallery species, expected at least "
            f"{MIN_FULL_GALLERY_SPECIES}."
        )

    methods = {
        "DINOv2": {
            "id": emb["embs_id_dinov2"],
            "gal": emb["embs_swi_dinov2"][gallery_mask],
            "ood": emb["embs_ood_dinov2"],
            "gal_labels": gal_labels,
            "kind": "nn",
        },
        "ArcFace-557": {
            "id": emb["embs_id_arc"],
            "gal": emb["embs_swi_arc"][gallery_mask],
            "ood": emb["embs_ood_arc"],
            "gal_labels": gal_labels,
            "kind": "nn",
        },
        "CE-Full": {
            "id": emb["embs_id_ce_full_norm"],
            "gal": emb["embs_swi_ce_full_norm"][gallery_mask],
            "ood": emb["embs_ood_ce_full_norm"],
            "gal_labels": gal_labels,
            "kind": "nn",
        },
        "Fusion": {
            "id": fuse(emb["embs_id_arc"], emb["embs_id_dinov2"]),
            "gal": fuse(emb["embs_swi_arc"][gallery_mask], emb["embs_swi_dinov2"][gallery_mask]),
            "ood": fuse(emb["embs_ood_arc"], emb["embs_ood_dinov2"]),
            "gal_labels": gal_labels,
            "kind": "nn",
        },
    }
    if "embs_id_Var_CvNxt_SupCon" in emb.files:
        methods["SupCon"] = {
            "id": emb["embs_id_Var_CvNxt_SupCon"],
            "gal": emb["embs_swi_Var_CvNxt_SupCon"][gallery_mask],
            "ood": emb["embs_ood_Var_CvNxt_SupCon"],
            "gal_labels": gal_labels,
            "kind": "nn",
    }
    if SCURD_PROJ_CACHE.exists():
        sc = np.load(SCURD_PROJ_CACHE, allow_pickle=False)
        if uses_full_rows:
            # Get the SC-URD projection over the FULL 954-row pool, then apply the
            # same gallery_mask (all-ones for full_swi, CE-train mask for ce_train).
            if "swi_pool" in sc.files and len(sc["swi_pool"]) == len(labels_swi):
                full_pool = sc["swi_pool"]
            else:
                if not SCURD_MAIN_CKPT.exists():
                    raise KeyError(
                        f"{SCURD_PROJ_CACHE} does not contain an aligned 'swi_pool' and "
                        f"SC-URD checkpoint is missing: {SCURD_MAIN_CKPT}"
                    )
                try:
                    import torch
                    device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
                except Exception:
                    device = "cpu"
                print(f"SC-URD projected cache lacks aligned swi_pool; projecting full SWI from {SCURD_MAIN_CKPT.name}")
                model, _ = load_scurd_model(SCURD_MAIN_CKPT, in_dim=emb["embs_swi_dinov2"].shape[1], device=device)
                full_pool = project_np(model, emb["embs_swi_dinov2"], device)
            if len(full_pool) != len(labels_swi):
                raise ValueError(
                    f"SC-URD swi_pool length={len(full_pool)} but labels_swi length={len(labels_swi)}. "
                    "The projected cache and embedding cache are not aligned."
                )
            sc_gal = full_pool[gallery_mask]
            sc_gal_labels = labels_swi[gallery_mask]
        else:
            sc_gal = sc["gal"]
            sc_gal_labels = gal_labels
        methods["SC-URD"] = {
            "id": sc["id"],
            "gal": sc_gal,
            "ood": sc["ood"],
            "gal_labels": sc_gal_labels,
            "kind": "scurd",
        }
    elif SCURD_MAIN_CKPT.exists():
        try:
            import torch
            device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
        except Exception:
            device = "cpu"
        print(f"WARNING: missing SC-URD projected cache; projecting from checkpoint {SCURD_MAIN_CKPT.name}")
        model, _ = load_scurd_model(SCURD_MAIN_CKPT, in_dim=emb["embs_id_dinov2"].shape[1], device=device)
        methods["SC-URD"] = {
            "id": project_np(model, emb["embs_id_dinov2"], device),
            "gal": project_np(model, emb["embs_swi_dinov2"][gallery_mask], device),
            "ood": project_np(model, emb["embs_ood_dinov2"], device),
            "gal_labels": gal_labels,
            "kind": "scurd",
        }
    else:
        print(f"WARNING: missing SC-URD projected cache and checkpoint: {SCURD_PROJ_CACHE}, {SCURD_MAIN_CKPT}")

    return emb, methods, labels_id, labels_swi, labels_ood, gal_labels, gallery_mask


def run_map(methods, labels_id):
    rows = []
    for name, md in methods.items():
        centered = md.get("kind") == "scurd"
        metrics = retrieval_quality(md["id"], labels_id, md["gal"], md["gal_labels"], centered=centered)
        rows.append({"method": name, "sample_space": "centered" if centered else "raw", **metrics})
        print(f"mAP {name}: {metrics['mAP_query']:.4f}, MRR={metrics['MRR_query']:.4f}")
    save_csv(rows, "retrieval_map_mrr_recall.csv")
    return rows


def run_headline_recompute(methods, labels_id, labels_ood, emb=None):
    """Recompute headline retrieval/OOD/adaptation metrics under the selected gallery scope.

    Also emits the fair RQ1 columns so the corrected tab:rq1 is one artifact:
      - Prototype (anchor-matched: one centroid per gallery species vs CE's one
        weight per class) — recomputed at the active gallery scope.
      - Native CE (954-way softmax, gallery-INDEPENDENT) carried forward from the
        original rq1_paradigm.json; identical to the published value by construction.
    """
    ood_mask = ood_test_mask_from_labels(labels_ood)
    # Fair-comparison helpers: prefer the package version, fall back to inline.
    try:
        from swid_retrieval.experiments.rq1_native import (
            prototype_macro_top1 as _pkg_proto, carry_forward_native as _pkg_native)
        _proto_fn = lambda md_: _pkg_proto(md_["id"], labels_id, md_["gal"], md_["gal_labels"],
                                           centered=(md_.get("kind") == "scurd"))["mean"]
        _native_loader = _pkg_native
    except Exception:
        _proto_fn = lambda md_: _prototype_macro_top1(md_["id"], labels_id, md_["gal"], md_["gal_labels"],
                                                       centered=(md_.get("kind") == "scurd"))
        _native_loader = _carry_forward_native
    native_map = _native_loader(os.environ.get("SOURCE_RESULTS_DIR", str(RESULTS_DIR)))
    # Models whose gallery embeddings include species they trained on (memorized
    # under the full 954 gallery). ArcFace-557/DINOv2/SC-URD have clean galleries.
    memorized_models = {"CE-Full", "ArcFace-954"}
    rows = []
    rq1_rows = []
    rq2_ood = {}
    rq3_gallery = {}
    for name, md in methods.items():
        ev1 = eval_method(md, md["id"], labels_id, md["gal"], md["gal_labels"])
        e1a = ev1["mean"]
        proto = _proto_fn(md)
        native = native_map.get(name)
        # CE-Full/ArcFace-954 gallery embeddings are of species they trained on
        # (memorized) under full_swi; under ce_train the gallery is CE-Full's exact
        # training images, so the flag holds for any full-row scope.
        gallery_memorized = (name in memorized_models) and (
            GALLERY_SCOPE in {"full", "full_swi", "all_swi", "954", "954_swi",
                              "ce_train", "ce_train_only", "cetrain"})
        sid = distance_scores(md["id"], md["gal"])
        sod = distance_scores(md["ood"], md["gal"])[ood_mask] if md.get("ood") is not None else None
        e3ev = incremental_gallery_eval(md, labels_id, labels_ood, K=10) if md.get("ood") is not None else None
        e3old = None if e3ev is None else e3ev["old_id_mean"]
        e3new = None if e3ev is None else e3ev["new_50_mean"]
        r1_boot = bootstrap_values(ev1["per_species"].values())
        row = {
            "method": name,
            "gallery_scope": GALLERY_SCOPE,
            "gallery_images": int(len(md["gal_labels"])),
            "gallery_species": int(len(set(md["gal_labels"]))),
            "E1A_native_ce_macro": native,
            "E1A_prototype_macro": proto,
            "E1A_public_id_macro_R1": e1a,
            "E1A_ci_lo": r1_boot["ci_lo"],
            "E1A_ci_hi": r1_boot["ci_hi"],
            "gallery_memorized": gallery_memorized,
            "E3_old_after_adding_50_ood_K10": e3old,
            "E3_new_50_after_adding_K10": e3new,
        }
        if sod is not None:
            row["OOD_AUROC"] = compute_auroc(sid, sod)
            row["OOD_FPR95"] = fpr_at_tpr(sid, sod)
        else:
            row["OOD_AUROC"] = None
            row["OOD_FPR95"] = None
        rows.append(row)
        rq1_rows.append({
            "method": name,
            "native": native,
            "proto": proto,
            "r1": e1a,
            "r1_lo": r1_boot["ci_lo"],
            "r1_hi": r1_boot["ci_hi"],
            "r1_per_sp": ev1["per_species"],
            "gallery_scope": GALLERY_SCOPE,
            "gallery_species": int(len(set(md["gal_labels"]))),
            "gallery_images": int(len(md["gal_labels"])),
            "gallery_memorized": gallery_memorized,
        })
        if sod is not None:
            rq2_ood[name] = {
                "AUROC": {"mean": row["OOD_AUROC"]},
                "FPR95": {"mean": row["OOD_FPR95"]},
                "gallery_scope": GALLERY_SCOPE,
            }
        if e3old is not None:
            rq3_gallery[name] = {
                "old_id_mean": e3old,
                "new_50_mean": e3new,
                "old_per_species": e3ev["old_per_species"],
                "new_per_species": e3ev["new_per_species"],
                "n_new_gallery_images": e3ev["n_new_gallery_images"],
                "n_new_query_images": e3ev["n_new_query_images"],
                "gallery_scope": GALLERY_SCOPE,
            }
        print(
            f"headline {name}: native={native if native is not None else 'n/a'} "
            f"proto={proto:.4f} R@1={e1a:.4f} "
            f"OOD={row['OOD_AUROC'] if row['OOD_AUROC'] is not None else 'n/a'} "
            f"E3old={e3old if e3old is not None else 'n/a'} "
            f"E3new={e3new if e3new is not None else 'n/a'}"
        )
    save_csv(rows, "headline_recomputed_selected_gallery.csv")
    save_json_to_results({
        "gallery_scope": GALLERY_SCOPE,
        "note": (
            "Fair RQ1 at the active gallery scope. 'native' = 954-way CE softmax "
            "(gallery-independent; carried forward from rq1_paradigm.json). 'proto' = "
            "nearest class-centroid macro-top-1 (one centroid per gallery species), the "
            "anchor-matched counterpart of CE's one weight per class — the fair paradigm "
            "headline. 'r1' = nearest-image retrieval (deployment-realistic). "
            "gallery_memorized=true marks CE-Full/ArcFace-954, whose gallery embeddings "
            "include species they trained on; ArcFace-557/DINOv2/SC-URD have clean galleries."
        ),
        "paradigm": rq1_rows,
    }, "rq1_full_gallery.json")
    save_json_to_results({
        "gallery_scope": GALLERY_SCOPE,
        "ood": rq2_ood,
    }, "rq2_ood_full_gallery.json")
    save_json_to_results({
        "gallery_scope": GALLERY_SCOPE,
        "gallery_extension": rq3_gallery,
    }, "rq3_adaptation_full_gallery_partial.json")
    return rows


def build_public_id_splits(labels_id):
    id_csv = ROOT_PATH / "ID_images_expanded.csv"
    if not id_csv.exists():
        raise FileNotFoundError(id_csv)
    id_df = pd.read_csv(id_csv)
    id_df["canon"] = id_df["label"].apply(canonical_label)
    if len(id_df) != len(labels_id):
        raise ValueError(f"ID_images_expanded rows={len(id_df)} but labels_id={len(labels_id)}")
    path_to_idx = {str(id_df.iloc[i]["file_path"]): i for i in range(len(id_df))}
    rng = np.random.RandomState(42)
    splits = {}
    for sp in sorted(set(labels_id)):
        sp_df = id_df[id_df["canon"] == sp].reset_index(drop=True)
        idx = np.arange(len(sp_df))
        rng.shuffle(idx)
        n_query = min(10, len(sp_df) // 2)
        splits[sp] = {
            "query": sp_df.iloc[idx[:n_query]]["file_path"].astype(str).tolist(),
            "pool": sp_df.iloc[idx[n_query:]]["file_path"].astype(str).tolist(),
        }
    return splits, path_to_idx


def eval_method_macro(md, q_emb, q_labels, g_emb, g_labels):
    if md.get("kind") == "scurd":
        return scurd_macro_top1(q_emb, q_labels, g_emb, g_labels, mode=md.get("scurd_mode", SCURD_MODE))
    return nn_macro_top1(q_emb, q_labels, g_emb, g_labels)


def eval_method(md, q_emb, q_labels, g_emb, g_labels):
    if md.get("kind") == "scurd":
        return scurd_eval(q_emb, q_labels, g_emb, g_labels, mode=md.get("scurd_mode", SCURD_MODE))
    return nn_eval(q_emb, q_labels, g_emb, g_labels)


def run_gallery_resampling(methods, labels_id):
    splits, path_to_idx = build_public_id_splits(labels_id)
    id_species = sorted(splits)
    k_values = [1, 3, 5, 10, "all"]
    strategies = ["A_swi_only", "B_public_only", "C_mixed"]
    rows = []
    for name, md in methods.items():
        if name not in {"DINOv2", "Fusion", "SupCon", "SC-URD", "ArcFace-557", "CE-Full"}:
            continue
        for strategy in strategies:
            strategy_k_values = ["base"] if strategy == "A_swi_only" else k_values
            for K in strategy_k_values:
                vals = []
                reps = 1 if K in {"all", "base"} else int(N_GALLERY_REPEATS)
                for rep in range(reps):
                    rng = np.random.RandomState(SEED + rep)
                    q_parts, q_labels = [], []
                    g_parts, g_labels = [], []
                    for sp in id_species:
                        q_idx = [path_to_idx[p] for p in splits[sp]["query"]]
                        q_parts.append(md["id"][q_idx])
                        q_labels.extend([sp] * len(q_idx))
                        pool = splits[sp]["pool"]
                        if K == "base":
                            sample_paths = []
                        elif K == "all":
                            sample_paths = pool
                        else:
                            k_eff = min(int(K), len(pool))
                            sample_paths = [pool[i] for i in rng.choice(len(pool), k_eff, replace=False)]
                        if sample_paths:
                            s_idx = [path_to_idx[p] for p in sample_paths]
                            g_parts.append(md["id"][s_idx])
                            g_labels.extend([sp] * len(s_idx))
                    q_emb = np.concatenate(q_parts)
                    q_labels = np.asarray(q_labels)
                    if strategy == "A_swi_only":
                        g_emb = md["gal"]
                        gl = md["gal_labels"]
                    elif strategy == "B_public_only":
                        if not g_parts:
                            vals.append(0.0)
                            continue
                        g_emb = np.concatenate(g_parts)
                        gl = np.asarray(g_labels)
                    else:
                        if g_parts:
                            g_emb = np.concatenate([md["gal"], np.concatenate(g_parts)])
                            gl = np.concatenate([md["gal_labels"], np.asarray(g_labels)])
                        else:
                            g_emb = md["gal"]
                            gl = md["gal_labels"]
                    vals.append(eval_method_macro(md, q_emb, q_labels, g_emb, gl))
                vals = np.asarray(vals, dtype=float)
                rows.append({
                    "method": name,
                    "strategy": strategy,
                    "K": str(K),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "n_repeats": int(len(vals)),
                })
                print(f"gallery {name} {strategy} K={K}: {vals.mean():.4f} +/- {vals.std(ddof=1) if len(vals)>1 else 0.0:.4f}")
    save_csv(rows, "gallery_resampling_variance.csv")
    save_json_to_results({
        "gallery_scope": GALLERY_SCOPE,
        "strategies": rows,
        "note": "A_swi_only uses the selected SWI gallery scope; with GALLERY_SCOPE=full_swi this is the full SWI gallery. B_public_only intentionally uses only public references.",
    }, "rq1_gallery_strategies_full_gallery.json")
    return rows


def run_reviewer_gap_full_gallery(methods, labels_id, labels_ood):
    if "SC-URD" not in methods:
        print("WARNING: SC-URD missing; reviewer-gap full-gallery evidence omitted")
        return {}
    ood_mask = ood_test_mask_from_labels(labels_ood)
    sc = methods["SC-URD"]
    sc_ev = scurd_eval(sc["id"], labels_id, sc["gal"], sc["gal_labels"])
    correct = sc_ev["preds"] == labels_id
    evidence_sim = sc_ev["evidence_sim"]
    thresholds = sorted(set(
        [float(evidence_sim.min())]
        + [float(np.quantile(evidence_sim, q)) for q in [0.25, 0.50, 0.75, 0.90]]
    ))
    op_rows = []
    for th in thresholds:
        keep = evidence_sim >= th
        op_rows.append({
            "gallery_scope": GALLERY_SCOPE,
            "similarity_threshold": float(th),
            "auto_decided_coverage_pct": float(100.0 * keep.mean()),
            "top1_accuracy_auto_decided": None if keep.sum() == 0 else float(correct[keep].mean()),
            "flagged_for_review_pct": float(100.0 * (1.0 - keep.mean())),
            "n_auto_decided": int(keep.sum()),
            "n_total": int(len(keep)),
        })
    save_csv(op_rows, "deployment_operating_point_scurd_full_gallery.csv")

    score = {}
    for name, md in methods.items():
        if md.get("ood") is None:
            continue
        sid = distance_scores(md["id"], md["gal"])
        sod = distance_scores(md["ood"], md["gal"])[ood_mask]
        score[name] = {
            "id": sid,
            "ood": sod,
            "AUROC": compute_auroc(sid, sod),
            "FPR95": fpr_at_tpr(sid, sod),
        }
    fpr_rows = []
    for baseline in ["SupCon", "Fusion", "DINOv2", "ArcFace-557"]:
        if baseline not in score or "SC-URD" not in score:
            continue
        ci = bootstrap_fpr95_difference(
            score["SC-URD"]["id"], score["SC-URD"]["ood"],
            score[baseline]["id"], score[baseline]["ood"],
            n_boot=N_BOOT, seed=SEED,
        )
        fpr_rows.append({
            "gallery_scope": GALLERY_SCOPE,
            "comparison": f"SC-URD vs {baseline}",
            "scurd_fpr95": score["SC-URD"]["FPR95"],
            "baseline_fpr95": score[baseline]["FPR95"],
            **ci,
            "n_boot": int(N_BOOT),
        })
    save_csv(fpr_rows, "fpr95_difference_bootstrap_ci_full_gallery.csv")

    source_rows = []
    sources = ood_sources_from_csv(labels_ood)
    if sources is not None:
        sid = score["SC-URD"]["id"]
        sod_all = distance_scores(sc["ood"], sc["gal"])
        for source in sorted(set(sources)):
            mask = (sources == source) & ood_mask
            if mask.sum() < 2:
                continue
            try:
                auroc = compute_auroc(sid, sod_all[mask])
                fpr95 = fpr_at_tpr(sid, sod_all[mask])
            except Exception as exc:
                print(f"warning: per-source OOD failed for {source}: {exc}")
                auroc, fpr95 = np.nan, np.nan
            source_rows.append({
                "gallery_scope": GALLERY_SCOPE,
                "ood_source_dataset": source,
                "n_species": int(len(set(np.asarray(labels_ood)[mask]))),
                "n_ood_images": int(mask.sum()),
                "AUROC": float(auroc) if np.isfinite(auroc) else None,
                "FPR95": float(fpr95) if np.isfinite(fpr95) else None,
            })
        save_csv(source_rows, "ood_by_source_scurd_full_gallery.csv")

    score_summary = {
        name: {
            "AUROC": float(vals["AUROC"]),
            "FPR95": float(vals["FPR95"]),
            "n_id": int(len(vals["id"])),
            "n_ood_test": int(len(vals["ood"])),
        }
        for name, vals in score.items()
    }
    out = {
        "gallery_scope": GALLERY_SCOPE,
        "operating_point": op_rows,
        "fpr95_difference_ci": fpr_rows,
        "ood_by_source": source_rows,
        "ood_score_summary": score_summary,
    }
    save_json_to_results(out, "reviewer_gap_full_gallery.json")
    return out


def run_rq5_ablation_full_gallery(methods, labels_id, labels_ood):
    ood_mask = ood_test_mask_from_labels(labels_ood)
    backbone_rows = []
    for name, md in methods.items():
        if name not in {"ArcFace-557", "DINOv2", "Fusion", "SupCon", "CE-Full", "SC-URD"}:
            continue
        ev1 = eval_method(md, md["id"], labels_id, md["gal"], md["gal_labels"])
        sid = distance_scores(md["id"], md["gal"])
        sod = distance_scores(md["ood"], md["gal"])[ood_mask] if md.get("ood") is not None else None
        e3 = incremental_gallery_eval(md, labels_id, labels_ood, K=10) if md.get("ood") is not None else None
        backbone_rows.append({
            "method": name,
            "gallery_scope": GALLERY_SCOPE,
            "exp1a_mean": ev1["mean"],
            "ood_auroc": None if sod is None else compute_auroc(sid, sod),
            "ood_fpr95": None if sod is None else fpr_at_tpr(sid, sod),
            "exp3_old": None if e3 is None else e3["old_id_mean"],
            "exp3_new": None if e3 is None else e3["new_50_mean"],
            "gallery_species": int(len(set(md["gal_labels"]))),
            "gallery_images": int(len(md["gal_labels"])),
        })
    save_csv(backbone_rows, "rq5_backbone_loss_full_gallery.csv")

    memory_rows = []
    if "SC-URD" in methods:
        base = methods["SC-URD"]
        for mode in ["raw", "centered", "logmeanexp", "prototype_mix", "centered_logmeanexp", "centered_prototype_mix"]:
            md = dict(base)
            md["scurd_mode"] = mode
            ev1 = eval_method(md, md["id"], labels_id, md["gal"], md["gal_labels"])
            sid = distance_scores(md["id"], md["gal"])
            sod = distance_scores(md["ood"], md["gal"])[ood_mask]
            e3 = incremental_gallery_eval(md, labels_id, labels_ood, K=10)
            memory_rows.append({
                "method": "SC-URD",
                "variant": "scurd_r01_e20",
                "memory_mode": mode,
                "gallery_scope": GALLERY_SCOPE,
                "exp1a_mean": ev1["mean"],
                "ood_auroc": compute_auroc(sid, sod),
                "ood_fpr95": fpr_at_tpr(sid, sod),
                "exp3_old": e3["old_id_mean"],
                "exp3_new": e3["new_50_mean"],
            })
    save_csv(memory_rows, "rq5_scurd_memory_modes_full_gallery.csv")
    out = {
        "gallery_scope": GALLERY_SCOPE,
        "backbone_loss_ablation": backbone_rows,
        "scurd_memory_ablation": memory_rows,
        "note": "This recomputes E1/OOD/E3 axes under the selected gallery scope. VN26 axes are not recomputed here because they already use SWI_pool/VN26-specific galleries.",
    }
    save_json_to_results(out, "rq5_ablation_full_gallery_partial.json")
    return out


def ood_test_mask_from_labels(labels_ood):
    species = sorted(set(labels_ood))
    rng = np.random.RandomState(42)
    rng.shuffle(species)
    n_val = max(1, int(0.2 * len(species)))
    test_species = set(species[n_val:])
    return np.asarray([s in test_species for s in labels_ood])


def ood_species_splits(labels_ood):
    ood_csv = ROOT_PATH / "OOD_images_expanded.csv"
    if not ood_csv.exists():
        raise FileNotFoundError(ood_csv)
    df = pd.read_csv(ood_csv)
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
    return splits


def source_from_path(path):
    p = str(path).lower()
    known = [
        ("bd11", "BD11"),
        ("bfs46", "BFS46"),
        ("dtsr14", "DTSR14"),
        ("fsdm41", "FSDM41"),
        ("goimai", "GOIMAI"),
        ("pca11", "PCA11"),
        ("vn26", "VN26"),
        ("wood-auth", "WOOD-AUTH"),
        ("wood_auth", "WOOD-AUTH"),
        ("wrd25", "WRD25"),
    ]
    for needle, label in known:
        if needle in p:
            return label
    parts = Path(path).parts
    for part in parts:
        low = part.lower()
        for needle, label in known:
            if needle in low:
                return label
    return "unknown"


def ood_sources_from_csv(labels_ood):
    ood_csv = ROOT_PATH / "OOD_images_expanded.csv"
    if not ood_csv.exists():
        print(f"WARNING: missing {ood_csv}; per-source OOD omitted")
        return None
    df = pd.read_csv(ood_csv)
    if len(df) != len(labels_ood):
        raise ValueError(f"OOD CSV length={len(df)} but labels_ood length={len(labels_ood)}")
    return np.asarray([source_from_path(p) for p in df["file_path"].astype(str)])


def incremental_gallery_eval(md, labels_id, labels_ood, K=10):
    splits = ood_species_splits(labels_ood)
    pool_idx = []
    query_idx = []
    query_labels = []
    for sp, s in splits.items():
        pool_idx.extend(s["pool_indices"][: min(K, len(s["pool_indices"]))])
        query_idx.extend(s["query_indices"])
        query_labels.extend([sp] * len(s["query_indices"]))
    if not pool_idx:
        return None
    g_emb = np.concatenate([md["gal"], md["ood"][pool_idx]])
    g_labels = np.concatenate([md["gal_labels"], np.asarray(labels_ood)[pool_idx]])
    old_ev = eval_method(md, md["id"], labels_id, g_emb, g_labels)
    if query_idx:
        new_ev = eval_method(md, md["ood"][query_idx], np.asarray(query_labels), g_emb, g_labels)
        new_mean = new_ev["mean"]
        new_per = new_ev["per_species"]
    else:
        new_mean = None
        new_per = {}
    return {
        "old_id_mean": old_ev["mean"],
        "old_per_species": old_ev["per_species"],
        "new_50_mean": new_mean,
        "new_per_species": new_per,
        "n_new_gallery_images": int(len(pool_idx)),
        "n_new_query_images": int(len(query_idx)),
    }


def incremental_old_retention(md, labels_id, labels_ood, K=10):
    ev = incremental_gallery_eval(md, labels_id, labels_ood, K=K)
    return None if ev is None else ev["old_id_mean"]


def load_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def project_np(model, arr, device, batch_size=8192):
    torch, _, _ = load_torch()
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(arr), batch_size):
            x = torch.tensor(arr[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs, axis=0)


def build_scurd_model_class():
    torch, nn, F = load_torch()

    class SCURDResidualHead(nn.Module):
        def __init__(self, in_dim, out_dim=512, beta=0.1, learnable_beta=False):
            super().__init__()
            self.base = nn.Linear(in_dim, out_dim, bias=False)
            self.adapter = nn.Sequential(
                nn.Linear(in_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
                nn.Linear(1024, out_dim),
            )
            if learnable_beta:
                beta = float(beta if isinstance(beta, (int, float)) else 0.1)
                beta = min(max(beta, 1e-4), 0.999)
                self.logit_beta = nn.Parameter(torch.tensor(np.log(beta / (1.0 - beta)), dtype=torch.float32))
            else:
                self.register_buffer("fixed_beta", torch.tensor(float(beta), dtype=torch.float32))
                self.logit_beta = None

        def beta_value(self):
            return torch.sigmoid(self.logit_beta) if self.logit_beta is not None else self.fixed_beta

        def forward(self, x):
            base = F.normalize(self.base(x), dim=1)
            delta = F.normalize(self.adapter(x), dim=1)
            beta = self.beta_value().to(x.device)
            return F.normalize(base + beta * delta, dim=1)

    return SCURDResidualHead


def load_scurd_model(ckpt_path, in_dim=768, device="cuda"):
    torch, _, _ = load_torch()
    SCURDResidualHead = build_scurd_model_class()
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = int(ckpt.get("in_dim", in_dim))
    out_dim = int(ckpt.get("out_dim", 512))
    beta = ckpt.get("beta", 0.1)
    learnable = bool(ckpt.get("learnable_beta", False))
    state = ckpt["model_state_dict"]
    model = SCURDResidualHead(in_dim, out_dim=out_dim, beta=0.1 if learnable else float(beta), learnable_beta=learnable).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def scurd_seed_suffix(seed):
    beta_tag = str(SCURD_BETA).replace(".", "")
    suffix = f"scurd_r{beta_tag}_e{SCURD_TRAIN_EPOCHS}_seed{int(seed)}"
    # Preserve the original default checkpoint names for lambda_cons=0.5, but
    # avoid collisions for consistency-loss ablations such as lambda_cons=0.
    if abs(float(SCURD_TRAIN_LAMBDA_CONS) - 0.5) > 1e-12:
        cons_tag = str(SCURD_TRAIN_LAMBDA_CONS).replace(".", "")
        suffix += f"_lc{cons_tag}"
    return suffix


def scurd_seed_ckpt_path(seed):
    return SCURD_CKPT_DIR / f"sc_urd_checkpoint_{scurd_seed_suffix(seed)}_{SCURD_CACHE_VERSION}.pt"


def scurd_seed_log_path(seed):
    return SCURD_CKPT_DIR / f"sc_urd_train_log_{scurd_seed_suffix(seed)}_{SCURD_CACHE_VERSION}.json"


def load_scurd_meta_cache():
    if not SCURD_META_CACHE.exists():
        raise FileNotFoundError(
            f"Missing SC-URD meta embedding cache: {SCURD_META_CACHE}. "
            "Run the original SC-URD/URD embedding-cache extraction once first."
        )
    c = np.load(SCURD_META_CACHE, allow_pickle=False)
    required = {"train_weak", "train_strong", "train_labels"}
    missing = required - set(c.files)
    if missing:
        raise KeyError(f"{SCURD_META_CACHE} missing keys: {sorted(missing)}")
    return {
        "train_weak": c["train_weak"].astype(np.float32),
        "train_strong": c["train_strong"].astype(np.float32),
        "train_labels": np.asarray([canonical_label(x) for x in c["train_labels"]]),
    }


def scurd_episode_indices(labels, n_way, k_support, q_query, rng):
    labels = np.asarray(labels)
    by_sp = {sp: np.where(labels == sp)[0] for sp in np.unique(labels)}
    valid = [sp for sp, idx in by_sp.items() if len(idx) >= k_support + q_query]
    if len(valid) < n_way:
        valid = [sp for sp, idx in by_sp.items() if len(idx) >= 2]
    if not valid:
        raise RuntimeError("SC-URD seed training needs at least one species with >=2 cached embeddings.")
    chosen = rng.choice(valid, size=min(int(n_way), len(valid)), replace=False)
    support, query, support_targets, query_targets = [], [], [], []
    for ci, sp in enumerate(chosen):
        idx = by_sp[sp].copy()
        rng.shuffle(idx)
        n_s = min(int(k_support), max(1, len(idx) // 2))
        n_q = min(int(q_query), max(1, len(idx) - n_s))
        support.extend(idx[:n_s].tolist())
        query.extend(idx[n_s:n_s + n_q].tolist())
        support_targets.extend([ci] * n_s)
        query_targets.extend([ci] * n_q)
    return (
        np.asarray(support, dtype=np.int64),
        np.asarray(query, dtype=np.int64),
        np.asarray(support_targets, dtype=np.int64),
        np.asarray(query_targets, dtype=np.int64),
    )


def urd_logits(query_z, support_z, support_targets, n_way, tau):
    torch, _, _ = load_torch()
    sim = query_z @ support_z.T / max(float(tau), 1e-6)
    logits = []
    support_targets = support_targets.to(sim.device)
    for c in range(int(n_way)):
        mask = support_targets == c
        if not bool(mask.any()):
            logits.append(torch.full((sim.shape[0],), -1e9, dtype=sim.dtype, device=sim.device))
        else:
            logits.append(torch.logsumexp(sim[:, mask], dim=1))
    return torch.stack(logits, dim=1)


def train_one_scurd_seed(seed, device):
    torch, _, F = load_torch()
    ckpt_path = scurd_seed_ckpt_path(seed)
    log_path = scurd_seed_log_path(seed)
    if ckpt_path.exists() and not SCURD_FORCE_RETRAIN_SEEDS:
        print(f"  loaded existing SC-URD seed checkpoint: {ckpt_path.name}")
        return ckpt_path

    print(f"  training SC-URD seed={seed} -> {ckpt_path.name}")
    set_all_seeds(seed)
    meta = load_scurd_meta_cache()
    train_w = torch.tensor(meta["train_weak"], dtype=torch.float32)
    train_s = torch.tensor(meta["train_strong"], dtype=torch.float32)
    labels = meta["train_labels"]
    in_dim = int(train_w.shape[1])

    SCURDResidualHead = build_scurd_model_class()
    model = SCURDResidualHead(in_dim, out_dim=512, beta=SCURD_BETA, learnable_beta=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=SCURD_TRAIN_LR, weight_decay=SCURD_WEIGHT_DECAY)
    rng = np.random.RandomState(int(seed))
    best_loss, best_state = float("inf"), None
    train_log = []

    for ep in range(1, SCURD_TRAIN_EPOCHS + 1):
        model.train()
        losses, cls_losses, cons_losses, accs = [], [], [], []
        for _ in range(SCURD_TRAIN_EPISODES):
            sup_idx, qry_idx, ys_np, yq_np = scurd_episode_indices(
                labels, SCURD_N_WAY, SCURD_K_SUPPORT, SCURD_Q_QUERY, rng
            )
            ys = torch.tensor(ys_np, dtype=torch.long, device=device)
            yq = torch.tensor(yq_np, dtype=torch.long, device=device)
            support_z = model(train_w[sup_idx].to(device))
            qw = model(train_w[qry_idx].to(device))
            qs = model(train_s[qry_idx].to(device))
            n_way_eff = int(yq.max().item() + 1)
            logits_w = urd_logits(qw, support_z, ys, n_way_eff, SCURD_TAU)
            logits_s = urd_logits(qs, support_z, ys, n_way_eff, SCURD_TAU)
            loss_cls = F.cross_entropy(logits_w, yq)
            pw = F.log_softmax(logits_w, dim=1)
            ps = F.log_softmax(logits_s, dim=1)
            loss_cons = 0.5 * (
                F.kl_div(pw, ps.exp(), reduction="batchmean")
                + F.kl_div(ps, pw.exp(), reduction="batchmean")
            )
            loss = loss_cls + SCURD_TRAIN_LAMBDA_CONS * loss_cons
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc = float((logits_w.argmax(1) == yq).float().mean().item())
            losses.append(float(loss.item()))
            cls_losses.append(float(loss_cls.item()))
            cons_losses.append(float(loss_cons.item()))
            accs.append(acc)
        rec = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "cls_loss": float(np.mean(cls_losses)),
            "id_cons_loss": float(np.mean(cons_losses)),
            "episode_acc": float(np.mean(accs)),
            "seed": int(seed),
            "beta": float(SCURD_BETA),
        }
        train_log.append(rec)
        print(
            f"    seed={seed} ep {ep:02d}/{SCURD_TRAIN_EPOCHS}: "
            f"loss={rec['loss']:.4f} cls={rec['cls_loss']:.4f} "
            f"cons={rec['id_cons_loss']:.4f} acc={rec['episode_acc']:.3f}"
        )
        if rec["loss"] < best_loss:
            best_loss = rec["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "out_dim": 512,
        "training_complete": True,
        "method": "sc_urd",
        "head_type": "residual",
        "epochs": int(SCURD_TRAIN_EPOCHS),
        "episodes_per_epoch": int(SCURD_TRAIN_EPISODES),
        "n_way": int(SCURD_N_WAY),
        "k_support": int(SCURD_K_SUPPORT),
        "q_query": int(SCURD_Q_QUERY),
        "lambda_cons": float(SCURD_TRAIN_LAMBDA_CONS),
        "lr": float(SCURD_TRAIN_LR),
        "weight_decay": float(SCURD_WEIGHT_DECAY),
        "tau": float(SCURD_TAU),
        "seed": int(seed),
        "beta": float(SCURD_BETA),
        "learnable_beta": False,
        "selection_metric": "lowest_training_loss",
        "best_training_loss": float(best_loss),
        "cache_version": SCURD_CACHE_VERSION,
        "research_cache_version": RESEARCH_CACHE_VERSION,
        "meta_cache": str(SCURD_META_CACHE),
    }
    torch.save(ckpt, ckpt_path)
    save_json({"config": {k: v for k, v in ckpt.items() if k not in {"model_state_dict"}}, "train_log": train_log},
              log_path.name)
    # Keep train logs beside checkpoints as well as in OUT_DIR.
    with open(log_path, "w") as f:
        json.dump({"config": {k: v for k, v in ckpt.items() if k not in {"model_state_dict"}}, "train_log": train_log},
                  f, indent=2, sort_keys=True)
    return ckpt_path


def ensure_scurd_seed_checkpoints(device):
    if not RUN_TRAIN_SCURD_SEEDS:
        return sorted(SCURD_CKPT_DIR.glob(SCURD_SEED_CKPT_GLOB))
    paths = []
    for seed in SCURD_TRAIN_SEEDS:
        paths.append(train_one_scurd_seed(seed, device))
    return paths


def exp4_key(prefix, method, split):
    return f"{prefix}__{method.replace('-', '_').replace('/', '_')}__{str(split)}"


def eval_e4a_scurd(model, device):
    if not EXP4_CACHE_PATH.exists():
        return None
    c4 = np.load(EXP4_CACHE_PATH, allow_pickle=False)
    scales = [256, 512, 768]
    mags = ["x10", "x20", "x50"]
    swi_parts, swi_labels = [], []
    vn_parts, vn_labels = [], []
    for s in scales:
        k = exp4_key("swi", "DINOv2", s)
        if k not in c4.files or f"{k}_lbl" not in c4.files:
            return None
        swi_parts.append(project_np(model, c4[k], device))
        swi_labels.append(np.asarray([canonical_label(x) for x in c4[f"{k}_lbl"]]))
    for m in mags:
        k = exp4_key("vn26", "DINOv2", m)
        if k not in c4.files or f"{k}_lbl" not in c4.files:
            return None
        vn_parts.append(project_np(model, c4[k], device))
        vn_labels.append(np.asarray([canonical_label(x) for x in c4[f"{k}_lbl"]]))
    swi_e = np.concatenate(swi_parts)
    swi_l = np.concatenate(swi_labels)
    vn_e = np.concatenate(vn_parts)
    vn_l = np.concatenate(vn_labels)
    common = sorted(set(swi_l) & set(vn_l))
    if not common:
        return None
    gm = np.asarray([x in common for x in swi_l])
    qm = np.asarray([x in common for x in vn_l])
    return scurd_macro_top1(vn_e[qm], vn_l[qm], swi_e[gm], swi_l[gm])


def run_scurd_seed_sensitivity(emb, labels_id, labels_ood, gal_labels, gallery_mask):
    try:
        import torch
    except Exception as exc:
        print(f"Skipping seed sensitivity: torch import failed: {exc}")
        return []
    device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    ckpts = ensure_scurd_seed_checkpoints(device)
    if not ckpts:
        ckpts = sorted(SCURD_CKPT_DIR.glob(SCURD_SEED_CKPT_GLOB))
    if not ckpts:
        print(f"No SC-URD checkpoints found with glob: {SCURD_CKPT_DIR / SCURD_SEED_CKPT_GLOB}")
        save_csv([], "scurd_seed_sensitivity.csv")
        save_csv([], "scurd_seed_sensitivity_summary.csv")
        return []
    ood_mask = ood_test_mask_from_labels(labels_ood)
    rows = []
    for ckpt_path in ckpts:
        print(f"Evaluating seed checkpoint: {ckpt_path.name}")
        model, ckpt = load_scurd_model(ckpt_path, in_dim=emb["embs_id_dinov2"].shape[1], device=device)
        md = {
            "id": project_np(model, emb["embs_id_dinov2"], device),
            "gal": project_np(model, emb["embs_swi_dinov2"][gallery_mask], device),
            "ood": project_np(model, emb["embs_ood_dinov2"], device),
            "gal_labels": gal_labels,
            "kind": "scurd",
        }
        e1a = scurd_macro_top1(md["id"], labels_id, md["gal"], md["gal_labels"])
        sid = distance_scores(md["id"], md["gal"])
        sod = distance_scores(md["ood"], md["gal"])[ood_mask]
        e3old = incremental_old_retention(md, labels_id, labels_ood, K=10)
        e4a = eval_e4a_scurd(model, device)
        rows.append({
            "checkpoint": ckpt_path.name,
            "seed": int(ckpt.get("seed", -1)),
            "epochs": int(ckpt.get("epochs", -1)),
            "episodes_per_epoch": int(ckpt.get("episodes_per_epoch", -1)),
            "beta": float(ckpt.get("beta", 0.1)) if isinstance(ckpt.get("beta", 0.1), (int, float)) else np.nan,
            "memory_mode": SCURD_MODE,
            "gallery_scope": GALLERY_SCOPE,
            "gallery_images": int(len(gal_labels)),
            "gallery_species": int(len(set(gal_labels))),
            "E1A_public_id_R1": e1a,
            "OOD_AUROC": compute_auroc(sid, sod),
            "OOD_FPR95": fpr_at_tpr(sid, sod),
            "E3_old_after_K10": e3old,
            "E4A_SWI_pool_to_VN26_all": e4a,
        })
    save_csv(rows, "scurd_seed_sensitivity.csv")
    metrics = ["E1A_public_id_R1", "OOD_AUROC", "OOD_FPR95", "E3_old_after_K10", "E4A_SWI_pool_to_VN26_all"]
    summary = []
    df = pd.DataFrame(rows)
    for m in metrics:
        vals = pd.to_numeric(df[m], errors="coerce").dropna().values
        summary.append({
            "metric": m,
            "mean": float(vals.mean()) if len(vals) else None,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else None,
            "min": float(vals.min()) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
            "n_checkpoints": int(len(vals)),
        })
    save_csv(summary, "scurd_seed_sensitivity_summary.csv")
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"ROOT_PATH={ROOT_PATH}")
    print(f"RESULTS_DIR={RESULTS_DIR}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"EMB_CACHE_PATH={EMB_CACHE_PATH}")
    print(f"SCURD_META_CACHE={SCURD_META_CACHE}")
    print(f"GALLERY_SCOPE={GALLERY_SCOPE}")
    print(f"SCURD train seeds={SCURD_TRAIN_SEEDS} epochs={SCURD_TRAIN_EPOCHS} episodes={SCURD_TRAIN_EPISODES}")
    emb, methods, labels_id, labels_swi, labels_ood, gal_labels, gallery_mask = load_embedding_artifacts()
    audit = protocol_audit_rows(methods, labels_id, labels_ood)
    save_csv(audit, "protocol_audit.csv")
    if GALLERY_SCOPE == "full_swi":
        bad = [
            r for r in audit
            if r["experiment_group"] == "main_gallery" and not r["is_full_swi_gallery"]
        ]
        if bad:
            raise RuntimeError(f"Protocol audit failed for full_swi gallery: {bad[:3]}")

    manifest = {
        "root_path": str(ROOT_PATH),
        "results_dir": str(RESULTS_DIR),
        "out_dir": str(OUT_DIR),
        "embedding_cache": str(EMB_CACHE_PATH),
        "exp4_cache": str(EXP4_CACHE_PATH),
        "scurd_projected_cache": str(SCURD_PROJ_CACHE),
        "scurd_meta_cache": str(SCURD_META_CACHE),
        "scurd_ckpt_glob": str(SCURD_CKPT_DIR / SCURD_SEED_CKPT_GLOB),
        "gallery_scope": GALLERY_SCOPE,
        "gallery_images": int(len(gal_labels)),
        "gallery_species": int(len(set(gal_labels))),
        "min_full_gallery_species": int(MIN_FULL_GALLERY_SPECIES),
        "protocol_audit": "protocol_audit.csv",
        "run_map": RUN_MAP,
        "run_headline_recompute": RUN_HEADLINE_RECOMPUTE,
        "run_gallery_resampling": RUN_GALLERY_RESAMPLING,
        "run_scurd_seed_sensitivity": RUN_SCURD_SEED_SENSITIVITY,
        "run_train_scurd_seeds": RUN_TRAIN_SCURD_SEEDS,
        "run_reviewer_gap_full_gallery": RUN_REVIEWER_GAP_FULL_GALLERY,
        "run_rq5_full_gallery": RUN_RQ5_FULL_GALLERY,
        "n_gallery_repeats": N_GALLERY_REPEATS,
        "scurd_seed_config": {
            "seeds": SCURD_TRAIN_SEEDS,
            "beta": SCURD_BETA,
            "mode": SCURD_MODE,
            "tau": SCURD_TAU,
            "top_m": SCURD_TOP_M,
            "epochs": SCURD_TRAIN_EPOCHS,
            "episodes_per_epoch": SCURD_TRAIN_EPISODES,
            "lambda_cons": SCURD_TRAIN_LAMBDA_CONS,
            "lr": SCURD_TRAIN_LR,
            "weight_decay": SCURD_WEIGHT_DECAY,
            "n_way": SCURD_N_WAY,
            "k_support": SCURD_K_SUPPORT,
            "q_query": SCURD_Q_QUERY,
            "force_retrain": SCURD_FORCE_RETRAIN_SEEDS,
        },
        "outputs": {},
    }
    if RUN_MAP:
        run_map(methods, labels_id)
        manifest["outputs"]["map"] = "retrieval_map_mrr_recall.csv"
    if RUN_HEADLINE_RECOMPUTE:
        run_headline_recompute(methods, labels_id, labels_ood, emb=emb)
        manifest["outputs"]["headline_recompute"] = "headline_recomputed_selected_gallery.csv"
    if RUN_GALLERY_RESAMPLING:
        run_gallery_resampling(methods, labels_id)
        manifest["outputs"]["gallery_resampling"] = "gallery_resampling_variance.csv"
    if RUN_REVIEWER_GAP_FULL_GALLERY:
        run_reviewer_gap_full_gallery(methods, labels_id, labels_ood)
        manifest["outputs"]["reviewer_gap_full_gallery"] = "reviewer_gap_full_gallery.json"
    if RUN_RQ5_FULL_GALLERY:
        run_rq5_ablation_full_gallery(methods, labels_id, labels_ood)
        manifest["outputs"]["rq5_ablation_full_gallery"] = "rq5_ablation_full_gallery_partial.json"
    if RUN_SCURD_SEED_SENSITIVITY:
        run_scurd_seed_sensitivity(emb, labels_id, labels_ood, gal_labels, gallery_mask)
        manifest["outputs"]["scurd_seed_sensitivity"] = "scurd_seed_sensitivity.csv"
        manifest["outputs"]["scurd_seed_sensitivity_summary"] = "scurd_seed_sensitivity_summary.csv"
    save_json(manifest, "variance_evidence_manifest.json")
    print("Done.")


if __name__ == "__main__":
    main()

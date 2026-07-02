# -*- coding: utf-8 -*-
"""
review_evidence_colab.py

Post-hoc evidence generator for reviewer-facing cleanup:
  1) Statistical cleanup tables for main claims.
  2) Simplified-method evidence summary.
  3) Per-species and failure taxonomy for retrieval.
  4) Optional retrieval-style anatomical plausibility figures via occlusion.

This script is intentionally standalone. It does NOT import final_research.py
and does NOT rerun RQ1-RQ5 experiments. It reads existing JSON/NPZ artifacts
from Drive and writes additional review evidence under:

    /content/drive/MyDrive/NCS/results/paper_reframe/review_evidence

Recommended Colab usage:

    !python review_evidence_colab.py

Optional qualitative interpretability figures, slower:

    !RUN_INTERPRETABILITY=1 N_INTERP_CASES=6 python review_evidence_colab.py
"""

import ast
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_PATH = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT_PATH / "results/paper_reframe"))
# Where to READ existing RQ JSONs (rq2/rq3/rq4). Defaults to RESULTS_DIR, but the
# full-954 orchestrator points this at the original paper_reframe (which holds
# them) while RESULTS_DIR/OUT_DIR stay in the isolated run dir.
SOURCE_RESULTS_DIR = Path(os.environ.get("SOURCE_RESULTS_DIR", RESULTS_DIR))
OUT_DIR = RESULTS_DIR / "review_evidence"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMB_CACHE_PATH = ROOT_PATH / os.environ.get("EMB_CACHE_NAME", "embedding_cache_v3.npz")
SCURD_PROJ_CACHE = (
    RESULTS_DIR
    / "research_directions"
    / os.environ.get("SCURD_PROJ_CACHE_NAME", "sc_urd_eval_embeddings_scurd_r01_e20_recomputed_v2.npz")
)

SCURD_MODE = os.environ.get("SCURD_MAIN_MODE", "centered").strip() or "centered"
RNG_SEED = int(os.environ.get("REVIEW_EVIDENCE_SEED", "42"))

# Gallery scope migration: default keeps the legacy standalone behaviour
# (id_only, 24 species); the swid_retrieval orchestrator sets full_swi so this
# evidence is recomputed against the corrected 954-species gallery.
GALLERY_SCOPE = os.environ.get("GALLERY_SCOPE", "id_only").strip().lower()
MIN_FULL_GALLERY_SPECIES = int(os.environ.get("MIN_FULL_GALLERY_SPECIES", "900"))
_FULL_SCOPES = {"full", "full_swi", "all_swi", "954", "954_swi"}
# Guard: review taxonomy supports full_swi / id_only only. ce_train robustness is
# produced by the variance engine, not here — fail loudly instead of silently
# falling back to id_only (which would give wrong per-species/confusion rows).
if GALLERY_SCOPE in {"ce_train", "ce_train_only", "cetrain"}:
    raise SystemExit(
        "review_evidence: GALLERY_SCOPE=ce_train is not supported here. Run review "
        "taxonomy under full_swi; the ce_train fairness check uses the variance engine."
    )
IS_FULL_GALLERY = GALLERY_SCOPE in _FULL_SCOPES
SCURD_MAIN_CKPT = (
    RESULTS_DIR / "research_directions"
    / os.environ.get("SCURD_MAIN_CKPT_NAME", "sc_urd_checkpoint_scurd_r01_e20_v2.pt")
)
DEVICE = os.environ.get("DEVICE", "cuda")


def canonical_label(s):
    return str(s).replace(" ", "_").lower().strip()


def load_json(name):
    path = SOURCE_RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, name):
    path = OUT_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"saved {path}")


def save_csv(rows, name):
    path = OUT_DIR / name
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")
    return df


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fuse(a, b):
    return norm(np.concatenate([norm(a), norm(b)], axis=1))


def bootstrap_mean_ci(values, n_boot=5000, seed=RNG_SEED):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    rng = np.random.RandomState(seed)
    boots = [values[rng.choice(len(values), len(values), replace=True)].mean() for _ in range(n_boot)]
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n": int(len(values)),
    }


def paired_wilcoxon(a_by_sp, b_by_sp):
    common = sorted(set(a_by_sp) & set(b_by_sp))
    if len(common) < 2:
        return {"p_value": None, "statistic": None, "n": len(common), "delta": None}
    a = np.asarray([a_by_sp[s] for s in common], dtype=float)
    b = np.asarray([b_by_sp[s] for s in common], dtype=float)
    out = {"n": int(len(common)), "delta": float(a.mean() - b.mean())}
    if np.allclose(a, b):
        out.update({"statistic": None, "p_value": 1.0})
        return out
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(a, b, zero_method="wilcox")
        out.update({"statistic": float(stat), "p_value": float(p)})
    except Exception as exc:
        out.update({"statistic": None, "p_value": None, "error": str(exc)})
    return out


def add_holm_adjusted_p(rows, group_key="claim", p_key="p_value", out_key="p_holm"):
    """Add Holm-Bonferroni adjusted p-values within each claim family.

    Rows without a finite p-value keep ``None``. Adjustment is performed
    independently per ``group_key`` so RQ1/RQ3/RQ4 claim families are not mixed.
    """
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        p = row.get(p_key)
        if p is None:
            row[out_key] = None
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            row[out_key] = None
            continue
        if not np.isfinite(p):
            row[out_key] = None
            continue
        row[p_key] = p
        groups[row.get(group_key, "")].append((i, p))

    for items in groups.values():
        if not items:
            continue
        ordered = sorted(items, key=lambda x: x[1])
        m = len(ordered)
        adjusted_sorted = []
        running_max = 0.0
        for rank, (idx, p) in enumerate(ordered):
            adj = min(1.0, (m - rank) * p)
            running_max = max(running_max, adj)
            adjusted_sorted.append((idx, running_max))
        for idx, adj in adjusted_sorted:
            rows[idx][out_key] = float(adj)
    return rows


def macro_per_species_from_preds(preds, labels):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_sp = {}
    for sp in sorted(np.unique(labels)):
        m = labels == sp
        per_sp[sp] = float((preds[m] == labels[m]).mean())
    return float(np.mean(list(per_sp.values()))), per_sp


def nearest_neighbor_eval(q_emb, q_labels, g_emb, g_labels):
    q = norm(q_emb)
    g = norm(g_emb)
    sims = q @ g.T
    top_idx = sims.argmax(axis=1)
    preds = np.asarray(g_labels)[top_idx]
    macro, per_sp = macro_per_species_from_preds(preds, q_labels)
    return {
        "macro": macro,
        "per_species": per_sp,
        "preds": preds,
        "top_idx": top_idx,
        "top_sim": sims[np.arange(len(q)), top_idx],
    }


def scurd_scores(query_embs, gallery_embs, gallery_labels, mode="centered", tau=0.07, top_m=50):
    """SC-URD class scorer compatible with the final_research centered mode."""
    q = np.asarray(query_embs, dtype=np.float32)
    g = np.asarray(gallery_embs, dtype=np.float32)
    gl = np.asarray(gallery_labels)
    if mode in {"centered", "centered_logmeanexp", "centered_prototype_mix"}:
        mu = g.mean(axis=0, keepdims=True)
        q = q - mu
        g = g - mu
    q = norm(q)
    g = norm(g)

    class_order = np.asarray(sorted(np.unique(gl)))
    class_to_col = {c: i for i, c in enumerate(class_order)}
    sims = q @ g.T
    top_m_eff = min(int(top_m), sims.shape[1])
    top_idx = np.argpartition(-sims, top_m_eff - 1, axis=1)[:, :top_m_eff]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)

    logits = np.full((len(q), len(class_order)), -1e9, dtype=np.float64)
    tau = max(float(tau), 1e-6)
    balance = mode in {"logmeanexp", "centered_logmeanexp"}
    proto_mix = mode in {"prototype_mix", "centered_prototype_mix"}
    for i in range(len(q)):
        labs = gl[top_idx[i]]
        for c in np.unique(labs):
            col = class_to_col[c]
            vals = top_sims[i, labs == c] / tau
            m = vals.max()
            score = m + np.log(np.exp(vals - m).sum())
            if balance:
                score -= np.log(max(1, np.sum(gl == c)))
            logits[i, col] = score

    if proto_mix:
        proto = []
        for c in class_order:
            proto.append(g[gl == c].mean(axis=0))
        proto = norm(np.asarray(proto, dtype=np.float32))
        proto_scores = (q @ proto.T) / tau
        sample_scores = np.where(logits < -1e8, proto_scores, logits)
        logits = 0.5 * sample_scores + 0.5 * proto_scores

    preds = class_order[logits.argmax(axis=1)]
    # Evidence item: nearest gallery sample within predicted class.
    evidence_idx = []
    for i, pred in enumerate(preds):
        candidates = np.where(gl == pred)[0]
        if len(candidates) == 0:
            evidence_idx.append(int(np.argmax(sims[i])))
        else:
            evidence_idx.append(int(candidates[np.argmax(sims[i, candidates])]))
    macro, per_sp = macro_per_species_from_preds(preds, np.asarray(query_labels_global))
    return {
        "macro": macro,
        "per_species": per_sp,
        "preds": preds,
        "top_idx": np.asarray(evidence_idx, dtype=int),
        "class_order": class_order,
    }


def method_eval(name, md, q_labels):
    if md.get("scurd"):
        global query_labels_global
        query_labels_global = q_labels
        return scurd_scores(md["id"], md["gal"], md["gal_labels"], mode=SCURD_MODE)
    return nearest_neighbor_eval(md["id"], q_labels, md["gal"], md["gal_labels"])


def _scurd_full_gallery(sc, emb, gallery_mask):
    """SC-URD gallery embeddings for the full 954-species gallery.

    Prefers the aligned `swi_pool` projected by
    swid_retrieval.embeddings.build_full954.project_scurd_full_pool(); falls back
    to projecting from the SC-URD checkpoint if that key is absent/misaligned.
    """
    swi_dinov2 = emb["embs_swi_dinov2"]
    if "swi_pool" in sc.files and len(sc["swi_pool"]) == len(swi_dinov2):
        return sc["swi_pool"][gallery_mask]
    if not SCURD_MAIN_CKPT.exists():
        raise KeyError(
            f"SC-URD projected cache lacks an aligned 'swi_pool' and the checkpoint is "
            f"missing: {SCURD_MAIN_CKPT}. Run build_full954.project_scurd_full_pool() first."
        )
    import torch
    from swid_retrieval.scurd import load_scurd_model, project_np
    device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Projecting full SWI through SC-URD from {SCURD_MAIN_CKPT.name}")
    model, _ = load_scurd_model(SCURD_MAIN_CKPT, in_dim=swi_dinov2.shape[1], device=device)
    return project_np(model, swi_dinov2[gallery_mask], device)


def load_artifacts():
    print(f"ROOT_PATH={ROOT_PATH}")
    print(f"RESULTS_DIR={RESULTS_DIR}")
    print(f"Loading embedding cache: {EMB_CACHE_PATH}")
    emb = np.load(EMB_CACHE_PATH, allow_pickle=False)

    labels_id = np.asarray([canonical_label(x) for x in emb["labels_id_dinov2"]])
    labels_swi = np.asarray([canonical_label(x) for x in emb["labels_swi_dinov2"]])
    labels_ood = np.asarray([canonical_label(x) for x in emb["labels_ood_dinov2"]])
    id_species = set(labels_id)
    if IS_FULL_GALLERY:
        gallery_mask = np.ones(len(labels_swi), dtype=bool)
        scope_label = "full_swi"
    else:
        gallery_mask = np.asarray([x in id_species for x in labels_swi])
        scope_label = "id_only"
    gal_labels = labels_swi[gallery_mask]

    print(f"ID: {len(labels_id)} images, {len(set(labels_id))} species")
    print(f"SWI gallery scope={scope_label}: {gallery_mask.sum()} images, {len(set(gal_labels))} species")
    print(f"OOD: {len(labels_ood)} images, {len(set(labels_ood))} species")
    if scope_label == "full_swi" and len(set(gal_labels)) < MIN_FULL_GALLERY_SPECIES:
        raise RuntimeError(
            f"Full-gallery protocol violation: only {len(set(gal_labels))} gallery species "
            f"(expected >= {MIN_FULL_GALLERY_SPECIES}). Point EMB_CACHE_NAME at the full-954 cache."
        )

    methods = {
        "DINOv2": {
            "id": emb["embs_id_dinov2"],
            "gal": emb["embs_swi_dinov2"][gallery_mask],
            "ood": emb["embs_ood_dinov2"],
            "gal_labels": gal_labels,
        },
        "ArcFace-557": {
            "id": emb["embs_id_arc"],
            "gal": emb["embs_swi_arc"][gallery_mask],
            "ood": emb["embs_ood_arc"],
            "gal_labels": gal_labels,
        },
        "CE-Full": {
            "id": emb["embs_id_ce_full_norm"],
            "gal": emb["embs_swi_ce_full_norm"][gallery_mask],
            "ood": emb["embs_ood_ce_full_norm"],
            "gal_labels": gal_labels,
        },
        "Fusion": {
            "id": fuse(emb["embs_id_arc"], emb["embs_id_dinov2"]),
            "gal": fuse(emb["embs_swi_arc"][gallery_mask], emb["embs_swi_dinov2"][gallery_mask]),
            "ood": fuse(emb["embs_ood_arc"], emb["embs_ood_dinov2"]),
            "gal_labels": gal_labels,
        },
    }

    if SCURD_PROJ_CACHE.exists():
        print(f"Loading SC-URD projected cache: {SCURD_PROJ_CACHE}")
        sc = np.load(SCURD_PROJ_CACHE, allow_pickle=False)
        sc_gal = _scurd_full_gallery(sc, emb, gallery_mask) if IS_FULL_GALLERY else sc["gal"]
        methods["SC-URD"] = {
            "id": sc["id"],
            "gal": sc_gal,
            "ood": sc["ood"],
            "gal_labels": gal_labels,
            "scurd": True,
        }
    else:
        print(f"WARNING: missing SC-URD projected cache: {SCURD_PROJ_CACHE}")

    paths = load_paths_for_interpretability(gallery_mask)
    return emb, methods, labels_id, labels_swi, labels_ood, gal_labels, gallery_mask, paths


def load_paths_for_interpretability(gallery_mask):
    """Return path arrays aligned with embedding cache extraction order."""
    out = {"id_paths": None, "swi_paths": None, "gal_paths": None}
    id_csv = ROOT_PATH / "ID_images_expanded.csv"
    manifest_path = ROOT_PATH / "swi_manifest.json"
    if id_csv.exists():
        id_df = pd.read_csv(id_csv)
        out["id_paths"] = id_df["file_path"].astype(str).values
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        # Must match the embedding-cache extraction order: the full-954 cache is
        # extracted over meta-train ∪ meta-val ∪ meta-test, the legacy cache over
        # meta-test only. Misalignment here silently corrupts confusion/failure rows.
        if IS_FULL_GALLERY:
            swi_items = manifest["meta-train"] + manifest["meta-val"] + manifest["meta-test"]
        else:
            swi_items = manifest["meta-test"]
        out["swi_paths"] = np.asarray([p for p, _ in swi_items])
        out["gal_paths"] = out["swi_paths"][gallery_mask]
    return out


def build_statistical_cleanup(method_evals, labels_id, rq2, rq3, rq4):
    rows = []

    # RQ1: recomputed per-species from embeddings, paired by species.
    sc = method_evals.get("SC-URD")
    for base in ["DINOv2", "Fusion", "ArcFace-557", "CE-Full"]:
        if sc is None or base not in method_evals:
            continue
        a = sc["per_species"]
        b = method_evals[base]["per_species"]
        wa = paired_wilcoxon(a, b)
        rows.append({
            "claim": "RQ1 cross-domain public ID retrieval",
            "unit": "species",
            "method_a": "SC-URD",
            "method_b": base,
            "mean_a": sc["macro"],
            "mean_b": method_evals[base]["macro"],
            "delta_a_minus_b": wa["delta"],
            "p_value": wa["p_value"],
            "n": wa["n"],
            "ci_a": bootstrap_mean_ci(a.values()),
            "ci_b": bootstrap_mean_ci(b.values()),
        })

    # RQ2: use existing bootstrap CI from JSON.
    ood = rq2["ood_detection"]
    for base in ["SupCon", "Fusion", "ArcFace-557", "DINOv2", "CE-Full"]:
        if "SC-URD" not in ood or base not in ood:
            continue
        rows.append({
            "claim": "RQ2 OOD detection AUROC",
            "unit": "bootstrap images/species as in final_research.py",
            "method_a": "SC-URD",
            "method_b": base,
            "mean_a": ood["SC-URD"]["AUROC"]["mean"],
            "mean_b": ood[base]["AUROC"]["mean"],
            "delta_a_minus_b": ood["SC-URD"]["AUROC"]["mean"] - ood[base]["AUROC"]["mean"],
            "p_value": None,
            "n": None,
            "ci_a": {
                "mean": ood["SC-URD"]["AUROC"]["mean"],
                "ci_lo": ood["SC-URD"]["AUROC"]["ci_lo"],
                "ci_hi": ood["SC-URD"]["AUROC"]["ci_hi"],
            },
            "ci_b": {
                "mean": ood[base]["AUROC"]["mean"],
                "ci_lo": ood[base]["AUROC"]["ci_lo"],
                "ci_hi": ood[base]["AUROC"]["ci_hi"],
            },
        })

    # RQ3: old retention after gallery expansion.
    ge = rq3["gallery_extension"]
    if "SC-URD" in ge:
        for base in ["DINOv2", "Fusion", "ArcFace-557", "ArcFace+CORAL", "CE-Full"]:
            if base not in ge:
                continue
            a = ge["SC-URD"]["old_per_species"]
            b = ge[base]["old_per_species"]
            wa = paired_wilcoxon(a, b)
            rows.append({
                "claim": "RQ3 old-species retention after K=10 gallery expansion",
                "unit": "species",
                "method_a": "SC-URD",
                "method_b": base,
                "mean_a": ge["SC-URD"]["old_id_mean"],
                "mean_b": ge[base]["old_id_mean"],
                "delta_a_minus_b": wa["delta"],
                "p_value": wa["p_value"],
                "n": wa["n"],
                "ci_a": bootstrap_mean_ci(a.values()),
                "ci_b": bootstrap_mean_ci(b.values()),
            })

    # RQ4: SWI pool -> VN26 all.
    cd = rq4["cross_domain"]
    if "SC-URD" in cd and "SWI_pool" in cd["SC-URD"]:
        sc_row = cd["SC-URD"]["SWI_pool"]["VN26_all"]
        for base in ["DINOv2", "Fusion", "ArcFace-557", "CE-Full"]:
            if base not in cd or "SWI_pool" not in cd[base] or "VN26_all" not in cd[base]["SWI_pool"]:
                continue
            br = cd[base]["SWI_pool"]["VN26_all"]
            wa = paired_wilcoxon(sc_row["per_species"], br["per_species"])
            rows.append({
                "claim": "RQ4 SWI-pool to VN26-all cross-domain generalization",
                "unit": "species",
                "method_a": "SC-URD",
                "method_b": base,
                "mean_a": sc_row["mean"],
                "mean_b": br["mean"],
                "delta_a_minus_b": wa["delta"],
                "p_value": wa["p_value"],
                "n": wa["n"],
                "ci_a": {
                    "mean": sc_row["mean"],
                    "ci_lo": sc_row["ci_lo"],
                    "ci_hi": sc_row["ci_hi"],
                },
                "ci_b": {
                    "mean": br["mean"],
                    "ci_lo": br["ci_lo"],
                    "ci_hi": br["ci_hi"],
                },
            })

    rows = add_holm_adjusted_p(rows)
    save_json(rows, "main_claim_statistical_cleanup.json")
    flat = []
    for r in rows:
        flat.append({
            "claim": r["claim"],
            "unit": r["unit"],
            "method_a": r["method_a"],
            "method_b": r["method_b"],
            "mean_a": r["mean_a"],
            "mean_b": r["mean_b"],
            "delta_a_minus_b": r["delta_a_minus_b"],
            "p_value": r["p_value"],
            "p_holm": r.get("p_holm"),
            "n": r["n"],
            "ci_a_lo": None if r["ci_a"] is None else r["ci_a"].get("ci_lo"),
            "ci_a_hi": None if r["ci_a"] is None else r["ci_a"].get("ci_hi"),
            "ci_b_lo": None if r["ci_b"] is None else r["ci_b"].get("ci_lo"),
            "ci_b_hi": None if r["ci_b"] is None else r["ci_b"].get("ci_hi"),
        })
    save_csv(flat, "main_claim_statistical_cleanup.csv")
    return rows


def build_failure_taxonomy(method_evals, labels_id, paths):
    # Per-species accuracy table.
    species = sorted(np.unique(labels_id))
    rows = []
    for sp in species:
        row = {"species": sp}
        for name, ev in method_evals.items():
            row[name] = ev["per_species"].get(sp, np.nan)
        row["SCURD_minus_DINOv2"] = row.get("SC-URD", np.nan) - row.get("DINOv2", np.nan)
        row["SCURD_minus_Fusion"] = row.get("SC-URD", np.nan) - row.get("Fusion", np.nan)
        row["SCURD_minus_ArcFace557"] = row.get("SC-URD", np.nan) - row.get("ArcFace-557", np.nan)
        rows.append(row)
    per_df = save_csv(rows, "rq1_per_species_accuracy.csv")

    # Confusion/failure pairs for each method.
    conf_rows = []
    for name, ev in method_evals.items():
        preds = np.asarray(ev["preds"])
        top_idx = np.asarray(ev["top_idx"])
        for true_sp in species:
            m = labels_id == true_sp
            wrong = preds[m] != true_sp
            if not wrong.any():
                continue
            pred_counts = Counter(preds[m][wrong])
            for pred_sp, count in pred_counts.most_common(5):
                pair_mask = m & (preds == pred_sp)
                q_global = np.where(pair_mask)[0][0]
                conf_rows.append({
                    "method": name,
                    "true_species": true_sp,
                    "predicted_species": pred_sp,
                    "count": int(count),
                    "n_true_images": int(m.sum()),
                    "error_rate_for_true_species": float(count / m.sum()),
                    "example_query_index": int(q_global),
                    "example_gallery_index": int(top_idx[q_global]),
                    "example_query_path": None if paths["id_paths"] is None else str(paths["id_paths"][q_global]),
                    "example_gallery_path": None if paths["gal_paths"] is None else str(paths["gal_paths"][top_idx[q_global]]),
                })
    conf_df = save_csv(conf_rows, "rq1_confusion_pairs_top5.csv")

    # Reviewer-friendly taxonomy summary.
    hard = []
    sc = per_df.set_index("species")
    if "SC-URD" in sc.columns:
        for sp, row in sc.sort_values("SC-URD").head(10).iterrows():
            hard.append({
                "taxonomy": "SC-URD hard species",
                "species": sp,
                "scurd_acc": row["SC-URD"],
                "dinov2_acc": row.get("DINOv2"),
                "fusion_acc": row.get("Fusion"),
                "likely_review_angle": "inspect acquisition/domain shift, insufficient gallery diversity, or visually similar anatomical texture",
            })
        for sp, row in sc.sort_values("SCURD_minus_DINOv2", ascending=False).head(10).iterrows():
            hard.append({
                "taxonomy": "largest SC-URD gain over DINOv2",
                "species": sp,
                "scurd_acc": row["SC-URD"],
                "dinov2_acc": row.get("DINOv2"),
                "delta": row["SCURD_minus_DINOv2"],
                "likely_review_angle": "SC-URD calibration appears to reduce cross-domain mismatch for this species",
            })
        for sp, row in sc.sort_values("SCURD_minus_Fusion").head(10).iterrows():
            hard.append({
                "taxonomy": "largest Fusion advantage over SC-URD",
                "species": sp,
                "scurd_acc": row["SC-URD"],
                "fusion_acc": row.get("Fusion"),
                "delta": row["SCURD_minus_Fusion"],
                "likely_review_angle": "dense/local nearest-neighbor evidence may be better than calibrated class scoring",
            })
    save_csv(hard, "rq1_failure_taxonomy_summary.csv")

    return per_df, conf_df


def write_method_narrative_recommendation():
    text = """# Simplified Method Narrative Recommendation

Use this as reviewer-safe framing in the main paper.

## Main method

Present only one default configuration in the Method section:

**SC-URD = DINOv2 features + residual projection head + gallery-centered retrieval calibration.**

The main inference mode should be `scurd_r01_e20 + centered`.

## Keep in main text

1. DINOv2 extracts open-set visual features.
2. A residual head adapts features toward wood retrieval.
3. Gallery centering calibrates query/gallery similarities at inference time.
4. The same embedding space supports ID retrieval, OOD distance scoring, and gallery expansion.

## Move to ablation/appendix

- raw
- logmeanexp
- prototype_mix
- centered_logmeanexp
- centered_prototype_mix
- adaptive mode
- alpha and hybrid variants
- density gate

## Claim wording

Prefer:

> SC-URD is a deployment-oriented open-set retrieval framework that improves
> cross-domain wood identification, OOD rejection, and old-species retention
> under gallery expansion.

Avoid:

> SC-URD is a universally best new architecture for wood identification.
"""
    path = OUT_DIR / "simplified_method_narrative.md"
    path.write_text(text)
    print(f"saved {path}")


def maybe_run_interpretability(methods, labels_id, paths):
    if os.environ.get("RUN_INTERPRETABILITY", "0") != "1":
        print("RUN_INTERPRETABILITY=0, skipping occlusion figures.")
        return
    if paths["id_paths"] is None or paths["gal_paths"] is None:
        print("Missing image paths; cannot run interpretability.")
        return
    if "SC-URD" not in methods:
        print("Missing SC-URD method; cannot run interpretability.")
        return

    # Imports are delayed so the default post-hoc tables do not require GPU libs.
    import cv2
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class SCURDResidualHead(nn.Module):
        def __init__(self, in_dim, out_dim=512, beta=0.1, learnable_beta=False):
            super().__init__()
            self.base = nn.Linear(in_dim, out_dim, bias=False)
            self.adapter = nn.Sequential(
                nn.Linear(in_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1024, out_dim),
            )
            self.register_buffer("fixed_beta", torch.tensor(float(beta), dtype=torch.float32))
            self.logit_beta = None

        def beta_value(self):
            return self.fixed_beta

        def forward(self, x):
            base = F.normalize(self.base(x), dim=1)
            delta = F.normalize(self.adapter(x), dim=1)
            beta = self.beta_value().to(x.device)
            return F.normalize(base + beta * delta, dim=1)

    ckpt_path = RESULTS_DIR / "research_directions" / "sc_urd_checkpoint_scurd_r01_e20_v2.pt"
    if not ckpt_path.exists():
        print(f"Missing checkpoint: {ckpt_path}")
        return

    print("Loading DINOv2 and SC-URD head for occlusion interpretability...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    head = SCURDResidualHead(ckpt.get("in_dim", 768), out_dim=512, beta=ckpt.get("beta", 0.1)).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()

    tf = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def embed_pil(img):
        x = tf(img).unsqueeze(0).to(device)
        z = dino(x).float()
        return head(z).cpu().numpy()[0]

    def occlusion_heatmap(query_path, target_emb, grid=7):
        img = Image.open(query_path).convert("RGB")
        arr = np.asarray(img).copy()
        base = embed_pil(img)
        base_sim = float(norm(base[None]) @ norm(target_emb[None]).T)
        h, w = arr.shape[:2]
        heat = np.zeros((grid, grid), dtype=np.float32)
        fill = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        for gy in range(grid):
            for gx in range(grid):
                oc = arr.copy()
                y0, y1 = int(gy * h / grid), int((gy + 1) * h / grid)
                x0, x1 = int(gx * w / grid), int((gx + 1) * w / grid)
                oc[y0:y1, x0:x1] = fill
                sim = float(norm(embed_pil(Image.fromarray(oc))[None]) @ norm(target_emb[None]).T)
                heat[gy, gx] = max(0.0, base_sim - sim)
        if heat.max() > 0:
            heat /= heat.max()
        heat_big = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
        return arr, heat_big, base_sim

    ev = method_eval("SC-URD", methods["SC-URD"], labels_id)
    correct = np.where(ev["preds"] == labels_id)[0]
    wrong = np.where(ev["preds"] != labels_id)[0]
    rng = np.random.RandomState(RNG_SEED)
    n_cases = int(os.environ.get("N_INTERP_CASES", "6"))
    chosen = []
    if len(correct):
        chosen.extend(rng.choice(correct, size=min(n_cases // 2, len(correct)), replace=False).tolist())
    if len(wrong):
        chosen.extend(rng.choice(wrong, size=min(n_cases - len(chosen), len(wrong)), replace=False).tolist())

    rows = []
    for rank, qi in enumerate(chosen):
        gi = int(ev["top_idx"][qi])
        q_path = paths["id_paths"][qi]
        g_path = paths["gal_paths"][gi]
        target = methods["SC-URD"]["gal"][gi]
        arr, heat, base_sim = occlusion_heatmap(q_path, target)
        gal_img = np.asarray(Image.open(g_path).convert("RGB"))

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        axes[0].imshow(arr)
        axes[0].set_title(f"Query\n{labels_id[qi]}", fontsize=8)
        axes[1].imshow(gal_img)
        axes[1].set_title(f"Retrieved\n{ev['preds'][qi]}", fontsize=8)
        axes[2].imshow(arr)
        axes[2].imshow(heat, cmap="magma", alpha=0.45)
        axes[2].set_title("Similarity occlusion\nhigher = more important", fontsize=8)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out = FIG_DIR / f"scurd_occlusion_case_{rank:02d}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        rows.append({
            "case": rank,
            "correct": bool(ev["preds"][qi] == labels_id[qi]),
            "true_species": labels_id[qi],
            "retrieved_species": str(ev["preds"][qi]),
            "query_path": str(q_path),
            "retrieved_gallery_path": str(g_path),
            "baseline_similarity": base_sim,
            "figure": str(out),
            "interpretation_note": (
                "Qualitative retrieval saliency only: it shows query regions whose occlusion "
                "reduces similarity to the retrieved gallery embedding. It is not a proof of "
                "vessel/ray/parenchyma localization without anatomical annotations."
            ),
        })
    save_csv(rows, "interpretability_occlusion_cases.csv")


def main():
    rq2 = load_json("rq2_ood_openset.json")
    rq3 = load_json("rq3_adaptation.json")
    rq4 = load_json("rq4_generalization.json")

    _, methods, labels_id, _, _, _, _, paths = load_artifacts()
    method_evals = {name: method_eval(name, md, labels_id) for name, md in methods.items()}

    print("\nRQ1 recomputed macro R@1 for review evidence")
    for name, ev in sorted(method_evals.items(), key=lambda kv: kv[1]["macro"], reverse=True):
        print(f"  {name:12s} {ev['macro']:.4f}")

    build_statistical_cleanup(method_evals, labels_id, rq2, rq3, rq4)
    build_failure_taxonomy(method_evals, labels_id, paths)
    write_method_narrative_recommendation()
    maybe_run_interpretability(methods, labels_id, paths)

    manifest = {
        "outputs_dir": str(OUT_DIR),
        "figures_dir": str(FIG_DIR),
        "gallery_scope": GALLERY_SCOPE,
        "is_full_swi_gallery": IS_FULL_GALLERY,
        "min_full_gallery_species": MIN_FULL_GALLERY_SPECIES,
        "inputs": {
            "embedding_cache": str(EMB_CACHE_PATH),
            "scurd_projected_cache": str(SCURD_PROJ_CACHE),
            "rq2": str(SOURCE_RESULTS_DIR / "rq2_ood_openset.json"),
            "rq3": str(SOURCE_RESULTS_DIR / "rq3_adaptation.json"),
            "rq4": str(SOURCE_RESULTS_DIR / "rq4_generalization.json"),
        },
        "notes": [
            "This script does not rerun training or RQ experiments.",
            "Interpretability is optional and qualitative; use RUN_INTERPRETABILITY=1.",
            "Per-species/failure taxonomy is retrieval-native and should be preferred over classifier CAM for SC-URD.",
        ],
    }
    save_json(manifest, "review_evidence_manifest.json")
    print("\nDone.")


if __name__ == "__main__":
    main()

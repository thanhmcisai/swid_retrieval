# -*- coding: utf-8 -*-
"""Reference-collection geometry diagnostics for the submission reframe.

This experiment is intentionally cache-only: it uses the already extracted
embeddings from the active deployment registry and does not read images or train
models. It quantifies why wood ID is better framed as reference-collection
matching:

  GEO-1: cross-domain same-species spread versus same-genus / different-genus
         separation in several embedding spaces.
  GEO-2: per-species geometry versus retrieval consequence.
  GEO-3: species-level versus genus-level retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import _norm, method_retrieval_eval, scurd_class_scores


METHODS = ("CE-Full", "DINOv2", "SC-URD")


def _genus(labels):
    return np.asarray([str(x).split("_")[0] for x in labels])


def _sample_idx(mask, rng, max_n):
    idx = np.flatnonzero(mask)
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return idx


def _cos_dist(a, b):
    return 1.0 - (_norm(a) @ _norm(b).T)


def _auc_higher_score_positive(pos_scores, neg_scores):
    """Mann-Whitney AUC without sklearn. Higher score = positive class."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks for ties.
    vals, start, counts = np.unique(scores[order], return_index=True, return_counts=True)
    for s, c in zip(start, counts):
        if c > 1:
            tied = order[s:s + c]
            ranks[tied] = ranks[tied].mean()
    rank_sum_pos = ranks[:len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _top1_preds(md, q_emb, g_emb, g_labels):
    if md.get("scurd_mode"):
        scores, classes = scurd_class_scores(
            q_emb,
            g_emb,
            g_labels,
            mode=md["scurd_mode"],
            prototype_mix_alpha=md.get("scurd_alpha"),
        )
        return classes[np.argmax(scores, axis=1)]
    sims = _norm(q_emb) @ _norm(g_emb).T
    return np.asarray(g_labels)[np.argmax(sims, axis=1)]


def _per_species_accuracy(preds, labels):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    return {sp: float((preds[labels == sp] == sp).mean()) for sp in sorted(set(labels))}


def _spearman(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan, int(len(df))
    return float(df["x"].rank().corr(df["y"].rank())), int(len(df))


def _geometry_for_method(name, md, labels_id, rng, max_q=40, max_same=80, max_inter=160):
    q_emb = md["id"]
    g_emb, g_labels = md["gal"]
    g_labels = np.asarray(g_labels)
    q_labels = np.asarray(labels_id)
    q_genus, g_genus = _genus(q_labels), _genus(g_labels)

    same_d, same_genus_d, diff_genus_d = [], [], []
    species_rows = []
    for sp in sorted(set(q_labels)):
        qi = _sample_idx(q_labels == sp, rng, max_q)
        gi_same = _sample_idx(g_labels == sp, rng, max_same)
        if len(qi) == 0 or len(gi_same) == 0:
            continue
        genus = str(sp).split("_")[0]
        gi_sg = _sample_idx((g_genus == genus) & (g_labels != sp), rng, max_inter)
        gi_dg = _sample_idx(g_genus != genus, rng, max_inter)

        d_same = _cos_dist(q_emb[qi], g_emb[gi_same]).reshape(-1)
        d_sg = _cos_dist(q_emb[qi], g_emb[gi_sg]).reshape(-1) if len(gi_sg) else np.array([])
        d_dg = _cos_dist(q_emb[qi], g_emb[gi_dg]).reshape(-1) if len(gi_dg) else np.array([])
        same_d.extend(d_same.tolist())
        same_genus_d.extend(d_sg.tolist())
        diff_genus_d.extend(d_dg.tolist())
        species_rows.append({
            "method": name,
            "species": sp,
            "genus": genus,
            "same_species_mean_distance": float(np.mean(d_same)),
            "same_genus_inter_mean_distance": float(np.mean(d_sg)) if len(d_sg) else np.nan,
            "different_genus_inter_mean_distance": float(np.mean(d_dg)) if len(d_dg) else np.nan,
            "same_genus_margin": float(np.mean(d_sg) - np.mean(d_same)) if len(d_sg) else np.nan,
            "different_genus_margin": float(np.mean(d_dg) - np.mean(d_same)) if len(d_dg) else np.nan,
            "n_query": int(len(qi)),
            "n_same_gallery": int(len(gi_same)),
            "n_same_genus_gallery": int(len(gi_sg)),
        })

    same_d = np.asarray(same_d)
    same_genus_d = np.asarray(same_genus_d)
    diff_genus_d = np.asarray(diff_genus_d)
    inter = np.concatenate([same_genus_d, diff_genus_d]) if len(diff_genus_d) else same_genus_d
    overlap = float(np.mean(same_d >= np.percentile(inter, 5))) if len(same_d) and len(inter) else np.nan
    summary = {
        "method": name,
        "same_species_mean_distance": float(np.mean(same_d)) if len(same_d) else np.nan,
        "same_genus_inter_mean_distance": float(np.mean(same_genus_d)) if len(same_genus_d) else np.nan,
        "different_genus_inter_mean_distance": float(np.mean(diff_genus_d)) if len(diff_genus_d) else np.nan,
        "same_genus_margin": float(np.mean(same_genus_d) - np.mean(same_d)) if len(same_genus_d) and len(same_d) else np.nan,
        "different_genus_margin": float(np.mean(diff_genus_d) - np.mean(same_d)) if len(diff_genus_d) and len(same_d) else np.nan,
        "same_vs_inter_auc": _auc_higher_score_positive(-same_d, -inter) if len(same_d) and len(inter) else np.nan,
        "tail_overlap_same_species_vs_inter": overlap,
        "n_same_pairs": int(len(same_d)),
        "n_same_genus_pairs": int(len(same_genus_d)),
        "n_different_genus_pairs": int(len(diff_genus_d)),
    }
    return summary, species_rows


def _retrieval_axes(M, labels_id):
    rows = []
    for name in [m for m in ("CE-Full", "DINOv2", "Fusion", "SC-URD") if m in M]:
        md = M[name]
        q = md["id"]
        g, gl = md["gal"]
        preds = _top1_preds(md, q, g, gl)
        species_acc = float(np.mean(preds == labels_id))
        genus_acc = float(np.mean(_genus(preds) == _genus(labels_id)))
        macro_sp = np.mean(list(_per_species_accuracy(preds, labels_id).values()))
        genus_labels = _genus(labels_id)
        macro_gen = np.mean([
            float((_genus(preds)[genus_labels == ge] == ge).mean())
            for ge in sorted(set(genus_labels))
        ])
        rows.append({
            "method": name,
            "species_micro_r1": species_acc,
            "species_macro_r1": float(macro_sp),
            "genus_micro_r1": genus_acc,
            "genus_macro_r1": float(macro_gen),
            "genus_minus_species_macro": float(macro_gen - macro_sp),
        })
    return rows


def run(M, ctx, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    labels_id = np.asarray(ctx["labels_id"])

    summary_rows, species_rows = [], []
    for name in [m for m in METHODS if m in M]:
        summary, rows = _geometry_for_method(name, M[name], labels_id, rng)
        summary_rows.append(summary)
        species_rows.extend(rows)

    geom_summary = pd.DataFrame(summary_rows)
    geom_species = pd.DataFrame(species_rows)

    # Deployment consequence: species geometry versus per-species accuracy/gain.
    consequence = []
    acc_by_method = {}
    for name in [m for m in ("CE-Full", "DINOv2", "SC-URD") if m in M]:
        md = M[name]
        preds = _top1_preds(md, md["id"], md["gal"][0], md["gal"][1])
        acc_by_method[name] = _per_species_accuracy(preds, labels_id)
    for _, row in geom_species.iterrows():
        sp = row["species"]
        rec = row.to_dict()
        for name, acc in acc_by_method.items():
            rec[f"{name}_species_r1"] = acc.get(sp, np.nan)
        if "SC-URD" in acc_by_method and "DINOv2" in acc_by_method:
            rec["SC-URD_minus_DINOv2_species_r1"] = acc_by_method["SC-URD"].get(sp, np.nan) - acc_by_method["DINOv2"].get(sp, np.nan)
        if "SC-URD" in acc_by_method and "CE-Full" in acc_by_method:
            rec["SC-URD_minus_CE-Full_species_r1"] = acc_by_method["SC-URD"].get(sp, np.nan) - acc_by_method["CE-Full"].get(sp, np.nan)
        consequence.append(rec)
    geom_conseq = pd.DataFrame(consequence)

    corr_rows = []
    for method in sorted(geom_conseq["method"].unique()) if not geom_conseq.empty else []:
        sub = geom_conseq[geom_conseq["method"] == method]
        for geom_col in ("same_species_mean_distance", "same_genus_margin", "different_genus_margin"):
            for target in ("CE-Full_species_r1", "SC-URD_species_r1", "SC-URD_minus_DINOv2_species_r1"):
                if geom_col in sub and target in sub:
                    rho, n = _spearman(sub[geom_col], sub[target])
                    corr_rows.append({"embedding_space": method, "geometry": geom_col, "target": target, "spearman_r": rho, "n_species": n})
    geom_corr = pd.DataFrame(corr_rows)
    genus = pd.DataFrame(_retrieval_axes(M, labels_id))

    paths = {
        "class_geometry_summary": out_dir / "class_geometry_summary.csv",
        "class_geometry_species": out_dir / "class_geometry_species.csv",
        "class_geometry_correlations": out_dir / "class_geometry_correlations.csv",
        "genus_level_retrieval": out_dir / "genus_level_retrieval.csv",
        "class_geometry": out_dir / "class_geometry.json",
    }
    geom_summary.to_csv(paths["class_geometry_summary"], index=False)
    geom_species.to_csv(paths["class_geometry_species"], index=False)
    geom_corr.to_csv(paths["class_geometry_correlations"], index=False)
    genus.to_csv(paths["genus_level_retrieval"], index=False)
    payload = {
        "gallery_scope": ctx.get("scope"),
        "methods": [m for m in METHODS if m in M],
        "summary": geom_summary.to_dict(orient="records"),
        "correlations": geom_corr.to_dict(orient="records"),
        "genus_level": genus.to_dict(orient="records"),
        "notes": (
            "Distances are cosine distances between public-ID queries and the active "
            "reference gallery. Same-species distances estimate cross-domain class "
            "spread; same-genus and different-genus distances estimate taxonomic "
            "separation. A high genus-minus-species gap means retrieval often finds "
            "the correct genus even when species-level identity is ambiguous."
        ),
    }
    paths["class_geometry"].write_text(json.dumps(payload, indent=2, sort_keys=True))
    for p in paths.values():
        print(f"saved {p}")
    return {k: str(v) for k, v in paths.items()}
